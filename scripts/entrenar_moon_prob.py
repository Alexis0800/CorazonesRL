"""
Genera datasets y entrena los 2 modelos aprendidos de moon_prob
(src/entorno/moon_model.py) a partir de las manos reales reconstruibles de
data/partidas_bridge.jsonl. Ver spec:
docs/superpowers/specs/2026-07-06-moon-prob-modelo-aprendido-design.md

Limitación de datos conocida: la memoria del pase (cartas dadas/recibidas)
SOLO se conoce con certeza para el asiento realmente logueado por el bridge
en cada partida (`partida.asiento_agente`) -- el bridge no registra el
intercambio de los otros 3 asientos. Al generar ejemplos desde las 4
perspectivas por mano, esas features quedan en 0 (sin dato) salvo cuando la
perspectiva evaluada ES ese asiento real. No es un bug: es la limitación
real de los datos disponibles, y coincide con el caso legítimo de "sin
información de pase" que también ocurre en producción (el 4º jugador nunca
tiene relación de pase conmigo).

Uso:
    python scripts/entrenar_moon_prob.py --partidas data/partidas_bridge.jsonl \
        --out-dir models/moon --epocas 300
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.captura.escritor import cargar_partidas
from src.captura.modelos import RegistroMano, RegistroPartida
from src.captura.replay import _preparar_motor, mano_reconstruible
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.moon_model import (
    DIM_PROPIO,
    DIM_RIVAL,
    EntradaBaza,
    _RedMoonMLP,
    _alguien_mas_tiene_puntos,
    features_propio,
    features_rival,
)

_DIRECCION_A_NUMERO_MANO = {"izquierda": 1, "derecha": 2, "enfrente": 3}


def _receptor_y_dador(direccion: Optional[str], seat: int) -> Tuple[Optional[int], Optional[int]]:
    """Índice de asiento receptor/dador del pase de `seat`, o (None, None) sin pase."""
    if direccion is None or direccion not in _DIRECCION_A_NUMERO_MANO:
        return None, None
    m = MotorCorazones()
    m.numero_mano = _DIRECCION_A_NUMERO_MANO[direccion]
    receptor = m.receptor_pase(seat)
    dador = next(d for d in range(4) if m.receptor_pase(d) == seat)
    return receptor, dador


def _lunaseat_de(mano: RegistroMano) -> Optional[int]:
    """Asiento que hizo el pozo (Pleno) en esta mano, o None si no hubo pozo.

    `puntuacion_mano` ya viene con la regla de Pleno aplicada (ver
    `MotorCorazones.calcular_puntuacion_mano` / `manual.py:_aplicar_pleno`):
    el tirador queda en 0 y los otros 3 en 26 -> la suma es 78, NUNCA 26
    (una mano normal siempre suma 26, sin importar cómo se reparten los
    puntos, así que ese caso nunca garantiza que exista un asiento en 0).
    """
    if mano.puntuacion_mano and sum(mano.puntuacion_mano) == 78:
        return mano.puntuacion_mano.index(0)
    return None


def ejemplos_de_mano(mano: RegistroMano, asiento_agente_real: int):
    """(features, label) para el modelo propio y para el rival, por cada
    baza resuelta, desde las 4 perspectivas posibles."""
    motor = _preparar_motor(mano)
    luna_seat = _lunaseat_de(mano)
    receptor_agente, dador_agente = _receptor_y_dador(mano.direccion_pase, asiento_agente_real)

    historial: List[EntradaBaza] = []
    ejemplos_propio = []
    ejemplos_rival = []
    vacios: List[set] = [set() for _ in range(4)]
    mesa_actual: list = []

    for j in mano.jugadas:
        actual = motor.obtener_jugador_actual()
        if actual != j.asiento:
            break  # datos reales inconsistentes (ver pimc_regret_real.py), se descarta el resto
        carta = Carta._TODAS[j.carta_id]
        palo_salida = motor.palo_de_salida

        if not mesa_actual:
            puntos_mano_actual = [jg.contar_puntos_bazas() for jg in motor.jugadores]
            for seat in range(4):
                if _alguien_mas_tiene_puntos(motor, seat):
                    continue
                dadas = mano.pase_dado if seat == asiento_agente_real else []
                recibidas = mano.pase_recibido if seat == asiento_agente_real else []
                feats = features_propio(
                    motor, seat, vacios, historial, dadas, recibidas,
                    [0, 0, 0, 0], puntos_mano_actual, None,
                )
                ejemplos_propio.append((feats, 1.0 if seat == luna_seat else 0.0))

                for rival in range(4):
                    if rival == seat:
                        continue
                    if seat == asiento_agente_real:
                        dadas_r = mano.pase_dado if rival == receptor_agente else []
                        recibidas_r = mano.pase_recibido if rival == dador_agente else []
                    else:
                        dadas_r, recibidas_r = [], []
                    feats_r = features_rival(
                        motor, rival, seat, vacios, historial,
                        dadas_r, recibidas_r, motor.corazones_rotos,
                    )
                    ejemplos_rival.append((feats_r, 1.0 if rival == luna_seat else 0.0))

        if mesa_actual and palo_salida is not None and carta.palo != palo_salida:
            vacios[actual].add(palo_salida)
        motor.jugar_carta(actual, carta)
        mesa_actual.append((actual, carta))
        if len(mesa_actual) == 4:
            ganador = motor.resolver_baza()
            historial.append(EntradaBaza(
                lider=mesa_actual[0][0],
                ganador=ganador,
                tenia_puntos=any(c.puntos > 0 for _, c in mesa_actual),
                lidero_corazon_o_dama=(
                    mesa_actual[0][1].es_corazon or mesa_actual[0][1].es_dama_de_picas
                ),
            ))
            mesa_actual = []

    return ejemplos_propio, ejemplos_rival


def construir_dataset(ruta_partidas: str):
    """Devuelve 2 dicts partida_id -> [(features, label), ...] (uno por modelo)
    y un dict partida_id -> timestamp (para el corte cronológico de validación)."""
    partidas = cargar_partidas(ruta_partidas)
    propio_por_partida: Dict[str, list] = {}
    rival_por_partida: Dict[str, list] = {}
    timestamp_por_partida: Dict[str, str] = {}
    avisos: List[str] = []

    for p in partidas:
        ep: list = []
        er: list = []
        for mano in p.manos:
            if not mano_reconstruible(mano):
                continue
            try:
                e1, e2 = ejemplos_de_mano(mano, p.asiento_agente)
            except ValueError as e:
                avisos.append(f"{p.partida_id}: mano {mano.numero_mano} descartada ({e})")
                continue
            ep.extend(e1)
            er.extend(e2)
        if ep or er:
            propio_por_partida[p.partida_id] = ep
            rival_por_partida[p.partida_id] = er
            timestamp_por_partida[p.partida_id] = p.timestamp

    if avisos:
        print(f"{len(avisos)} manos descartadas por datos inconsistentes:")
        for a in avisos:
            print(f"  - {a}")

    return propio_por_partida, rival_por_partida, timestamp_por_partida
