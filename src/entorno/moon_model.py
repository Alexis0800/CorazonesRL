"""
Modelos aprendidos de moon_prob (reemplazan la heurística de coeficientes
fijos, antes duplicada en corazones_rllib.py, recomendador.py y
pimc_regret.py). Ver spec:
docs/superpowers/specs/2026-07-06-moon-prob-modelo-aprendido-design.md

Dos modelos, con esquemas de features distintos (información disponible
distinta):
  - "propio": mi propio pozo, mano exacta conocida -- reutiliza el vector
    completo de ObservacionBuilder(dim=332).
  - "rival": ¿un rival específico está armando el pozo? -- SOLO señales
    públicas de ese rival (nunca su mano real).

En ambos casos hay un gate DURO (no aprendido): si cualquier otro jugador ya
capturó puntos en la mano, el pozo del objetivo es 0 exacto -- es una regla
del juego, no algo incierto que el modelo deba aprender.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_V13
from src.entorno.observacion import ObservacionBuilder

DIM_PROPIO = DIM_V13 + 1  # vector v13 completo (moon_prob en 0) + razón bazas-con-puntos
DIM_RIVAL = 272  # ver features_rival() para el desglose exacto de este número


@dataclass
class EntradaBaza:
    """Una baza ya resuelta, para trackear comportamiento (no solo captura final).

    `lider` es None para una baza cerrada por remate ("se lleva el resto") --
    ahí no hay una carta de salida real que analizar.
    """
    lider: Optional[int]
    ganador: int
    tenia_puntos: bool
    lidero_corazon_o_dama: bool


def _alguien_mas_tiene_puntos(motor: MotorCorazones, idx: int) -> bool:
    """Gate duro: si CUALQUIER otro jugador ya capturó puntos, el pozo de
    `idx` es imposible (el pozo exige TODOS los puntos para un solo jugador)."""
    return any(
        j.contar_puntos_bazas() > 0
        for i, j in enumerate(motor.jugadores) if i != idx
    )


def _ganador_parcial(motor: MotorCorazones) -> Optional[int]:
    """Quién va ganando la baza en curso, posiblemente incompleta.

    None si la mesa está vacía (nadie ha jugado esta baza todavía).
    """
    if not motor.mesa:
        return None
    palo_salida = motor.palo_de_salida
    ganador_idx, carta_mas_alta = motor.mesa[0]
    for idx, carta in motor.mesa[1:]:
        if carta.palo == palo_salida and carta.valor > carta_mas_alta.valor:
            ganador_idx, carta_mas_alta = idx, carta
    return ganador_idx


def _ratio_bazas_con_puntos(historial: List[EntradaBaza], idx: int) -> float:
    """Bazas-con-puntos que ganó `idx` / bazas-con-puntos jugadas hasta ahora.

    0/0 -> 0.0: ninguna baza con puntos jugada aún no es señal de nada.
    """
    con_puntos = [h for h in historial if h.tenia_puntos]
    if not con_puntos:
        return 0.0
    ganadas = sum(1 for h in con_puntos if h.ganador == idx)
    return ganadas / len(con_puntos)


def _tasa_lidero_corazon_dama(historial: List[EntradaBaza], idx: int) -> float:
    """De las bazas que `idx` lideró, en qué fracción lideró con corazón o Q♠
    -- señal de intención más fuerte que solo "ganó la baza": liderar con
    corazones sin necesidad es deliberado."""
    lideradas = [h for h in historial if h.lider == idx]
    if not lideradas:
        return 0.0
    return sum(1 for h in lideradas if h.lidero_corazon_o_dama) / len(lideradas)


def features_propio(
    motor: MotorCorazones,
    agente_idx: int,
    vacios: List[set],
    historial: List[EntradaBaza],
    cartas_dadas: List[int],
    cartas_recibidas: List[int],
    puntuacion_historica: List[int],
    puntos_mano_actual: List[int],
    dama_picas_en: Optional[int],
) -> np.ndarray:
    """Features para "mi propio pozo": reutiliza el vector COMPLETO de
    ObservacionBuilder(dim=332) desde mi perspectiva real (mano exacta,
    cementerio, vacíos, memoria del pase v13 -- todo ya implementado), con
    los 2 slots de moon_prob en 0 (son el objetivo a predecir, no pueden ser
    también entrada), más 1 feature bonus: razón bazas-con-puntos que gané.
    """
    builder = ObservacionBuilder(dim=DIM_V13)
    puedo_alimentar = any(
        puntuacion_historica[j] >= 85 for j in range(4) if j != agente_idx
    )
    base = builder.construir(
        motor=motor,
        agente_idx=agente_idx,
        vacios=vacios,
        puntuacion_historica=puntuacion_historica,
        puntos_mano_actual=puntos_mano_actual,
        dama_picas_en=dama_picas_en,
        moon_prob_agente=0.0,
        moon_prob_rival=0.0,
        puedo_alimentar=puedo_alimentar,
        cartas_pasadas=cartas_dadas,
        cartas_recibidas=cartas_recibidas,
    )
    ratio = _ratio_bazas_con_puntos(historial, agente_idx)
    return np.concatenate([base, np.array([ratio], dtype=np.float32)])
