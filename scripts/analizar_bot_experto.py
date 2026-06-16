#!/usr/bin/env python
"""
Análisis detallado de BotExperto vs v7_golden.

Simula N partidas completas y registra métricas por mano:
  - Captura de Q♠ por tipo de agente
  - Intentos y bloqueos de shooting the moon
  - Distribución de puntos por mano
  - Tasa de manos con 0 puntos
  - Win rate por posición (seat)

Uso:
    python scripts/analizar_bot_experto.py
    python scripts/analizar_bot_experto.py --partidas 200 --verbose
    python scripts/analizar_bot_experto.py --modelo models/v7_golden/snapshots/snapshot_0014900000
"""
from __future__ import annotations

import sys
import os
import argparse
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta

_PALO_STR = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
_VALOR_STR = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
    8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A",
}

_MODELO_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "v7_golden", "snapshots", "snapshot_0014900000",
)

# ------------------------------------------------------------------
# Carga del modelo RL
# ------------------------------------------------------------------

def _detectar_vecnorm(ruta_modelo: str) -> Optional[str]:
    """Busca el .pkl de VecNormalize asociado al snapshot."""
    candidatos = [
        ruta_modelo + "_vecnorm.pkl",
        ruta_modelo.replace(".zip", "") + "_vecnorm.pkl",
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None


def cargar_politica_rl(ruta_modelo: str, agente_idx: int = 0):
    """Carga MaskablePPO + VecNormalize, retorna callable (motor, idx, legales) → Carta."""
    from sb3_contrib import MaskablePPO
    from src.agentes.politica_rl import PoliticaSB3

    ruta_clean = ruta_modelo.replace(".zip", "")
    modelo = MaskablePPO.load(ruta_clean, device="cpu")

    # Compatibilidad 190 → 194 dims (modelos v5)
    if modelo.observation_space.shape[0] == 190:
        from src.torneo.elo import _Modelo190Wrapper
        modelo = _Modelo190Wrapper(modelo)

    vecnorm = _detectar_vecnorm(ruta_modelo)
    return PoliticaSB3(modelo, agente_idx, vecnorm)


# ------------------------------------------------------------------
# Estructuras de datos
# ------------------------------------------------------------------

@dataclass
class EstadoMano:
    """Estadísticas de una mano (13 bazas)."""
    puntos: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    q_capturador: int = -1        # quién tomó Q♠ (-1 si no hubo)
    pozo_exitoso: int = -1        # quién hizo pozo (-1 si no hubo)
    pozo_bloqueado: bool = False  # hubo intento de pozo que se frustró


@dataclass
class EstadoPartida:
    """Estadísticas de una partida completa."""
    ganador: int = -1
    manos: List[EstadoMano] = field(default_factory=list)
    puntuaciones_finales: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    num_manos: int = 0


# ------------------------------------------------------------------
# Simulación de partidas
# ------------------------------------------------------------------

def _analizar_mano(motor: MotorCorazones, pts_mano: List[int]) -> EstadoMano:
    """Extrae estadísticas de la mano a partir del estado del motor post-mano."""
    estado = EstadoMano(puntos=list(pts_mano))

    # Identificar capturador de Q♠ y conteo de corazones por jugador
    corazones_por_j: List[int] = [0, 0, 0, 0]
    for i in range(4):
        for carta in motor.jugadores[i].bazas_ganadas:
            if carta.es_dama_de_picas:
                estado.q_capturador = i
            if carta.es_corazon:
                corazones_por_j[i] += 1

    # Detectar pozo: un jugador tiene todos los corazones (13) + Q♠
    total_corazones = sum(corazones_por_j)
    if total_corazones == 13:  # todos los corazones fueron jugados
        ganador_corazones = [i for i in range(4) if corazones_por_j[i] == 13]
        if ganador_corazones:
            shooter = ganador_corazones[0]
            if estado.q_capturador == shooter:
                estado.pozo_exitoso = shooter
            else:
                # Alguien recogió todos los corazones pero otro tiene Q♠
                # → intento de pozo que fue bloqueado
                if corazones_por_j[shooter] >= 9:
                    estado.pozo_bloqueado = True

    # Pozo detectado también por puntuación: el pozo da 0 al shooter y 26 a los demás
    if pts_mano.count(26) == 3 and pts_mano.count(0) == 1:
        estado.pozo_exitoso = pts_mano.index(0)
        estado.pozo_bloqueado = False
    elif pts_mano.count(26) == 2 and 0 in pts_mano:
        # Pozo parcialmente bloqueado o distribución extraña
        pass

    return estado


def simular_partida(
    politicas: Dict[int, object],
    seed: int,
    verbose: bool = False,
) -> EstadoPartida:
    """
    Simula una partida completa de Hearts hasta que un jugador alcanza 100 pts.

    Args:
        politicas: Mapeo {seat_idx: callable(motor, idx, legales) → Carta} para los 4 seats.
        seed: Semilla para reproducibilidad.
        verbose: Si True, imprime bazas detalladas.

    Returns:
        EstadoPartida con estadísticas de la partida.
    """
    random.seed(seed)
    motor = MotorCorazones()
    resultado = EstadoPartida()

    if verbose:
        print(f"\n{'═'*60}")
        print(f"PARTIDA seed={seed}")

    while all(j.puntuacion_historica < 100 for j in motor.jugadores):
        # Limpiar bazas del estado previo
        for j in motor.jugadores:
            j.bazas_ganadas = []

        motor.repartir()
        resultado.num_manos += 1

        if verbose:
            print(f"\n  ── Mano #{resultado.num_manos} ──")

        for _ in range(13):
            for turno in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)
                carta = politicas[idx](motor, idx, legales)
                motor.jugar_carta(idx, carta)

            ganador = motor.resolver_baza()
            if verbose:
                pts_b = sum(c.puntos for _, c in (motor.mesa or []))
                print(f"    → J{ganador} [{pts_b}pts]")

        pts_mano = motor.calcular_puntuacion_mano()
        motor.aplicar_puntuacion()

        estado_mano = _analizar_mano(motor, pts_mano)
        resultado.manos.append(estado_mano)

        if verbose:
            pts_str = " | ".join(
                f"J{i}:{pts_mano[i]}" for i in range(4))
            print(f"  Mano: {pts_str}")
            if estado_mano.pozo_exitoso >= 0:
                print(f"  *** POZO de J{estado_mano.pozo_exitoso}! ***")

    puntuaciones = [motor.jugadores[i].puntuacion_historica for i in range(4)]
    resultado.puntuaciones_finales = puntuaciones
    resultado.ganador = min(range(4), key=lambda i: puntuaciones[i])

    if verbose:
        print(f"\n  FINAL: {puntuaciones}  → J{resultado.ganador} gana")

    return resultado


# ------------------------------------------------------------------
# Análisis y métricas
# ------------------------------------------------------------------

def _etiqueta(seat: int, asignaciones: Dict[int, str]) -> str:
    return asignaciones.get(seat, f"J{seat}")


def calcular_metricas(
    partidas: List[EstadoPartida],
    asignaciones: Dict[int, str],
) -> Dict[str, Dict]:
    """
    Calcula métricas agregadas por tipo de agente.

    Args:
        partidas: Lista de EstadoPartida.
        asignaciones: Mapeo {seat_idx: etiqueta} ("rl", "experto", "bot_c", etc.)

    Returns:
        Dict con métricas por etiqueta.
    """
    etiquetas = list(set(asignaciones.values()))
    metricas: Dict[str, Dict] = {
        et: {
            "victorias": 0,
            "puntos_total": 0,
            "manos_total": 0,
            "manos_cero": 0,
            "q_capturadas": 0,
            "pozos_exitosos": 0,
            "pozos_intentados_fallidos": 0,  # pozo propio bloqueado por otro
            "pozos_bloqueados": 0,           # bloqueó el pozo de otro
            "puntos_por_mano": [],           # para histograma
        }
        for et in etiquetas
    }

    for partida in partidas:
        # Victorias
        ganador_et = asignaciones[partida.ganador]
        metricas[ganador_et]["victorias"] += 1

        for mano in partida.manos:
            for seat, et in asignaciones.items():
                pts = mano.puntos[seat]
                metricas[et]["puntos_total"] += pts
                metricas[et]["manos_total"] += 1
                metricas[et]["puntos_por_mano"].append(pts)
                if pts == 0:
                    metricas[et]["manos_cero"] += 1

            # Q♠
            if mano.q_capturador >= 0:
                et_q = asignaciones[mano.q_capturador]
                metricas[et_q]["q_capturadas"] += 1

            # Pozos
            if mano.pozo_exitoso >= 0:
                et_p = asignaciones[mano.pozo_exitoso]
                metricas[et_p]["pozos_exitosos"] += 1
            elif mano.pozo_bloqueado:
                # Quién bloqueó: el que tomó Q♠ o el que tomó el corazón decisivo
                # Aproximación: el que tomó Q♠ cuando no fue el shooter
                if mano.q_capturador >= 0:
                    et_b = asignaciones[mano.q_capturador]
                    metricas[et_b]["pozos_bloqueados"] += 1

    return metricas


def imprimir_metricas(
    metricas: Dict[str, Dict],
    num_partidas: int,
    asignaciones: Dict[int, str],
) -> None:
    print(f"\n{'═'*72}")
    print(f"ANÁLISIS AGREGADO — {num_partidas} partidas")
    print(f"Configuración: {asignaciones}")
    print(f"{'─'*72}")

    orden = sorted(metricas.keys())

    # Encabezado
    print(f"  {'Agente':<14} {'W%':>6} {'Pts/mano':>10} {'0-pt%':>7} "
          f"{'Q♠%':>6} {'Pozo%':>7} {'BlqPozo':>9}")
    print(f"  {'─'*14} {'─'*6} {'─'*10} {'─'*7} {'─'*6} {'─'*7} {'─'*9}")

    # Calcular totales de manos únicas (una sola vez, no multiplicado por nº seats)
    total_manos_partida = sum(
        metricas[et]["manos_total"]
        for et in orden
    ) // len(orden)  # cada mano se cuenta N veces (una por seat con esa etiqueta)

    for et in orden:
        m = metricas[et]
        n_manos = m["manos_total"]
        if n_manos == 0:
            continue

        seats_et = sum(1 for s, e in asignaciones.items() if e == et)
        manos_reales = n_manos // seats_et if seats_et > 1 else n_manos
        q_manos = total_manos_partida if total_manos_partida > 0 else manos_reales

        win_pct = m["victorias"] / num_partidas
        pts_avg = m["puntos_total"] / n_manos
        cero_pct = m["manos_cero"] / n_manos
        # Q♠: cuántas manos se capturó Q♠ sobre total de manos de la partida
        q_pct = m["q_capturadas"] / q_manos if q_manos > 0 else 0
        pozo_pct = (m["pozos_exitosos"] / q_manos) if q_manos > 0 else 0
        blq_pct = m["pozos_bloqueados"]

        print(f"  {et:<14} {win_pct:>6.1%} {pts_avg:>10.1f} {cero_pct:>7.1%} "
              f"{q_pct:>6.1%} {pozo_pct:>7.1%} {blq_pct:>9d}")

    print(f"{'─'*72}")
    print(f"  (Q♠% = % de manos donde este agente capturó Q♠)")
    print(f"  (0-pt% = manos donde el agente acumuló 0 puntos)")
    print(f"  (BlqPozo = veces que bloqueó el pozo de otro)")

    # Distribución de puntos por mano (histograma simplificado)
    print(f"\n  DISTRIBUCIÓN DE PUNTOS POR MANO:")
    print(f"  {'Agente':<14} {'0':>5} {'1-3':>5} {'4-8':>5} {'9-13':>6} "
          f"{'14-25':>7} {'26':>5}")
    print(f"  {'─'*14} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*7} {'─'*5}")

    for et in orden:
        m = metricas[et]
        hist = m["puntos_por_mano"]
        if not hist:
            continue
        n = len(hist)
        b0 = sum(1 for x in hist if x == 0) / n
        b1 = sum(1 for x in hist if 1 <= x <= 3) / n
        b4 = sum(1 for x in hist if 4 <= x <= 8) / n
        b9 = sum(1 for x in hist if 9 <= x <= 13) / n
        b14 = sum(1 for x in hist if 14 <= x <= 25) / n
        b26 = sum(1 for x in hist if x == 26) / n
        print(f"  {et:<14} {b0:>5.1%} {b1:>5.1%} {b4:>5.1%} {b9:>6.1%} "
              f"{b14:>7.1%} {b26:>5.1%}")

    print(f"{'═'*72}")


def imprimir_fallos_criticos(
    partidas: List[EstadoPartida],
    asignaciones: Dict[int, str],
    max_ejemplos: int = 10,
) -> None:
    """Identifica y muestra los peores casos del BotExperto."""
    seats_experto = [s for s, et in asignaciones.items() if et == "experto"]
    if not seats_experto:
        return

    print(f"\n{'═'*72}")
    print("FALLOS CRÍTICOS DEL BOTEXPERTO")
    print(f"{'─'*72}")

    tipo_fallos: Dict[str, List[Tuple[int, int, int]]] = {
        "Q♠ capturada": [],
        "Pozo enemigo no bloqueado": [],
        "Mano >18 pts": [],
    }

    for i, partida in enumerate(partidas):
        for j, mano in enumerate(partida.manos):
            for seat in seats_experto:
                # Q♠ capturada por BotExperto (no haciendo pozo)
                if mano.q_capturador == seat and mano.pozo_exitoso != seat:
                    tipo_fallos["Q♠ capturada"].append((i, j, seat))

                # Mano con muchos puntos
                if mano.puntos[seat] > 18 and mano.pozo_exitoso != seat:
                    tipo_fallos["Mano >18 pts"].append(
                        (i, j, mano.puntos[seat]))

                # Pozo enemigo no bloqueado
                if mano.pozo_exitoso >= 0 and mano.pozo_exitoso not in seats_experto:
                    tipo_fallos["Pozo enemigo no bloqueado"].append(
                        (i, j, mano.pozo_exitoso))

    for fallo, casos in tipo_fallos.items():
        print(f"\n  [{fallo}]: {len(casos)} ocurrencias")
        for caso in casos[:max_ejemplos]:
            print(f"    Partida #{caso[0]+1}, Mano #{caso[1]+1}: {caso[2:]}")

    print(f"{'═'*72}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Análisis BotExperto vs v7_golden")
    parser.add_argument("--modelo", default=_MODELO_DEFAULT,
                        help="Ruta al snapshot RL (sin .zip)")
    parser.add_argument("--partidas", type=int, default=100,
                        help="Número de partidas a simular")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar detalle de cada baza")
    parser.add_argument("--config", default="rl_vs_experto",
                        choices=["rl_vs_experto", "experto_vs_bots", "rl_vs_bots"],
                        help="Configuración de partida")
    args = parser.parse_args()

    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo

    print(f"Cargando modelo RL: {args.modelo}")
    politica_rl = cargar_politica_rl(args.modelo, agente_idx=0)

    if args.config == "rl_vs_experto":
        # Seat 0: RL, Seat 1: BotExperto, Seat 2: conservador, Seat 3: evasivo
        asignaciones = {0: "rl", 1: "experto", 2: "bot_c", 3: "bot_e"}
        print(f"Configuración: RL(J0) vs BotExperto(J1) vs bots(J2,J3)")

        partidas: List[EstadoPartida] = []
        for seed in range(args.seed, args.seed + args.partidas):
            experto = BotExperto()  # instancia fresca por partida
            politicas = {
                0: politica_rl,
                1: experto,
                2: bot_conservador,
                3: bot_evasivo,
            }
            r = simular_partida(politicas, seed=seed, verbose=args.verbose)
            partidas.append(r)
            if (seed - args.seed + 1) % 20 == 0:
                print(f"  {seed - args.seed + 1}/{args.partidas} partidas...")

    elif args.config == "experto_vs_bots":
        # Seat 0: BotExperto, Seats 1-3: bots
        asignaciones = {0: "experto", 1: "bot_c", 2: "bot_a", 3: "bot_e"}
        print(f"Configuración: BotExperto(J0) vs bots(J1,J2,J3)")

        partidas = []
        for seed in range(args.seed, args.seed + args.partidas):
            experto = BotExperto()
            politicas = {
                0: experto,
                1: bot_conservador,
                2: bot_agresivo,
                3: bot_evasivo,
            }
            r = simular_partida(politicas, seed=seed, verbose=args.verbose)
            partidas.append(r)
            if (seed - args.seed + 1) % 20 == 0:
                print(f"  {seed - args.seed + 1}/{args.partidas} partidas...")

    else:  # rl_vs_bots
        asignaciones = {0: "rl", 1: "bot_c", 2: "bot_a", 3: "bot_e"}
        print(f"Configuración: RL(J0) vs bots(J1,J2,J3)")

        partidas = []
        for seed in range(args.seed, args.seed + args.partidas):
            politicas = {
                0: politica_rl,
                1: bot_conservador,
                2: bot_agresivo,
                3: bot_evasivo,
            }
            r = simular_partida(politicas, seed=seed, verbose=args.verbose)
            partidas.append(r)
            if (seed - args.seed + 1) % 20 == 0:
                print(f"  {seed - args.seed + 1}/{args.partidas} partidas...")

    metricas = calcular_metricas(partidas, asignaciones)
    imprimir_metricas(metricas, args.partidas, asignaciones)
    imprimir_fallos_criticos(partidas, asignaciones)


if __name__ == "__main__":
    main()
