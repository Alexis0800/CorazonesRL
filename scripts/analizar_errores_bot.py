"""
Compara las decisiones de BotExperto contra PIMC para detectar errores.

Por cada decisión no trivial de BotExperto (>=2 cartas legales), PIMC calcula
el puntaje esperado de TODAS las opciones. Si BotExperto elige peor que PIMC,
se registra como "divergencia" con el coste (penalización extra esperada).

Uso:
    python scripts/analizar_errores_bot.py --partidas 200 --mundos 30
    python scripts/analizar_errores_bot.py --partidas 50 --mundos 50 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dominio.motor import MotorCorazones
from src.agentes.bot_experto import BotExperto
from src.mcts.pimc import _puntaje_esperado_por_carta
from src.mcts.analisis import pimc_exacto
from src.agentes.heuristicos import bot_evasivo


# ──────────────────────────────────────────────────────────────
# Tipos de datos
# ──────────────────────────────────────────────────────────────

@dataclass
class Divergencia:
    baza: int
    situacion: str          # "liderar" | "seguir" | "descartar"
    modo_bot: str           # MINIMIZAR | POZO | BLOQUEAR_POZO | ALIMENTAR
    carta_bot: str
    carta_pimc: str
    score_bot: float
    score_pimc: float
    coste: float            # score_bot - score_pimc (>0 = bot eligió peor)
    num_opciones: int
    exacto: bool = False    # True si se usó enumeración completa


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_PALO_LABEL = {0: "T", 1: "D", 2: "P", 3: "C"}


def _nombre(c) -> str:
    return f"{c.valor}{_PALO_LABEL[c.palo]}"


def _situacion(motor: MotorCorazones, agente: int) -> str:
    if not motor.mesa:
        return "liderar"
    palo = motor.palo_de_salida
    if any(c.palo == palo for c in motor.jugadores[agente].mano):
        return "seguir"
    return "descartar"


def _crear_bots_rollout():
    return {i: bot_evasivo for i in range(4)}


# ──────────────────────────────────────────────────────────────
# Análisis de una mano
# ──────────────────────────────────────────────────────────────

def analizar_mano(
    bots: List[BotExperto],
    num_mundos: int,
    rng: np.random.Generator,
    umbral: float,
) -> tuple[List[Divergencia], int]:
    """
    Ejecuta una mano completa. Para el jugador 0, compara bot vs PIMC
    en cada decisión no trivial.

    Returns (divergencias, total_decisiones_no_triviales).
    """
    motor = MotorCorazones()
    motor.repartir()

    divergencias: List[Divergencia] = []
    decisiones_no_triviales = 0

    while motor._mano_activa:
        if len(motor.mesa) == 4:
            motor.resolver_baza()
            if not motor._mano_activa:
                break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        if not legales:
            break

        if idx == 0:
            # Capturar situación ANTES de que el bot actualice su estado interno
            sit = _situacion(motor, 0)

            # Bot decide (esto actualiza el estado interno del bot pero NO el motor)
            modo = bots[0]._modo(motor, 0)
            carta_bot = bots[0](motor, 0, legales)

            if len(legales) >= 2:
                decisiones_no_triviales += 1
                # Voids conocidos por el jugador 0 (inferidos por BotExperto durante la mano)
                vacios: Dict[int, Set] = {
                    i: bots[0]._vacios[i]
                    for i in range(1, 4)
                    if bots[0]._vacios[i]
                }
                # PIMC evalúa el motor en su estado ACTUAL (antes de jugar carta_bot)
                # Usar pimc_exacto: enumeración completa cuando viable, sampling cuando no
                _, scores, exacto = pimc_exacto(
                    motor, 0, legales,
                    vacios=vacios,
                    rng=rng,
                    crear_bots=_crear_bots_rollout,
                    fallback_mundos=num_mundos,
                )
                carta_optima = min(legales, key=lambda c: scores[c.id])
                score_bot = scores[carta_bot.id]
                score_opt = scores[carta_optima.id]
                coste = score_bot - score_opt

                if coste > umbral:
                    divergencias.append(Divergencia(
                        baza=motor.numero_baza,
                        situacion=sit,
                        modo_bot=modo,
                        carta_bot=_nombre(carta_bot),
                        carta_pimc=_nombre(carta_optima),
                        score_bot=round(score_bot, 2),
                        score_pimc=round(score_opt, 2),
                        coste=round(coste, 2),
                        num_opciones=len(legales),
                        exacto=exacto,
                    ))

            motor.jugar_carta(0, carta_bot)
        else:
            carta = bots[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)

    return divergencias, decisiones_no_triviales


# ──────────────────────────────────────────────────────────────
# Reporte
# ──────────────────────────────────────────────────────────────

def imprimir_reporte(
    todas: List[Divergencia],
    total_partidas: int,
    total_decisiones: int,
    verbose: bool,
) -> None:
    n = len(todas)
    print(f"\n{'='*62}")
    print(f"RESUMEN  ({total_partidas} manos | {total_decisiones} decisiones no triviales)")
    print(f"{'='*62}")

    if not todas:
        print("  No se detectaron divergencias con el umbral dado.")
        return

    coste_total = sum(d.coste for d in todas)
    coste_medio = coste_total / n

    print(f"  Divergencias encontradas : {n:,} ({100*n/total_decisiones:.1f}% de decisiones)")
    print(f"  Coste promedio por error : {coste_medio:.2f} pts esperados extra")
    print(f"  Coste total acumulado    : {coste_total:.1f} pts extra")
    print(f"  Coste medio por mano     : {coste_total/total_partidas:.2f} pts/mano")

    n_exactas = sum(1 for d in todas if d.exacto)
    if n_exactas > 0:
        coste_exacto = sum(d.coste for d in todas if d.exacto)
        print(f"  Decisiones exactas (enum): {n_exactas} | coste exacto: {coste_exacto:.1f} pts")

    # Por baza
    print(f"\n--- Por número de baza ---")
    by_baza: Dict[int, List[float]] = defaultdict(list)
    for d in todas:
        by_baza[d.baza].append(d.coste)
    for baza in sorted(by_baza):
        costes = by_baza[baza]
        bar = "#" * min(30, int(sum(costes) / coste_total * 30 * 5))
        print(f"  Baza {baza:2d}: {len(costes):3d} err | "
              f"med={sum(costes)/len(costes):.2f} | max={max(costes):.2f} | {bar}")

    # Por situación
    print(f"\n--- Por situación ---")
    by_sit: Dict[str, List[float]] = defaultdict(list)
    for d in todas:
        by_sit[d.situacion].append(d.coste)
    for sit, costes in sorted(by_sit.items(), key=lambda x: -sum(x[1])):
        print(f"  {sit:10s}: {len(costes):3d} err | "
              f"coste total={sum(costes):.1f} | med={sum(costes)/len(costes):.2f}")

    # Por modo
    print(f"\n--- Por modo de BotExperto ---")
    by_modo: Dict[str, List[float]] = defaultdict(list)
    for d in todas:
        by_modo[d.modo_bot].append(d.coste)
    for modo, costes in sorted(by_modo.items(), key=lambda x: -sum(x[1])):
        print(f"  {modo:16s}: {len(costes):3d} err | "
              f"coste total={sum(costes):.1f} | med={sum(costes)/len(costes):.2f}")

    # Cartas que el bot elige frecuentemente mal
    print(f"\n--- Cartas elegidas por bot cuando PIMC difiere ---")
    freq_bot: Dict[str, List[float]] = defaultdict(list)
    for d in todas:
        freq_bot[d.carta_bot].append(d.coste)
    top = sorted(freq_bot.items(), key=lambda x: -len(x[1]))[:12]
    for carta, costes in top:
        print(f"  Bot juega {carta:>5}: {len(costes):3d} veces | "
              f"coste med={sum(costes)/len(costes):.2f}")

    # Peores divergencias individuales
    print(f"\n--- 20 peores decisiones (mayor penalización puntual) ---")
    peores = sorted(todas, key=lambda d: -d.coste)[:20]
    hdr = f"  {'Bz':>3} {'Sit':>9} {'Modo':>16} {'Bot':>5} {'PIMC':>5} {'E_bot':>6} {'E_pimc':>6} {'Coste':>6}"
    print(hdr)
    for d in peores:
        print(f"  {d.baza:>3} {d.situacion:>9} {d.modo_bot:>16} "
              f"{d.carta_bot:>5} {d.carta_pimc:>5} "
              f"{d.score_bot:>6.1f} {d.score_pimc:>6.1f} {d.coste:>6.2f}")

    if verbose:
        print(f"\n--- Divergencias en bazas tempranas (1-6) ---")
        tempranas = sorted(
            [d for d in todas if d.baza <= 6],
            key=lambda d: -d.coste,
        )
        for d in tempranas:
            print(f"  B{d.baza:2d} {d.situacion:9} {d.modo_bot:16} "
                  f"bot={d.carta_bot} pimc={d.carta_pimc} coste={d.coste:.2f}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analiza errores de BotExperto usando PIMC como juez"
    )
    parser.add_argument("--partidas", type=int, default=100,
                        help="Manos a analizar (default: 100)")
    parser.add_argument("--mundos", type=int, default=30,
                        help="Mundos PIMC por decisión (default: 30)")
    parser.add_argument("--umbral", type=float, default=0.5,
                        help="Mínimo coste para registrar divergencia (default: 0.5 pts)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar detalle de errores en bazas tempranas")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Analizando {args.partidas} manos | "
          f"PIMC {args.mundos} mundos/dec | umbral={args.umbral} pts")
    t0 = time.time()

    todas: List[Divergencia] = []
    total_decisiones = 0

    for i in range(args.partidas):
        bots = [BotExperto() for _ in range(4)]
        divs, ndec = analizar_mano(bots, args.mundos, rng, args.umbral)
        todas.extend(divs)
        total_decisiones += ndec

        if (i + 1) % max(1, args.partidas // 10) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.partidas - i - 1) / rate
            print(f"  Mano {i+1:4d}/{args.partidas} | "
                  f"Diverg: {len(todas):4d} | "
                  f"ETA: {eta:.0f}s", flush=True)

    print(f"\nCompletado en {time.time()-t0:.1f}s")
    imprimir_reporte(todas, args.partidas, total_decisiones, args.verbose)


if __name__ == "__main__":
    main()
