#!/usr/bin/env python
"""
Análisis profundo de manos con PIMC exacto (enumeración completa).

Cuando la mano llega a baza ≥ 8, el número de distribuciones posibles
de cartas es manejable (~17M en baza 8, ~757K en baza 9). Este script
usa enumeración completa para obtener el ground truth de cada decisión.

Modos:
  perfil   — Análisis ultra-detallado de UNA mano específica
  evaluar  — Evalúa N manos y muestra estadísticas agregadas
  comparar — Compara dos políticas A/B en las mismas manos

Uso:
  python scripts/analisis_profundo.py perfil --seed 42
  python scripts/analisis_profundo.py perfil --seed 42 --politica bot_experto
  python scripts/analisis_profundo.py evaluar --manos 100
  python scripts/analisis_profundo.py comparar --manos 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.mcts.analisis import (
    perfil_mano,
    evaluar_mano,
    comparar_politicas,
    estadisticas_manos,
)
from src.agentes.bot_experto import BotExperto


def _cargar_politica(nombre: str):
    """Carga una política por nombre."""
    if nombre == "bot_experto":
        bot = BotExperto()
        return bot, "BotExperto"

    if nombre.startswith("modelo:"):
        ruta = nombre.split(":", 1)[1]
        from src.agentes.politica_rl import PoliticaSB3
        from sb3_contrib import MaskablePPO
        ruta_clean = ruta.replace(".zip", "")
        modelo = MaskablePPO.load(ruta_clean, device="cpu")
        if modelo.observation_space.shape[0] == 190:
            from src.torneo.elo import _Modelo190Wrapper
            modelo = _Modelo190Wrapper(modelo)
        # Buscar vecnorm
        vecnorm = None
        for c in [ruta + "_vecnorm.pkl", ruta_clean + "_vecnorm.pkl"]:
            if os.path.exists(c):
                vecnorm = c
                break
        pol = PoliticaSB3(modelo, 0, vecnorm)
        nombre_corto = os.path.basename(ruta_clean)
        return pol, f"RL({nombre_corto})"

    raise ValueError(f"Política desconocida: {nombre}")


def cmd_perfil(args):
    """Análisis ultra-detallado de una mano."""
    pol = None
    nombre_pol = "BotExperto"
    if args.politica != "bot_experto":
        pol, nombre_pol = _cargar_politica(args.politica)

    print(f"Política: {nombre_pol}")
    print(f"Rollout: {args.rollout}")

    t0 = time.time()
    perfil = perfil_mano(
        seed=args.seed,
        agente_idx=args.jugador,
        politica_evaluada=pol,
        rollout_tipo=args.rollout,
        verbose=not args.quiet,
    )

    # Resumen final
    if args.quiet:
        n_err = sum(1 for d in perfil.decisiones if d.coste > 0)
        n_exactas = sum(1 for d in perfil.decisiones if d.exacto)
        coste = sum(d.coste for d in perfil.decisiones if d.coste > 0)
        print(f"Seed {args.seed}: {n_err}/{len(perfil.decisiones)} errores | "
              f"Coste: {coste:.1f} | Exactas: {n_exactas}/{len(perfil.decisiones)} | "
              f"Pts: {perfil.puntuacion_final} | "
              f"{time.time()-t0:.1f}s")


def cmd_evaluar(args):
    """Evaluación de N manos."""
    pol = None
    nombre_pol = "BotExperto"
    if args.politica != "bot_experto":
        pol, nombre_pol = _cargar_politica(args.politica)

    print(f"Evaluando {args.manos} manos con {nombre_pol} como J{args.jugador}")
    print(f"Rollout: {args.rollout} | Seed inicial: {args.seed}")

    t0 = time.time()
    stats = estadisticas_manos(
        politica=pol or BotExperto(),
        num_manos=args.manos,
        seed_inicial=args.seed,
        agente_idx=args.jugador,
        verbose=not args.quiet,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"ESTADÍSTICAS ({args.manos} manos en {elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  Puntuación media : {stats.puntuacion_media:.2f} ± {stats.puntuacion_std:.2f}")
    print(f"  Manos con 0 pts  : {stats.tasa_cero*100:.1f}% ({int(stats.tasa_cero*args.manos)})")
    print(f"  Tasa de pozo     : {stats.tasa_pozo*100:.1f}%")
    print(f"  Tasa captura Q♠  : {stats.tasa_q_capturada*100:.1f}%")
    print(f"\n  Distribución de puntuaciones:")
    for pts in sorted(stats.distribucion):
        freq = stats.distribucion[pts]
        bar = "█" * freq
        print(f"    {pts:3d}: {freq:3d} {bar}")


def cmd_comparar(args):
    """Comparación A/B de dos políticas."""
    pol_a, nombre_a = _cargar_politica(args.politica_a)
    pol_b, nombre_b = _cargar_politica(args.politica_b)

    print(f"Comparando: {nombre_a} vs {nombre_b}")
    print(f"{args.manos} manos | Rollout: {args.rollout} | Seed: {args.seed}")

    t0 = time.time()
    comp = comparar_politicas(
        politica_a=pol_a,
        politica_b=pol_b,
        nombre_a=nombre_a,
        nombre_b=nombre_b,
        num_manos=args.manos,
        seed_inicial=args.seed,
        verbose=not args.quiet,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"COMPARACIÓN ({args.manos} manos en {elapsed:.1f}s)")
    print(f"{'='*55}")
    print(f"  {nombre_a}: {comp.media_a:.2f} pts/mano")
    print(f"  {nombre_b}: {comp.media_b:.2f} pts/mano")
    print(f"  Diferencia : {comp.media_a - comp.media_b:+.2f} pts/mano")
    print(f"  A favor de {nombre_a}: {comp.manos_a_favor_de_a}")
    print(f"  A favor de {nombre_b}: {comp.manos_a_favor_de_b}")
    print(f"  Empates              : {comp.empates}")

    import numpy as np
    from scipy import stats as sp_stats
    if len(comp.puntuacion_a) > 1:
        # Paired t-test
        t, p = sp_stats.ttest_rel(comp.puntuacion_a, comp.puntuacion_b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  t-test pareado: t={t:.2f}, p={p:.4f} ({sig})")


def main():
    parser = argparse.ArgumentParser(
        description="Análisis profundo de manos con PIMC exacto"
    )
    sub = parser.add_subparsers(dest="cmd")

    # ── perfil ──
    p_perfil = sub.add_parser("perfil", help="Análisis ultra-detallado de UNA mano")
    p_perfil.add_argument("--seed", type=int, default=42)
    p_perfil.add_argument("--jugador", type=int, default=0)
    p_perfil.add_argument("--politica", default="bot_experto")
    p_perfil.add_argument("--rollout", default="experto",
                          choices=["evasivo", "experto", "mixto"])
    p_perfil.add_argument("--quiet", action="store_true")

    # ── evaluar ──
    p_eval = sub.add_parser("evaluar", help="Evalúa N manos")
    p_eval.add_argument("--manos", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--jugador", type=int, default=0)
    p_eval.add_argument("--politica", default="bot_experto")
    p_eval.add_argument("--rollout", default="experto",
                        choices=["evasivo", "experto", "mixto"])
    p_eval.add_argument("--quiet", action="store_true")

    # ── comparar ──
    p_comp = sub.add_parser("comparar", help="Compara dos políticas A/B")
    p_comp.add_argument("--politica-a", required=True)
    p_comp.add_argument("--politica-b", required=True)
    p_comp.add_argument("--manos", type=int, default=50)
    p_comp.add_argument("--seed", type=int, default=42)
    p_comp.add_argument("--rollout", default="experto",
                        choices=["evasivo", "experto", "mixto"])
    p_comp.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.cmd == "perfil":
        cmd_perfil(args)
    elif args.cmd == "evaluar":
        cmd_evaluar(args)
    elif args.cmd == "comparar":
        cmd_comparar(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
