"""
Análisis de errores de un snapshot usando PIMC como oráculo.

Uso:
    # Perfil detallado de 3 manos (decisión a decisión)
    python analizar_errores.py --modo perfil --snapshot models/v8/snapshots/snapshot_000019914752

    # Estadísticas sobre 200 manos
    python analizar_errores.py --modo stats --snapshot models/v8/snapshots/snapshot_000019914752

    # Comparar v8 vs BotExperto en las mismas manos
    python analizar_errores.py --modo comparar --snapshot models/v8/snapshots/snapshot_000019914752
"""
from __future__ import annotations

import argparse

from src.mcts.analisis import (
    comparar_politicas,
    estadisticas_manos,
    perfil_mano,
)
from src.rllib.utils import cargar_policy_desde_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True,
                   help="Ruta al directorio del snapshot RLlib")
    p.add_argument("--modo", choices=["perfil", "stats", "comparar"],
                   default="perfil")
    p.add_argument("--manos", type=int, default=3,
                   help="Manos para --modo perfil (default 3)")
    p.add_argument("--stats-manos", type=int, default=200,
                   help="Manos para --modo stats (default 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollout", choices=["evasivo", "experto", "mixto"],
                   default="experto",
                   help="Política de rollout PIMC (default experto)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\nCargando snapshot: {args.snapshot}")
    politica = cargar_policy_desde_checkpoint(args.snapshot)
    print("Snapshot cargado correctamente.\n")

    if args.modo == "perfil":
        for i in range(args.manos):
            seed = args.seed + i
            perfil_mano(
                seed=seed,
                agente_idx=0,
                politica_evaluada=politica,
                rollout_tipo=args.rollout,
                verbose=True,
            )

    elif args.modo == "stats":
        print(f"Evaluando {args.stats_manos} manos...")
        stats = estadisticas_manos(
            politica=politica,
            num_manos=args.stats_manos,
            seed_inicial=args.seed,
            verbose=True,
        )
        print(f"\n{'='*50}")
        print(f"  ESTADÍSTICAS ({args.stats_manos} manos)")
        print(f"{'='*50}")
        print(f"  Puntuación media : {stats.puntuacion_media:.2f} pts")
        print(f"  Desviación std   : {stats.puntuacion_std:.2f} pts")
        print(f"  Tasa 0 pts       : {stats.tasa_cero*100:.1f}%")
        print(f"  Tasa pozo (luna) : {stats.tasa_pozo*100:.1f}%")
        print(f"  Tasa Q♠ capturada: {stats.tasa_q_capturada*100:.1f}%")
        print(f"\n  Distribución de puntuaciones:")
        for pts in sorted(stats.distribucion):
            freq = stats.distribucion[pts]
            barra = "█" * (freq * 40 // args.stats_manos)
            print(f"    {pts:2d} pts: {barra} {freq}")

    elif args.modo == "comparar":
        from src.agentes.bot_experto import BotExperto
        bot = BotExperto()

        print(f"Comparando v8 vs BotExperto en {args.stats_manos} manos...")
        resultado = comparar_politicas(
            politica_a=politica,
            politica_b=bot,
            nombre_a="v8",
            nombre_b="BotExperto",
            num_manos=args.stats_manos,
            seed_inicial=args.seed,
            verbose=True,
        )
        print(f"\n{'='*50}")
        print(f"  RESULTADO FINAL")
        print(f"{'='*50}")
        print(f"  v8       : {resultado.media_a:.2f} pts/mano")
        print(f"  BotExperto: {resultado.media_b:.2f} pts/mano")
        print(f"  Manos ganadas v8: {resultado.manos_a_favor_de_a}")
        print(f"  Manos ganadas BE: {resultado.manos_a_favor_de_b}")
        print(f"  Empates         : {resultado.empates}")
        ganador = "v8" if resultado.media_a < resultado.media_b else "BotExperto"
        print(f"  → Gana: {ganador} "
              f"(Δ {abs(resultado.media_a - resultado.media_b):.2f} pts)")


if __name__ == "__main__":
    main()
