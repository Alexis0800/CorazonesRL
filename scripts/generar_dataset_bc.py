"""
Generador de dataset BC (Behavioral Cloning) usando PIMC como oracle.

Genera pares (observación, acción_óptima) consultando PIMC en cada
turno del agente. Usa multiprocessing para acelerar la generación.

Uso:
    # Rápido (30 mundos, rollout mixto, 4 workers)
    python scripts/generar_dataset_bc.py --manos 100 --output datasets/bc_v1

    # Alta calidad (100 mundos, BotExperto, 8 workers)
    python scripts/generar_dataset_bc.py --manos 500 --mundos 100 --rollout experto --workers 8 --output datasets/bc_v1_hq

    # Con semilla fija para reproducibilidad
    python scripts/generar_dataset_bc.py --manos 200 --seed 42 --output datasets/bc_v1
"""

from __future__ import annotations
from src.mcts.dataset import generar_dataset, guardar_dataset

import argparse
import sys
import os

# Asegurar que src/ está en el path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generar dataset BC con PIMC + multiprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/generar_dataset_bc.py --manos 100 --output datasets/bc_v1
  python scripts/generar_dataset_bc.py --manos 500 --mundos 100 --rollout experto --workers 8
        """,
    )
    parser.add_argument(
        "--manos", type=int, default=100,
        help="Número de manos a generar (default: 100)")
    parser.add_argument(
        "--mundos", type=int, default=30,
        help="Mundos PIMC por decisión (default: 30)")
    parser.add_argument(
        "--rollout", type=str, default="mixto",
        choices=["evasivo", "experto", "mixto"],
        help="Política de PIMC: evasivo (rápido), experto (preciso), mixto (balance)")
    parser.add_argument(
        "--oponentes", type=str, default="experto",
        choices=["heuristicos", "experto", "mixto"],
        help="Oponentes reales: heuristicos, experto, mixto (default: experto)")
    parser.add_argument(
        "--mcts", action="store_true", default=False,
        help="Usar MCTS multi-step en vez de PIMC one-step")
    parser.add_argument(
        "--mcts-sims", type=int, default=100,
        help="Simulaciones MCTS por decisión (default: 100)")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Número de procesos paralelos (default: 4)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla base para reproducibilidad (default: 42)")
    parser.add_argument(
        "--output", type=str, default="datasets/bc_dataset",
        help="Prefijo de salida (sin extensión). Genera .npz + .json")

    args = parser.parse_args()

    print("=" * 60)
    print("  GENERADOR DE DATASET BC (PIMC + multiprocessing)")
    print("=" * 60)
    print(f"  Manos:        {args.manos}")
    print(f"  Mundos PIMC:  {args.mundos}")
    print(f"  Rollout PIMC: {args.rollout}")
    print(f"  Oponentes:    {args.oponentes}")
    print(f"  Oracle:       {'MCTS' if args.mcts else 'PIMC'} "
          f"({'{} sims'.format(args.mcts_sims) if args.mcts else '{} mundos'.format(args.mundos)})")
    print(f"  Workers:      {args.workers}")
    print(f"  Seed base:    {args.seed}")
    print(f"  Output:       {args.output}.npz + .json")
    print("-" * 60)

    obs, actions, meta = generar_dataset(
        num_manos=args.manos,
        num_mundos=args.mundos,
        rollout_tipo=args.rollout,
        tipo_oponentes=args.oponentes,
        num_workers=args.workers,
        seed=args.seed,
        use_mcts=args.mcts,
        mcts_simulaciones=args.mcts_sims,
    )

    guardar_dataset(args.output, obs, actions, meta)

    print(f"\n  ✅ Dataset generado: {len(obs)} pares (obs, action)")
    print(f"  📁 {args.output}.npz  ({obs.nbytes / 1024:.0f} KB)")
    print(f"  📁 {args.output}.json")
    print(f"  ⏱️  Tiempo total: {meta['tiempo_total_s']:.0f}s "
          f"({meta['tiempo_por_mano_s']:.1f}s/mano)")


if __name__ == "__main__":
    main()
