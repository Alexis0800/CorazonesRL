"""
Pipeline de entrenamiento de Hearts con Ray RLlib.

Uso:
    python train_rllib.py --total-steps 20000000 --output-dir models/v_rllib
    python train_rllib.py --total-steps 20000000 --output-dir models/v_rllib --workers 4 --gpus 1

Fases de self-play (automáticas según progreso de entrenamiento):
    0–5%   → 3 bots heurísticos simples
    5–15%  → 2 bots + BotExperto
    15–40% → BotExperto + snapshots históricos
    40–70% → mayoría snapshots + BotExperto
    70–100% → snapshots puros
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import ray

from src.rllib.config import build_ppo_config
from src.rllib.callbacks import HeartsCallbacks
from src.rllib.opponent_pool import OpponentPool
from src.rllib.utils import guardar_snapshot, listar_snapshots, podar_snapshots
from src.entorno.dimensiones import DIM_ENTORNO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar agente Hearts con RLlib PPO")
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--output-dir", type=str, default="models/v_rllib")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--snapshot-interval", type=int, default=200_000,
                   help="Pasos entre cada snapshot guardado")
    p.add_argument("--max-snapshots", type=int, default=50)
    p.add_argument("--obs-dim", type=int, default=DIM_ENTORNO)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    snapshot_dir = os.path.join(args.output_dir, "snapshots")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Guardar configuración
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    ray.init(ignore_reinit_error=True)
    print(f"[train_rllib] Ray {ray.__version__} inicializado")
    print(f"[train_rllib] Output: {args.output_dir}")
    print(f"[train_rllib] Total steps: {args.total_steps:,}")

    pool = OpponentPool(
        agente_idx=0,
        snapshot_dir=snapshot_dir,
        max_snapshots=args.max_snapshots,
        obs_dim=args.obs_dim,
    )

    # Configuración inicial (Fase 0: solo bots)
    config = build_ppo_config(
        opponent_factory=pool.make_factory(progress=0.0),
        obs_dim=args.obs_dim,
        lr=args.lr,
        train_batch_size=args.batch_size,
        num_rollout_workers=args.workers,
        num_gpus=args.gpus,
    )
    config = config.callbacks(HeartsCallbacks)

    algo = config.build()
    print("[train_rllib] Algoritmo construido. Iniciando entrenamiento...")

    pasos_totales = 0
    ultimo_snapshot = 0
    proxima_fase = -1
    iter_count = 0

    try:
        while pasos_totales < args.total_steps:
            result = algo.train()
            iter_count += 1
            pasos_totales = result.get("timesteps_total", pasos_totales)
            progress = pasos_totales / args.total_steps

            reward_medio = result.get("episode_reward_mean", float("nan"))
            ep_len_medio = result.get("episode_len_mean", float("nan"))

            if iter_count % 10 == 0:
                print(
                    f"[iter {iter_count}] "
                    f"steps={pasos_totales:,} ({100*progress:.1f}%) | "
                    f"reward={reward_medio:.2f} | "
                    f"ep_len={ep_len_medio:.1f}"
                )

            # Guardar snapshot periódico
            if pasos_totales - ultimo_snapshot >= args.snapshot_interval:
                ruta = guardar_snapshot(algo, pasos_totales, snapshot_dir)
                print(f"[train_rllib] Snapshot guardado: {ruta}")
                ultimo_snapshot = pasos_totales

                # Cargar snapshot en el pool para self-play
                try:
                    policy = algo.get_policy()
                    pool.add_snapshot(policy)
                except Exception as e:
                    print(f"[train_rllib] Warning: no se pudo añadir snapshot al pool: {e}")

                # Podar snapshots excedentes
                podar_snapshots(snapshot_dir, mantener=args.max_snapshots)

            # Actualizar oponentes según la fase
            fase_actual = _fase(progress)
            if fase_actual != proxima_fase:
                proxima_fase = fase_actual
                new_factory = pool.make_factory(progress=progress)
                algo.workers.foreach_worker(
                    lambda w: w.foreach_env(
                        lambda env: setattr(env, "_opponents",
                                            new_factory() if new_factory else {})
                    )
                )
                print(f"[train_rllib] Fase {fase_actual} activada (progress={100*progress:.1f}%)")

    except KeyboardInterrupt:
        print("\n[train_rllib] Interrumpido por el usuario. Guardando checkpoint final...")

    finally:
        ruta_final = guardar_snapshot(algo, pasos_totales, snapshot_dir)
        print(f"[train_rllib] Checkpoint final: {ruta_final}")
        algo.stop()
        ray.shutdown()


def _fase(progress: float) -> int:
    if progress < 0.05:
        return 0
    elif progress < 0.15:
        return 1
    elif progress < 0.40:
        return 2
    elif progress < 0.70:
        return 3
    else:
        return 4


if __name__ == "__main__":
    main()
