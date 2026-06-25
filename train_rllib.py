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
import math
import os

import ray
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from src.entorno.dimensiones import DIM_ENTORNO
from src.rllib.callbacks import HeartsCallbacks
from src.rllib.config import build_ppo_config
from src.rllib.opponent_pool import OpponentPool
from src.rllib.utils import guardar_snapshot, podar_snapshots

console = Console()

_FASES = {
    0: ("Bots simples",         "cyan"),
    1: ("BotExperto",           "blue"),
    2: ("Experto + snapshots",  "yellow"),
    3: ("Snapshots mayoritario","orange1"),
    4: ("Snapshots puros",      "green"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenar agente Hearts con RLlib PPO")
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--output-dir", type=str, default="models/v_rllib")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--snapshot-interval", type=int, default=100_000,
                   help="Pasos entre cada snapshot guardado (default 100k)")
    p.add_argument("--max-snapshots", type=int, default=50)
    p.add_argument("--obs-dim", type=int, default=DIM_ENTORNO)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def _fase(progress: float) -> int:
    if progress < 0.05:
        return 0
    elif progress < 0.15:
        return 1
    elif progress < 0.40:
        return 2
    elif progress < 0.70:
        return 3
    return 4


def _build_panel(
    iter_count: int,
    pasos_totales: int,
    total_steps: int,
    reward_medio: float,
    reward_max: float,
    ep_len: float,
    fase_actual: int,
    num_snapshots: int,
    log_path: str,
) -> Panel:
    progress_pct = 100.0 * pasos_totales / total_steps
    fase_label, fase_color = _FASES.get(fase_actual, ("?", "white"))

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    grid.add_row(
        f"[bold]Iteración:[/bold] {iter_count}",
        f"[bold]Steps:[/bold] {pasos_totales:,} / {total_steps:,}  ({progress_pct:.1f}%)",
    )
    reward_color = "green" if (not math.isnan(reward_medio) and reward_medio > 0) else "red"
    grid.add_row(
        f"[bold]Reward medio:[/bold] [{reward_color}]{reward_medio:.3f}[/{reward_color}]",
        f"[bold]Reward máx:[/bold]  {reward_max:.2f}",
    )
    grid.add_row(
        f"[bold]Ep len:[/bold] {ep_len:.1f}",
        f"[bold]Snapshots:[/bold] {num_snapshots}",
    )
    grid.add_row(
        f"[bold]Fase:[/bold] [{fase_color}]{fase_actual} — {fase_label}[/{fase_color}]",
        f"[bold]Log:[/bold] {log_path}",
    )

    return Panel(grid, title="[bold white]Hearts RLlib — Entrenamiento[/bold white]", border_style="blue")


def main() -> None:
    args = parse_args()

    snapshot_dir = os.path.join(args.output_dir, "snapshots")
    log_dir = os.path.join(args.output_dir, "logs")
    log_path = os.path.join(log_dir, "eval_log.jsonl")
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Pasar log_dir a HeartsCallbacks ANTES de construir el algo
    HeartsCallbacks._log_dir = log_dir

    ray.init(ignore_reinit_error=True)
    console.print(f"[green]Ray {ray.__version__} inicializado[/green]")
    console.print(f"Output: [bold]{args.output_dir}[/bold]  |  Workers: {args.workers}  |  Steps: {args.total_steps:,}")

    pool = OpponentPool(
        agente_idx=0,
        snapshot_dir=snapshot_dir,
        max_snapshots=args.max_snapshots,
        obs_dim=args.obs_dim,
    )

    # opponent_factory=None en el config para que sea serializable en checkpoints.
    # La factory real se inyecta en los envs vía foreach_env justo después de build.
    config = build_ppo_config(
        opponent_factory=None,
        obs_dim=args.obs_dim,
        lr=args.lr,
        train_batch_size=args.batch_size,
        num_rollout_workers=args.workers,
        num_gpus=args.gpus,
    )
    config = config.callbacks(HeartsCallbacks)

    algo = config.build_algo()

    # Inyectar fase inicial en todos los envs ahora que ya están creados
    factory_inicial = pool.make_factory(progress=0.0)
    algo.env_runner_group.foreach_env(
        lambda env: setattr(env, "_opponent_factory", factory_inicial)
    )

    console.print("[bold green]Algoritmo construido. Iniciando entrenamiento...[/bold green]\n")

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress_bar.add_task("Steps", total=args.total_steps)

    pasos_totales = 0
    ultimo_snapshot = 0
    proxima_fase = -1
    iter_count = 0
    reward_medio = float("nan")
    reward_max = float("nan")
    ep_len = float("nan")
    fase_actual = 0
    num_snapshots = 0

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while pasos_totales < args.total_steps:
                result = algo.train()
                iter_count += 1

                pasos_totales = result.get("timesteps_total", pasos_totales)
                progress = pasos_totales / args.total_steps

                # Métricas en Ray 2.55.1 están bajo "env_runners"
                env_r = result.get("env_runners", {})
                reward_medio = env_r.get("episode_reward_mean", float("nan"))
                reward_max   = env_r.get("episode_reward_max",  float("nan"))
                ep_len       = env_r.get("episode_len_mean",    float("nan"))

                # Guardar snapshot periódico
                if pasos_totales - ultimo_snapshot >= args.snapshot_interval:
                    ruta = guardar_snapshot(algo, pasos_totales, snapshot_dir)
                    ultimo_snapshot = pasos_totales
                    num_snapshots += 1

                    try:
                        policy = algo.get_policy()
                        pool.add_snapshot(policy)
                    except Exception:
                        pass

                    podar_snapshots(snapshot_dir, mantener=args.max_snapshots)

                    # Escribir métricas al log
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "tipo": "snapshot",
                            "paso": pasos_totales,
                            "reward_medio": reward_medio,
                            "reward_max": reward_max,
                            "ep_len": ep_len,
                            "fase": fase_actual,
                            "snapshot": os.path.basename(ruta),
                        }) + "\n")

                # Actualizar fase de oponentes
                nueva_fase = _fase(progress)
                if nueva_fase != proxima_fase:
                    proxima_fase = nueva_fase
                    fase_actual = nueva_fase
                    new_factory = pool.make_factory(progress=progress)
                    algo.env_runner_group.foreach_env(
                        lambda env: setattr(env, "_opponent_factory", new_factory)
                    )

                # Actualizar panel + barra de progreso
                progress_bar.update(task, completed=min(pasos_totales, args.total_steps))
                live.update(Group(
                    _build_panel(
                        iter_count, pasos_totales, args.total_steps,
                        reward_medio, reward_max, ep_len,
                        fase_actual, num_snapshots, log_path,
                    ),
                    progress_bar,
                ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrumpido. Guardando checkpoint final...[/yellow]")

    finally:
        ruta_final = guardar_snapshot(algo, pasos_totales, snapshot_dir)
        console.print(f"[green]Checkpoint final:[/green] {ruta_final}")
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
