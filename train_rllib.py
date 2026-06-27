"""
Pipeline de entrenamiento de Hearts con Ray RLlib.

Uso:
    python train_rllib.py --total-steps 20000000 --output-dir models/v8
    python train_rllib.py --total-steps 20000000 --output-dir models/v8 --workers 4 --use-lstm

Curriculum de 5 fases (automático según progreso):
    0–5%    → Fase 0: 3 bots simples (bootstrap)
    5–15%   → Fase 1: 2 bots simples + 1 BotExperto
    15–40%  → Fase 2: 1 BotExperto + 2 snapshots
    40–70%  → Fase 3: 3 snapshots del pool completo (self-play puro)
    70–100% → Fase 4: 3 snapshots recientes (presión máxima)

Principios de diseño (v10 — PARTIDA COMPLETA):
  - Episodio = una partida completa a 100 puntos (el marcador persiste entre manos).
  - Recompensa = R_terminal por puesto final (1º=+1, 2º=+0.3, 3º=−0.3, 4º=−1)
    + shaping PBRS sobre el marcador (garantizado no-farmeable).
  - gamma alto (0.999) para propagar el puesto final; debe coincidir env↔PPO.
  - Rotación multi-posición: el agente entrena desde las 4 posiciones aleatoriamente.
  - La factory se refresca cada FACTORY_REFRESH_STEPS para incorporar snapshots nuevos.

Ver: docs/Rediseño_v10_partida_completa.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Forzar UTF-8 en stdout/stderr: rich (panel Live + mensajes) usa caracteres
# unicode que rompen en consolas Windows cp1252 cuando la salida se redirige
# (p.ej. ejecución en background). Sin esto, un print con unicode mata el run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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
from src.rllib.eval_bots import evaluar_vs_bots
from src.rllib.opponent_pool import OpponentPool
from src.rllib.utils import guardar_snapshot, podar_snapshots, preservar_elite

console = Console()

# Refrescar la factory cada N pasos para incorporar nuevos snapshots al pool
FACTORY_REFRESH_STEPS = 500_000

# Evaluar contra bots fijos cada N snapshots (métrica absoluta independiente del pool)
EVAL_BOT_INTERVAL = 5   # cada 5 snapshots ≈ cada 500K pasos

_FASES = {
    0: ("Bootstrap (3 bots)",                "cyan"),
    1: ("Experto (2 bots + BotExperto)",     "yellow"),
    2: ("Mix (BotExperto + 2 snaps)",        "green"),
    3: ("Self-play puro (3 snaps)",          "blue"),
    4: ("Self-play duro (3 snaps recientes)", "magenta"),
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
    p.add_argument("--lr", type=float, default=3e-4,
                   help="LR inicial (default 3e-4)")
    p.add_argument("--lr-end", type=float, default=1e-4,
                   help="LR mínimo al final del entrenamiento (default 1e-4, nunca decae a cero)")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--no-random-position", action="store_true",
                   help="Desactivar rotación multi-posición (agente siempre en idx=0)")
    p.add_argument("--gamma", type=float, default=0.999,
                   help="Factor de descuento. Alto porque el episodio es una partida "
                        "completa (~100-170 steps). Debe coincidir env↔PPO (PBRS).")
    p.add_argument("--phi-lambda", type=float, default=0.5,
                   help="Peso del potencial Φ del shaping PBRS (default 0.5)")
    p.add_argument("--limite-partida", type=int, default=100,
                   help="Puntos para terminar la partida (default 100)")
    p.add_argument("--baza-reward-weight", type=float, default=0.15,
                   help="OBSOLETO (v9): ignorado en v10. Se mantiene por compatibilidad.")
    p.add_argument("--use-lstm", action="store_true",
                   help="Usar HeartsLSTMModel (LSTM 256) en lugar del MLP estándar")
    p.add_argument("--lstm-hidden", type=int, default=256,
                   help="Tamaño del estado oculto LSTM (default 256, solo con --use-lstm)")
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
    # Reward de partida ≈ [-1.5, +1.5] (R_terminal ±1/±0.3 + shaping PBRS).
    # >0 indica que el agente tiende a quedar en la mitad alta de la tabla.
    reward_color = "green" if (not math.isnan(reward_medio) and reward_medio > 0) else "red"
    grid.add_row(
        f"[bold]Reward medio:[/bold] [{reward_color}]{reward_medio:.3f}[/{reward_color}]",
        f"[bold]Reward máx:[/bold]  {reward_max:.2f}",
    )
    grid.add_row(
        f"[bold]Ep len (steps/partida):[/bold] {ep_len:.0f}",
        f"[bold]Snapshots:[/bold] {num_snapshots}",
    )
    grid.add_row(
        f"[bold]Fase:[/bold] [{fase_color}]{fase_actual} — {fase_label}[/{fase_color}]",
        f"[bold]Log:[/bold] {log_path}",
    )

    return Panel(grid, title="[bold white]Hearts RLlib — Entrenamiento[/bold white]", border_style="blue")


def main() -> None:
    args = parse_args()
    random_position = not args.no_random_position

    snapshot_dir = os.path.join(args.output_dir, "snapshots")
    elite_dir = os.path.join(args.output_dir, "elite")
    log_dir = os.path.join(args.output_dir, "logs")
    log_path = os.path.join(log_dir, "eval_log.jsonl")
    bot_eval_log_path = os.path.join(log_dir, "bot_eval_log.jsonl")
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            **vars(args),
            "random_position": random_position,
            "lr_schedule": f"{args.lr} -> {args.lr_end}",
            "curriculum": "5-fases",
        }, f, indent=2)

    HeartsCallbacks._log_dir = log_dir

    ray.init(ignore_reinit_error=True)
    console.print(f"[green]Ray {ray.__version__} inicializado[/green]")
    console.print(
        f"Output: [bold]{args.output_dir}[/bold]  |  Workers: {args.workers}  |  "
        f"Steps: {args.total_steps:,}  |  Posición aleatoria: {random_position}"
    )

    pool = OpponentPool(
        snapshot_dir=snapshot_dir,
        max_snapshots=args.max_snapshots,
        obs_dim=args.obs_dim,
    )

    from src.entorno.recompensas_partida import RewardConfigPartida
    reward_config = RewardConfigPartida(
        PHI_LAMBDA=args.phi_lambda,
        LIMITE_PARTIDA=args.limite_partida,
    )

    config = build_ppo_config(
        opponent_factory=None,
        obs_dim=args.obs_dim,
        random_position=random_position,
        reward_config=reward_config,
        gamma=args.gamma,
        lr=args.lr,
        lr_end=args.lr_end,
        total_steps=args.total_steps,
        train_batch_size=args.batch_size,
        num_rollout_workers=args.workers,
        num_gpus=args.gpus,
        use_lstm=args.use_lstm,
        lstm_hidden_size=args.lstm_hidden,
    )
    config = config.callbacks(HeartsCallbacks)

    algo = config.build_algo()

    # Inyectar factory inicial en todos los envs
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
    ultimo_factory_refresh = 0
    fase_actual = 0
    iter_count = 0
    reward_medio = float("nan")
    reward_max = float("nan")
    ep_len = float("nan")
    num_snapshots = 0
    snapshots_desde_ultima_eval = 0

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while pasos_totales < args.total_steps:
                result = algo.train()
                iter_count += 1

                pasos_totales = result.get("timesteps_total", pasos_totales)
                progress = pasos_totales / args.total_steps

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

                    # Evaluación periódica contra bots fijos (métrica absoluta)
                    snapshots_desde_ultima_eval += 1
                    if snapshots_desde_ultima_eval >= EVAL_BOT_INTERVAL:
                        snapshots_desde_ultima_eval = 0
                        try:
                            policy = algo.get_policy()
                            metricas_bot = evaluar_vs_bots(
                                policy, obs_dim=args.obs_dim, n_partidas=50
                            )
                            metricas_bot["tipo"] = "bot_eval"
                            metricas_bot["paso"] = pasos_totales
                            metricas_bot["fase"] = fase_actual
                            with open(bot_eval_log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(metricas_bot) + "\n")

                            # Preservar el mejor modelo (elite) — protege contra
                            # el pruning y contra la regresión del self-play.
                            # Score compuesto centrado en el desafío real (experto).
                            score = (
                                0.5 * metricas_bot.get("top2_rate_vs_experto", 0.0)
                                + 0.3 * metricas_bot.get("win_rate_vs_experto", 0.0)
                                + 0.2 * metricas_bot.get("top2_rate", 0.0)
                            )
                            if preservar_elite(ruta, elite_dir, score, pasos_totales):
                                console.print(
                                    f"[green][elite] preservado[/green] "
                                    f"(paso {pasos_totales:,}, score {score:.3f})"
                                )
                        except Exception as exc:
                            console.print(f"[yellow]bot_eval error: {exc}[/yellow]")

                # Refrescar factory cuando cambia de fase O cada FACTORY_REFRESH_STEPS.
                # Esto garantiza que los snapshots nuevos se incorporen al pool de rivales.
                nueva_fase = _fase(progress)
                fase_cambio = nueva_fase != fase_actual
                factory_stale = (pasos_totales - ultimo_factory_refresh) >= FACTORY_REFRESH_STEPS

                if fase_cambio or factory_stale:
                    fase_actual = nueva_fase
                    ultimo_factory_refresh = pasos_totales
                    new_factory = pool.make_factory(progress=progress)
                    algo.env_runner_group.foreach_env(
                        lambda env: setattr(env, "_opponent_factory", new_factory)
                    )

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
