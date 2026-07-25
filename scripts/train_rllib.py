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

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

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
from src.entorno.dimensiones import DIM_ENTORNO, DIM_V12
from src.entorno.moon_model import RUTA_MOON
from src.rllib.callbacks import HeartsCallbacks
from src.rllib.config import build_ppo_config
from src.rllib.eval_bots import evaluar_vs_bots
from src.rllib.opponent_pool import OpponentPool
from src.rllib.utils import (
    guardar_snapshot, podar_snapshots, preservar_elite, listar_snapshots,
)

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
    p.add_argument("--resume", default=None,
                   help="Reanuda desde un checkpoint. 'auto' = último snapshot de "
                        "output-dir/snapshots; o pasa la ruta a un snapshot concreto.")
    p.add_argument("--obs-dim", type=int, default=DIM_ENTORNO)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="LR inicial (default 3e-4)")
    p.add_argument("--lr-end", type=float, default=1e-4,
                   help="LR mínimo al final del entrenamiento (default 1e-4, nunca decae a cero)")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--no-random-position", action="store_true",
                   help="Desactivar rotación multi-posición (agente siempre en idx=0)")
    p.add_argument("--entropy-coeff", type=float, default=0.05,
                   help="Coef. de entropía. Bajar (~0.01) para fine-tune desde BC "
                        "(evita que la exploración erosione la política pre-entrenada).")
    p.add_argument("--gamma", type=float, default=0.999,
                   help="Factor de descuento. Alto porque el episodio es una partida "
                        "completa (~100-170 steps). Debe coincidir env↔PPO (PBRS).")
    p.add_argument("--phi-rank", action="store_true",
                   help="PBRS con potencial por PUESTO continuo (Φ_rank) en vez "
                        "del de medias — corrige el 9.8%% de estados donde Φ "
                        "contradice a R_terminal en la frontera 2º/3º. A/B del "
                        "Run A (docs/plan_mejora_vs_humanos_2026-07-25.md).")
    p.add_argument("--phi-lambda", type=float, default=0.5,
                   help="Peso del potencial Φ del shaping PBRS (default 0.5)")
    p.add_argument("--limite-partida", type=int, default=100,
                   help="Puntos para terminar la partida (default 100)")
    p.add_argument("--ancla-experto", action="store_true",
                   help="Mantener 1 BotExperto como ancla en fases 3-4 (evita "
                        "regresión del self-play puro). Default off = como v10.")
    p.add_argument("--bc-weights", type=str, default=None,
                   help="Pickle de pesos BC (entrenar_bc.py) para inicializar la "
                        "política PPO por imitación de PIMC. Usar con LR bajo.")
    p.add_argument("--humano-bc", type=str, default=None,
                   help="Ruta a pesos.npz del bot de imitación humana "
                        "(scripts/entrenar_bc_humano.py). Lo mete al pool como "
                        "oponente 'duro' para romper la burbuja de self-play "
                        "(79%% vs bots, 24%% vs humanos). Ver docs/auditoria_moon_2026-07-20.md.")
    p.add_argument("--prob-humano", type=float, default=0.5,
                   help="Fracción de slots de oponente duro cubiertos por el bot "
                        "humano (si --humano-bc). Default 0.5.")
    p.add_argument("--mesa-humana", type=float, default=0.0,
                   help="Fracción de MESAS completas con 2 clones humanos + 1 ancla "
                        "(si --humano-bc). Exposición mayoritaria al meta humano; "
                        "la vía por-slot (prob-humano) se diluye a ~1/6.")
    p.add_argument("--temp-humano", type=float, default=1.0,
                   help="Temperatura de muestreo del clon humano (softmax/T). "
                        "1.0 = estocástico como un humano; 0 o negativo = argmax.")
    p.add_argument("--progress-fino", action="store_true",
                   help="Calcula el progress del curriculum RELATIVO al paso "
                        "reanudado: (pasos-reanudado)/(total-reanudado). Sin esto, "
                        "un fine-tune con --resume arranca con progress≈1 y queda "
                        "atrapado en Fase 4 self-play (bug del run v10d).")
    p.add_argument("--moon-dir", type=str, default=None,
                   help="Directorio de pesos moon (propio.pt/rival.pt) para las "
                        "features [187:189]. Default None = RUTA_MOON "
                        "(models/moon_realfull, rival AUC 0.707).")
    p.add_argument("--lunero-garantizado", action="store_true",
                   help="Fases 2-4: un slot de oponente es SIEMPRE OponenteLunar "
                        "(ModoLunar + BotExperto) — presión de luna al estilo "
                        "humano oportunista (~2.5%%/mano, compromiso mid-mano).")
    p.add_argument("--pool-diverso", action="store_true",
                   help="El oponente duro de cada fase es un arquetipo humano al "
                        "azar (experto/castigador/lunatico/atacante), no solo "
                        "BotExperto. Mejora la generalización a juego variado.")
    p.add_argument("--con-pase", action="store_true",
                   help="Habilita la fase de PASE. Fuerza obs_dim>=DIM_V12. "
                        "Con --obs-dim 332 (DIM_V13) añade memoria del pase: "
                        "cartas dadas al receptor + recibidas del dador.")
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

    # v10b: el pase requiere las 4 features de pase en la obs (DIM_V12).
    if args.con_pase and args.obs_dim < DIM_V12:
        args.obs_dim = DIM_V12

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
        anclar_experto=args.ancla_experto,
        pool_diverso=args.pool_diverso,
        humano_bc_path=args.humano_bc,
        prob_humano=args.prob_humano,
        mesa_humana=args.mesa_humana,
        temp_humano=args.temp_humano if args.temp_humano > 0 else None,
        lunero_garantizado=args.lunero_garantizado,
    )

    from src.entorno.recompensas_partida import RewardConfigPartida
    reward_config = RewardConfigPartida(
        PHI_LAMBDA=args.phi_lambda,
        LIMITE_PARTIDA=args.limite_partida,
        PHI_RANK=args.phi_rank,
    )

    config = build_ppo_config(
        opponent_factory=None,
        obs_dim=args.obs_dim,
        random_position=random_position,
        reward_config=reward_config,
        con_pase=args.con_pase,
        moon_dir=args.moon_dir,
        gamma=args.gamma,
        entropy_coeff=args.entropy_coeff,
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
    # Tolerancia a fallos: en Windows un worker puede morir (RAM/Ray). Que Ray los
    # reinicie en vez de tumbar el entrenamiento con un access violation.
    config = config.fault_tolerance(
        restart_failed_env_runners=True,
        ignore_env_runner_failures=True,
        restart_failed_sub_environments=True,
    )

    algo = config.build_algo()

    # --- Reanudar desde checkpoint (--resume) O inicializar por BC ---
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            _snaps = listar_snapshots(snapshot_dir)
            resume_path = _snaps[-1][1] if _snaps else None
        elif os.path.isdir(args.resume):
            resume_path = args.resume
        if not resume_path:
            console.print(
                f"[yellow]--resume='{args.resume}' sin checkpoint válido; "
                f"empezando de cero.[/yellow]"
            )

    paso_reanudado = 0
    if resume_path:
        # pyarrow (que usa RLlib para leer el checkpoint) exige ruta ABSOLUTA en
        # Windows; una relativa da "URI has empty scheme".
        algo.restore(os.path.abspath(resume_path))
        try:
            paso_reanudado = int(os.path.basename(resume_path).split("_")[1])
        except (IndexError, ValueError):
            paso_reanudado = 0
        console.print(
            f"[green]Reanudado desde:[/green] {resume_path} (paso {paso_reanudado:,})"
        )
    elif args.bc_weights:
        # BC (opcional): carga pesos pre-entrenados por imitación de PIMC. El
        # value_head queda en su init (PPO lo aprende). LR bajo para no destruir BC.
        import pickle as _pickle
        with open(args.bc_weights, "rb") as _f:
            _bc = _pickle.load(_f)
        algo.get_policy().set_weights(_bc["weights"])
        console.print(f"[green]Política inicializada desde BC:[/green] {args.bc_weights}")

    # Inyectar factory inicial (en la FASE correcta si se reanuda).
    # Progress del curriculum. Con --progress-fino es RELATIVO al tramo de este
    # run (fix del bug v10d: resume a 32.44M/35M daba progress=0.93 → todo el
    # fine-tune atrapado en Fase 4 self-play, exposición humana real 1/6).
    _prog_base = paso_reanudado if args.progress_fino else 0
    _prog_total = max(1, args.total_steps - _prog_base)

    def _progress_de(pasos: int) -> float:
        return max(0.0, (pasos - _prog_base)) / _prog_total

    factory_inicial = pool.make_factory(progress=_progress_de(paso_reanudado))
    try:
        algo.env_runner_group.foreach_env(
            lambda env: setattr(env, "_opponent_factory", factory_inicial)
        )
    except Exception as _exc:
        console.print(f"[yellow]Aviso inyectando factory inicial: {_exc}[/yellow]")

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

    pasos_totales = paso_reanudado
    ultimo_snapshot = paso_reanudado
    ultimo_factory_refresh = paso_reanudado
    fase_actual = _fase(_progress_de(paso_reanudado))
    iter_count = 0
    reward_medio = float("nan")
    reward_max = float("nan")
    ep_len = float("nan")
    num_snapshots = len(listar_snapshots(snapshot_dir))
    snapshots_desde_ultima_eval = 0

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while pasos_totales < args.total_steps:
                result = algo.train()
                iter_count += 1

                pasos_totales = result.get("timesteps_total", pasos_totales)
                progress = _progress_de(pasos_totales)

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
                                policy, obs_dim=args.obs_dim, n_partidas=150,
                                con_pase=args.con_pase,
                            )
                            # RAIL humano: si hay clon humano-BC, medir también
                            # win-rate contra 3 clones — el proxy de humanos. Sin
                            # esto, el score de elite es 100% métrica de bots (la
                            # burbuja) y preserva regresiones vs humanos como la
                            # de v10d (0.46→0.30). Ver docs/auditoria_moon_*.md.
                            win_humano = None
                            if args.humano_bc:
                                from src.rllib.eval_bots import (
                                    _eval_model_vs_factory, _obtener_modelo, _agregar,
                                )
                                from src.rllib.opponent_pool import SnapshotPolicy
                                import numpy as _np
                                _d = _np.load(args.humano_bc)
                                _pesos_h = {k: _d[k] for k in _d.files}
                                _temp = args.temp_humano if args.temp_humano > 0 else None

                                def _fac_humana(ai=0):
                                    return {i: SnapshotPolicy.from_weights(
                                                _pesos_h, obs_dim=args.obs_dim,
                                                temperatura=_temp)
                                            for i in range(4) if i != ai}

                                _model_eval = _obtener_modelo(policy, args.obs_dim)
                                _res_h = _eval_model_vs_factory(
                                    _model_eval, _fac_humana, 150,
                                    obs_dim=args.obs_dim, con_pase=args.con_pase,
                                    moon_dir=args.moon_dir or RUTA_MOON,
                                )
                                win_humano = _agregar(_res_h)["win"]
                                metricas_bot["win_rate_vs_humano"] = round(win_humano, 4)

                            metricas_bot["tipo"] = "bot_eval"
                            metricas_bot["paso"] = pasos_totales
                            metricas_bot["fase"] = fase_actual
                            with open(bot_eval_log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(metricas_bot) + "\n")

                            # Preservar el mejor modelo (elite) — protege contra
                            # el pruning y contra la regresión del self-play.
                            # Con clon humano: el score pesa el proxy de humanos
                            # (el objetivo real); sin él, score legacy de bots.
                            if win_humano is not None:
                                score = (
                                    0.5 * win_humano
                                    + 0.3 * metricas_bot.get("top2_rate_vs_experto", 0.0)
                                    + 0.2 * metricas_bot.get("top2_rate", 0.0)
                                )
                            else:
                                score = (
                                    0.5 * metricas_bot.get("top2_rate_vs_experto", 0.0)
                                    + 0.3 * metricas_bot.get("win_rate_vs_experto", 0.0)
                                    + 0.2 * metricas_bot.get("top2_rate", 0.0)
                                )
                            if preservar_elite(ruta, elite_dir, score,
                                               pasos_totales, max_elite=10):
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
                    try:
                        algo.env_runner_group.foreach_env(
                            lambda env: setattr(env, "_opponent_factory", new_factory)
                        )
                    except Exception as _exc:
                        console.print(
                            f"[yellow]Aviso refrescando factory (continuo): {_exc}[/yellow]"
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
