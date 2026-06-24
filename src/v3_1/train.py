"""
Entrenamiento autónomo v3.1 — MLP + 228 dims + MCTS + Self-Play con BotExperto.

Pipeline de entrenamiento:
  - MLPFeatureExtractorV31 (3 capas: 512→256→128)
  - MCTSBuffer con oráculo PIMC (Expert Iteration)
  - Fictitious Self-Play: 1 BotExperto fijo + snapshots + bots heurísticos
  - Decaimiento progresivo de prob_bot (cosine decay)
  - BC fine-tune: 2 epochs sobre buffer MCTS
  - Evaluación periódica (formato estándar + fácil)

Cambios vs v3:
  - 228 dims (vs 265): sin duplicados, + patrones de rivales
  - MLP (vs Transformer): más estable, más rápido
  - Sin BC reg (diagnóstico: anclaba al modelo)

Uso:
    python -m src.v3_1.train --total-steps 2000000 --output-dir models/v3_1
    python -m src.v3_1.train --resume models/v3_1/snapshots/snapshot_bc_pretrained
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

_proyecto = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _proyecto not in sys.path:
    sys.path.insert(0, _proyecto)


# ------------------------------------------------------------------
# Constantes v3.1
# ------------------------------------------------------------------
PROB_BOT_START: float = 0.50
PROB_BOT_END: float = 0.05
EVAL_PARTIDAS: int = 200
ELO_PARTIDAS: int = 30
ELO_MAX_SNAPSHOTS: int = 12
BEST_TOP: int = 2
MIN_SNAPSHOT_STEPS: int = 100_000
MAX_SNAPSHOTS_POOL: int = 50
MIN_BC_SAMPLES: int = 100
MCTS_FREQUENCY: float = 0.60

# ── Cosine LR Schedule ──
LR_MAX: float = 1e-4    # LR al inicio (rápido aprendizaje de reglas)
LR_MIN: float = 1e-6    # LR al final (fine-tuning sin olvidar)


def cosine_lr_schedule(progress_remaining: float) -> float:
    """Cosine annealing: LR_MAX → LR_MIN sobre el progreso total.

    Args:
        progress_remaining: Fracción restante del entrenamiento [1.0 → 0.0].

    Returns:
        Learning rate actual según cosine schedule.
    """
    progress = 1.0 - progress_remaining
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------------
# Utilidades: barra de progreso
# ------------------------------------------------------------------

def _progress_bar(
    current: int, total: int, width: int = 36,
    steps_per_sec: float = 0.0, prefix: str = "",
) -> str:
    if total <= 0:
        ratio = 1.0
    else:
        ratio = min(current / total, 1.0)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    pct = ratio * 100
    parts = [f"{prefix} [{bar}] {current:,}/{total:,} ({pct:.0f}%)"]
    if steps_per_sec > 0:
        remaining = total - current
        if remaining > 0:
            eta_sec = remaining / steps_per_sec
            if eta_sec < 60:
                eta_str = f"{eta_sec:.0f}s"
            elif eta_sec < 3600:
                eta_str = f"{eta_sec/60:.0f}m"
            else:
                eta_str = f"{eta_sec/3600:.1f}h"
            parts.append(f" | {steps_per_sec:,.0f} st/s | ETA {eta_str}")
        else:
            parts.append(f" | {steps_per_sec:,.0f} st/s")
    return "".join(parts)


class _ProgressBarCallback(BaseCallback):
    def __init__(self, total: int, label: str = "Chunk", log_every: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self._total = total
        self._label = label
        self._log_every = log_every
        self._start_ts: int = 0
        self._start_time: float = 0.0
        self._iter_count: int = 0
        self._last_metrics: Dict[str, float] = {}

    def _on_training_start(self) -> None:
        self._start_ts = self.model.num_timesteps
        self._start_time = time.time()
        self._iter_count = 0

    def _on_step(self) -> bool:
        self._iter_count += 1
        current = self.model.num_timesteps - self._start_ts
        elapsed = time.time() - self._start_time
        steps_per_sec = current / elapsed if elapsed > 0 else 0

        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                self._last_metrics = {
                    k: v for k, v in self.model.logger.name_to_value.items()
                    if not k.startswith("time/")
                }
            except Exception:
                pass

        if self._iter_count % self._log_every == 0 or current >= self._total:
            bar = _progress_bar(min(current, self._total), self._total,
                                steps_per_sec=steps_per_sec, prefix=f"  {self._label}")
            metrics_str = self._format_metrics()
            print(f"\r{bar}{metrics_str}", end="", flush=True)
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._start_time
        current = self.model.num_timesteps - self._start_ts
        steps_per_sec = current / elapsed if elapsed > 0 else 0
        bar = _progress_bar(current, self._total, steps_per_sec=steps_per_sec,
                            prefix=f"  {self._label}")
        print(f"\r{bar}")

    def _format_metrics(self) -> str:
        if not self._last_metrics:
            return ""
        key_metrics = [
            ("loss", "loss"), ("entropy_loss", "ent"), ("value_loss", "vloss"),
            ("approx_kl", "kl"), ("clip_fraction", "clip"),
            ("explained_variance", "ev"),
        ]
        parts = []
        for full_key, short_key in key_metrics:
            val = self._last_metrics.get(f"train/{full_key}")
            if val is not None:
                if abs(val) < 0.01:
                    parts.append(f"{short_key}={val:.2e}")
                elif abs(val) < 1:
                    parts.append(f"{short_key}={val:.3f}")
                else:
                    parts.append(f"{short_key}={val:.2f}")
        return " | " + " ".join(parts) if parts else ""


# ------------------------------------------------------------------
# prob_bot decay
# ------------------------------------------------------------------

def prob_bot_actual(
    paso_actual: int, total_pasos: int,
    inicio: float = PROB_BOT_START, fin: float = PROB_BOT_END,
) -> float:
    if total_pasos <= 0:
        return fin
    progreso = min(paso_actual / total_pasos, 1.0)
    return fin + 0.5 * (inicio - fin) * (1.0 + math.cos(math.pi * progreso))


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def log_eval(
    log_path: str, paso: int, wr: float, avg_score: float,
    num_partidas: int, prob_bot: float,
    posiciones: Optional[List[int]] = None,
    scores_por_jugador: Optional[Dict[str, float]] = None,
    **kwargs,
) -> None:
    entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "paso": paso, "wr": round(wr, 4),
        "avg_score": round(avg_score, 2),
        "num_partidas": num_partidas,
        "prob_bot": round(prob_bot, 4),
    }
    if posiciones is not None:
        entry["posiciones"] = [int(p) for p in posiciones]
        total = sum(posiciones)
        if total > 0:
            if "top1_rate" not in kwargs:
                entry["top1_rate"] = round(posiciones[0] / total, 4)
            if "top2_rate" not in kwargs:
                entry["top2_rate"] = round(
                    (posiciones[0] + posiciones[1]) / total, 4)
    for key in ("top1_rate", "top2_rate", "formato"):
        if key in kwargs and kwargs[key] is not None:
            entry[key] = kwargs[key]
    if scores_por_jugador is not None:
        entry["scores_por_jugador"] = {
            k: round(v, 2) for k, v in scores_por_jugador.items()
        }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_bc_loss(log_path: str, paso: int, bc_loss: float, buffer_size: int) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tipo": "bc_finetune", "paso": paso,
        "bc_loss": round(bc_loss, 6), "buffer_size": buffer_size,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Self-play
# ------------------------------------------------------------------

def _cargar_snapshots_v31(snaps_dir: str, min_steps: int, max_snapshots: int) -> list:
    from sb3_contrib import MaskablePPO

    if not os.path.exists(snaps_dir):
        return []
    archivos = glob.glob(os.path.join(snaps_dir, "snapshot_*.zip"))
    archivos_validos = []
    for a in archivos:
        nombre = os.path.basename(a).replace(".zip", "")
        try:
            paso = int(nombre.replace("snapshot_", ""))
            if paso >= min_steps:
                archivos_validos.append((paso, a))
        except ValueError:
            continue
    archivos_validos.sort(key=lambda x: x[0])
    if len(archivos_validos) > max_snapshots:
        archivos_validos = archivos_validos[-max_snapshots:]

    snapshots = []
    for _, path in archivos_validos:
        try:
            model = MaskablePPO.load(path, device="cpu")
            snapshots.append(model)
        except Exception:
            continue
    return snapshots


def crear_entorno_self_play_v31(
    version: str = "v3_1",
    prob_bot: float = 0.50,
    min_steps: int = MIN_SNAPSHOT_STEPS,
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
    agente_idx: int = 0,
    modelo_actual: Optional[Any] = None,
    fase: int = 0,
    oracle_buffer: Optional[object] = None,
    oracle_rng: Optional[np.random.Generator] = None,
):
    """Crea un entorno con oponentes según la fase de curriculum.

    Fase 0 (calentamiento): [Modelo, Bot, Bot, Bot]
    Fase 1 (self-play incipiente): [Modelo, Modelo_old, Bot, Bot]
    Fase 2 (self-play dominante): [Modelo, Modelo, Modelo_old, Bot]

    En fases 1-2, Modelo_old es un snapshot histórico que representa
    la versión anterior del modelo. Esto crea una dinámica de brazos
    evolutivos: el modelo actual debe aprender a explotar los errores
    de su versión pasada.
    """
    from src.agentes.bot_experto import BotExperto
    from src.agentes.politica_rl import PoliticaSB3
    from src.agentes.heuristicos import BOTS_DISPONIBLES
    from src.v3_1.entorno import CorazonesEnvV31

    snaps_dir = os.path.join("models", version, "snapshots")
    snapshots = _cargar_snapshots_v31(snaps_dir, min_steps, max_snapshots)

    politicas: Dict[int, object] = {}
    rivales = [(agente_idx + offset) % 4 for offset in (1, 2, 3)]

    if fase == 0:
        # ── Fase 0: calentamiento contra bots ──
        politicas[rivales[0]] = random.choice(BOTS_DISPONIBLES)
        politicas[rivales[1]] = random.choice(BOTS_DISPONIBLES)
        politicas[rivales[2]] = random.choice(BOTS_DISPONIBLES)

    elif fase == 1:
        # ── Fase 1: 1 snapshot viejo + 2 bots ──
        if snapshots and modelo_actual is not None:
            viejo = random.choice(snapshots)
            politicas[rivales[0]] = PoliticaSB3(viejo, rivales[0])
        else:
            politicas[rivales[0]] = BotExperto()
        politicas[rivales[1]] = random.choice(BOTS_DISPONIBLES)
        politicas[rivales[2]] = random.choice(BOTS_DISPONIBLES)

    else:
        # ── Fase 2: modelo actual + snapshot viejo + bot ──
        if modelo_actual is not None:
            politicas[rivales[0]] = PoliticaSB3(modelo_actual, rivales[0])
        else:
            politicas[rivales[0]] = BotExperto()
        if snapshots:
            viejo = random.choice(snapshots)
            politicas[rivales[1]] = PoliticaSB3(viejo, rivales[1])
        else:
            politicas[rivales[1]] = BotExperto()
        politicas[rivales[2]] = random.choice(BOTS_DISPONIBLES)

    return CorazonesEnvV31(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
        oracle_buffer=oracle_buffer,
        oracle_rng=oracle_rng,
    )


# ------------------------------------------------------------------
# Guardado de snapshot
# ------------------------------------------------------------------

def _guardar_snapshot(modelo, paso: int, output_dir: str,
                      vecnorm_path: Optional[str] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, f"snapshot_{paso:010d}")
    modelo.save(snapshot_path)
    print(f"  📦 Snapshot guardado: {snapshot_path}.zip")
    if vecnorm_path and os.path.exists(vecnorm_path):
        import shutil
        dst = os.path.join(output_dir, f"snapshot_{paso:010d}_vecnorm.pkl")
        shutil.copy2(vecnorm_path, dst)
        print(f"  📊 VecNormalize guardado: {dst}")


# ------------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------------

def _evaluar_estandar(modelo, num_partidas: int = EVAL_PARTIDAS,
                      seed: int = 42):
    """Evalúa contra [Experto, Experto, BotRotativo]."""
    from src.v3_1.entorno import CorazonesEnvV31
    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import BOTS_DISPONIBLES

    victorias = 0
    scores: list[float] = []
    posiciones = [0, 0, 0, 0]
    scores_por_tipo: Dict[str, list] = {
        "modelo": [], "experto_1": [], "experto_2": [], "bot": []}

    for i in range(num_partidas):
        agente_idx = i % 4
        rng = np.random.default_rng(seed + i)

        politicas: Dict[int, object] = {}
        oponentes = [(agente_idx + d) % 4 for d in (1, 2, 3)]
        politicas[oponentes[0]] = BotExperto()
        politicas[oponentes[1]] = BotExperto()
        politicas[oponentes[2]] = random.choice(BOTS_DISPONIBLES)

        env = CorazonesEnvV31(agente_idx=agente_idx,
                              politicas_oponentes=politicas)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        terminated, truncated = False, False

        while not terminated and not truncated:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        puntos_mano = env.motor.calcular_puntuacion_mano()
        puntos_agente = puntos_mano[agente_idx]
        scores.append(puntos_agente)
        if puntos_agente <= 8:
            victorias += 1

        ranking = sorted(enumerate(puntos_mano), key=lambda x: x[1])
        for pos, (jug, _) in enumerate(ranking):
            if jug == agente_idx:
                posiciones[pos] += 1

        scores_por_tipo["modelo"].append(puntos_agente)
        scores_por_tipo["experto_1"].append(puntos_mano[oponentes[0]])
        scores_por_tipo["experto_2"].append(puntos_mano[oponentes[1]])
        scores_por_tipo["bot"].append(puntos_mano[oponentes[2]])
        env.close()

    wr = victorias / num_partidas
    avg_score = float(np.mean(scores))
    top1_rate = posiciones[0] / num_partidas
    top2_rate = (posiciones[0] + posiciones[1]) / num_partidas
    avg_por_tipo = {k: float(np.mean(v)) if v else 0.0 for k,
                    v in scores_por_tipo.items()}
    return wr, avg_score, posiciones, avg_por_tipo, top1_rate, top2_rate


def _evaluar_facil(modelo, num_partidas: int = 50, seed: int = 42):
    """Evalúa contra [Experto, Bot, Bot]."""
    from src.v3_1.entorno import CorazonesEnvV31
    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import BOTS_DISPONIBLES

    victorias = 0
    scores: list[float] = []
    posiciones = [0, 0, 0, 0]
    scores_por_tipo: Dict[str, list] = {
        "modelo": [], "experto": [], "bot_1": [], "bot_2": []}

    for i in range(num_partidas):
        agente_idx = i % 4
        rng = np.random.default_rng(seed + i)

        politicas: Dict[int, object] = {}
        oponentes = [(agente_idx + d) % 4 for d in (1, 2, 3)]
        politicas[oponentes[0]] = BotExperto()
        politicas[oponentes[1]] = random.choice(BOTS_DISPONIBLES)
        politicas[oponentes[2]] = random.choice(BOTS_DISPONIBLES)

        env = CorazonesEnvV31(agente_idx=agente_idx,
                              politicas_oponentes=politicas)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        terminated, truncated = False, False

        while not terminated and not truncated:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        puntos_mano = env.motor.calcular_puntuacion_mano()
        puntos_agente = puntos_mano[agente_idx]
        scores.append(puntos_agente)
        if puntos_agente <= 8:
            victorias += 1

        ranking = sorted(enumerate(puntos_mano), key=lambda x: x[1])
        for pos, (jug, _) in enumerate(ranking):
            if jug == agente_idx:
                posiciones[pos] += 1

        scores_por_tipo["modelo"].append(puntos_agente)
        scores_por_tipo["experto"].append(puntos_mano[oponentes[0]])
        scores_por_tipo["bot_1"].append(puntos_mano[oponentes[1]])
        scores_por_tipo["bot_2"].append(puntos_mano[oponentes[2]])
        env.close()

    wr = victorias / num_partidas
    avg_score = float(np.mean(scores))
    top1_rate = posiciones[0] / num_partidas
    top2_rate = (posiciones[0] + posiciones[1]) / num_partidas
    avg_por_tipo = {k: float(np.mean(v)) if v else 0.0 for k,
                    v in scores_por_tipo.items()}
    return wr, avg_score, posiciones, avg_por_tipo, top1_rate, top2_rate


# ------------------------------------------------------------------
# Entrenamiento principal
# ------------------------------------------------------------------

def entrenar_auto(
    total_steps: int = 10_000_000,
    snapshot_every: int = 100_000,
    eval_every: int = 100_000,
    elo_every: int = 20,
    eval_partidas: int = EVAL_PARTIDAS,
    prob_bot_start: float = PROB_BOT_START,
    prob_bot_end: float = PROB_BOT_END,
    seed: int = 42,
    device: str = "cpu",
    resume_from: Optional[str] = None,
    output_dir: str = "models/v3_1",
    best_top: int = BEST_TOP,
    mcts_enabled: bool = False,
    mcts_frequency: float = 0.05,
    bc_dataset_path: Optional[str] = None,
) -> int:
    """Ejecuta entrenamiento autónomo v3.1 con MLP + self-play.

    Args:
        total_steps: Pasos totales a entrenar.
        snapshot_every: Guardar snapshot cada N pasos.
        eval_every: Evaluar cada N pasos.
        elo_every: Torneo Elo cada N snapshots.
        eval_partidas: Partidas por evaluación.
        prob_bot_start: prob_bot inicial.
        prob_bot_end: prob_bot final.
        seed: Semilla aleatoria.
        device: Dispositivo (cpu, cuda, dml).
        resume_from: Ruta a snapshot para reanudar.
        output_dir: Directorio de salida.
        best_top: Cuántos mejores snapshots guardar.
        mcts_enabled: Activar MCTS oracle guidance.
        mcts_frequency: Fracción de episodios con MCTS.
        bc_dataset_path: Dataset para BC pre-training.

    Returns:
        Paso final alcanzado.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.v3_1.train_mcts import MCTSBuffer, entrenar_bc_epoch, entrenar_bc_dataset
    from src.v3_1.red import obtener_policy_kwargs_v31

    # Directorios
    dir_snapshots = os.path.join(output_dir, "snapshots")
    dir_vecnorm = os.path.join(output_dir, "vecnorm")
    dir_best = os.path.join(output_dir, "best")
    os.makedirs(dir_snapshots, exist_ok=True)
    os.makedirs(dir_vecnorm, exist_ok=True)
    os.makedirs(dir_best, exist_ok=True)

    eval_log_path = os.path.join(output_dir, "eval_log.jsonl")
    version_name = os.path.basename(output_dir)
    step_counter = 0
    snapshot_count = 0

    # ── Carga / creación de modelo ──
    if resume_from:
        if not resume_from.endswith(".zip"):
            resume_from += ".zip"
        ruta = resume_from if os.path.isabs(
            resume_from) else os.path.join(_proyecto, resume_from)
        if not os.path.exists(ruta):
            print(f"ERROR: No se encuentra {ruta}")
            sys.exit(1)

        step_counter = _extraer_paso(ruta)
        print("=" * 60)
        print(f"🔄 REANUDANDO v3.1 desde snapshot paso {step_counter:,}")
        print("=" * 60)

        modelo = MaskablePPO.load(ruta, device=device)

        def _make_env():
            progress = step_counter / total_steps
            return crear_entorno_self_play_v31(
                version=version_name,
                prob_bot=prob_bot_actual(
                    step_counter, total_steps, prob_bot_start, prob_bot_end),
                agente_idx=random.randint(0, 3),
                modelo_actual=modelo if progress >= 0.30 else None,
                fase=(0 if progress < 0.10 else
                      1 if progress < 0.30 else 2),
            )

        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        modelo.set_env(venv)
        snapshot_count = step_counter // snapshot_every
    else:
        print("=" * 60)
        print(f"🚀 ENTRENANDO v3.1 desde cero ({total_steps:,} pasos)")
        print(f"   Observación: 228 dims | MLP [512,256,128] | Multi-position")
        print(
            f"   LR: cosine {LR_MAX:.0e}→{LR_MIN:.0e} | ent_coef=2e-3 | terminal-only reward")
        print(f"   Fase 0 (0-10%): Bots | Fase 1 (10-30%): +Snapshot")
        print(f"   Fase 2 (30%+): Self-play 4 posiciones (shared weights)")
        print(f"   Device: {device} | Output: {output_dir}")
        print("=" * 60)

        policy_kwargs = obtener_policy_kwargs_v31(features_dim=128)

        def _make_env():
            return crear_entorno_self_play_v31(
                version=version_name, prob_bot=prob_bot_start, agente_idx=0,
                modelo_actual=None, fase=0)

        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)

        modelo = MaskablePPO(
            "MlpPolicy", venv, policy_kwargs=policy_kwargs,
            verbose=0, device=device, seed=seed,
            tensorboard_log=os.path.join(output_dir, "logs"),
            learning_rate=cosine_lr_schedule,  # 1e-4 → 1e-6 cosine annealing
            n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=2e-3,
            vf_coef=0.25, max_grad_norm=0.5,
        )

    # ── SIN BC pre-training (curriculum puro desde cero) ──

    # ── MCTS buffer ──
    mcts_buffer = MCTSBuffer(capacity=100_000) if mcts_enabled else None

    # ── Training loop ──
    vecnorm_pkl = os.path.join(dir_vecnorm, "v31_vecnorm.pkl")
    pasos_restantes = total_steps - step_counter
    start_time = time.time()

    while step_counter < total_steps:
        # ── Determinar fase del curriculum ──
        progress = step_counter / total_steps
        if progress < 0.10:          # 0-10%: calentamiento contra bots
            fase = 0
        elif progress < 0.30:        # 10-30%: self-play incipiente
            fase = 1
        else:                         # 30-100%: self-play dominante
            fase = 2

        # ── Rotación de posición: el modelo aprende desde las 4 sillas ──
        agente_idx = random.randint(0, 3)

        pb = prob_bot_actual(step_counter, total_steps,
                             prob_bot_start, prob_bot_end)
        enviar_oraculo = (
            mcts_enabled and mcts_buffer is not None
            and random.random() < mcts_frequency
        )

        def _make_env_dyn():
            return crear_entorno_self_play_v31(
                version=version_name, prob_bot=pb,
                agente_idx=agente_idx,
                modelo_actual=modelo if fase >= 2 else None,
                fase=fase,
                oracle_buffer=mcts_buffer if enviar_oraculo else None,
                oracle_rng=np.random.default_rng(
                    seed + step_counter) if enviar_oraculo else None,
            )

        env_actual = modelo.get_env()
        if isinstance(env_actual, VecNormalize):
            env_actual.venv = DummyVecEnv([_make_env_dyn])
            _ = env_actual.reset()

        chunk = min(snapshot_every, pasos_restantes)
        pb_callback = _ProgressBarCallback(
            total=chunk, label=f"Snap {snapshot_count + 1:02d}", log_every=5)
        modelo.learn(total_timesteps=chunk, reset_num_timesteps=False,
                     log_interval=1, callback=pb_callback, progress_bar=False)
        step_counter += chunk
        pasos_restantes -= chunk
        snapshot_count += 1

        # ── BC fine-tune DESACTIVADO (daña rendimiento con terminal-only rewards) ──

        elapsed = time.time() - start_time
        steps_per_sec = step_counter / elapsed if elapsed > 0 else 0
        bar_global = _progress_bar(
            step_counter, total_steps, steps_per_sec=steps_per_sec, prefix="  Total")
        print(f"\n{'─'*60}")
        print(f"📊 {bar_global} | prob_bot={pb:.3f}")
        print(f"{'─'*60}")

        # Guardar VecNormalize
        env_actual = modelo.get_env()
        if isinstance(env_actual, VecNormalize):
            os.makedirs(os.path.dirname(vecnorm_pkl), exist_ok=True)
            env_actual.save(vecnorm_pkl)

        # Guardar snapshot
        _guardar_snapshot(modelo, step_counter, dir_snapshots,
                          vecnorm_pkl if os.path.exists(vecnorm_pkl) else None)

        # ── Evaluar ──
        if step_counter > 0 and step_counter % eval_every == 0:
            print(f"\n📈 Evaluando ({eval_partidas} manos)...")

            wr, score, pos, spj, t1, t2 = _evaluar_estandar(
                modelo, eval_partidas, seed + step_counter)
            log_eval(eval_log_path, step_counter, wr, score, eval_partidas, pb,
                     posiciones=pos, scores_por_jugador=spj, top1_rate=t1, top2_rate=t2, formato="dificil")
            pos_str = " > ".join(f"{p}×{c}" for p, c in zip(
                ["1°", "2°", "3°", "4°"], pos))
            print(
                f"   DIFICIL [M,E,E,B]: WR≤8={wr:.0%} Top1={t1:.0%} Top2={t2:.0%} Score={score:.1f} | {pos_str}")

            wr2, score2, pos2, spj2, t1_2, t2_2 = _evaluar_facil(
                modelo, eval_partidas // 2, seed + step_counter + 1)
            log_eval(eval_log_path, step_counter, wr2, score2, eval_partidas // 2, pb,
                     posiciones=pos2, scores_por_jugador=spj2, top1_rate=t1_2, top2_rate=t2_2, formato="facil")
            pos_str2 = " > ".join(f"{p}×{c}" for p, c in zip(
                ["1°", "2°", "3°", "4°"], pos2))
            print(
                f"   FACIL   [M,E,b,b]: WR≤8={wr2:.0%} Top1={t1_2:.0%} Top2={t2_2:.0%} Score={score2:.1f} | {pos_str2}")

    # ── Fin ──
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Entrenamiento v3.1 COMPLETADO en {total_time/3600:.1f}h")
    print(f"   Último snapshot: paso {step_counter:,}")
    print("=" * 60)
    modelo.save(os.path.join(dir_snapshots, f"snapshot_{step_counter:010d}"))
    return step_counter


def _extraer_paso(ruta: str) -> int:
    import re
    match = re.search(r'snapshot_(\d+)', ruta)
    return int(match.group(1)) if match else 0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Entrenamiento v3.1 — MLP + 228 dims")
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--snapshot-every", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--elo-every", type=int, default=20)
    parser.add_argument("--eval-partidas", type=int, default=EVAL_PARTIDAS)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "dml", "xpu"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/v3_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcts", action="store_true")
    parser.add_argument("--mcts-frequency", type=float, default=MCTS_FREQUENCY)
    parser.add_argument("--bc-dataset", type=str, default=None)
    args = parser.parse_args()

    entrenar_auto(
        total_steps=args.total_steps,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        elo_every=args.elo_every,
        eval_partidas=args.eval_partidas,
        seed=args.seed,
        device=args.device,
        resume_from=args.resume,
        output_dir=args.output_dir,
        mcts_enabled=args.mcts,
        mcts_frequency=args.mcts_frequency,
        bc_dataset_path=args.bc_dataset,
    )
