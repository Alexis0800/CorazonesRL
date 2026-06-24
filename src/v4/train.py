"""
Entrenamiento autónomo v4 — Self-play puro con bots heurísticos.

Pipeline:
  - MLPFeatureExtractorV31 (3 capas: 512→256→128)
  - Terminal-only rewards (26 - puntos)
  - Cosine LR: 1e-4 → 1e-6
  - Multi-position rotation (4 sillas, shared weights)
  - Curriculum 3 fases: bots → snapshots → self-play
  - Sin BC pre-training, sin MCTS BC fine-tune
  - Evaluación contra BotExperto (DIFICIL/FACIL)

Uso:
    python -m src.v4.train --total-steps 10000000 --output-dir models/v4
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
# Constantes v4
# ------------------------------------------------------------------
EVAL_PARTIDAS: int = 200
MIN_SNAPSHOT_STEPS: int = 100_000
MAX_SNAPSHOTS_POOL: int = 50
MCTS_FREQUENCY: float = 0.60

LR_MAX: float = 1e-4
LR_MIN: float = 1e-6


def cosine_lr_schedule(progress_remaining: float) -> float:
    """Cosine annealing: LR_MAX → LR_MIN."""
    progress = 1.0 - progress_remaining
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------

def _progress_bar(current: int, total: int, width: int = 36,
                  steps_per_sec: float = 0.0, prefix: str = "") -> str:
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
    def __init__(self, total: int, label: str = "Chunk", log_every: int = 5):
        super().__init__()
        self._total = total
        self._label = label
        self._log_every = log_every
        self._start_time: float = 0.0
        self._iter_count: int = 0
        self._last_metrics: Dict[str, float] = {}

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._iter_count = 0

    def _on_step(self) -> bool:
        self._iter_count += 1
        current = self.model.num_timesteps - self.model.num_timesteps
        # Recalculate from scratch
        elapsed = time.time() - self._start_time
        steps_per_sec = self._iter_count / elapsed if elapsed > 0 else 0

        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                self._last_metrics = {
                    k: v for k, v in self.model.logger.name_to_value.items()
                    if not k.startswith("time/")
                }
            except Exception:
                pass

        if self._iter_count % self._log_every == 0 or self._iter_count >= self._total:
            bar = _progress_bar(min(self._iter_count, self._total), self._total,
                                steps_per_sec=steps_per_sec, prefix=f"  {self._label}")
            metrics_str = self._format_metrics()
            print(f"\r{bar}{metrics_str}", end="", flush=True)
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._start_time
        steps_per_sec = self._iter_count / elapsed if elapsed > 0 else 0
        bar = _progress_bar(self._iter_count, self._total,
                            steps_per_sec=steps_per_sec, prefix=f"  {self._label}")
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
# Logging
# ------------------------------------------------------------------

def log_eval(log_path: str, paso: int, wr: float, avg_score: float,
             num_partidas: int, posiciones=None, scores_por_jugador=None, **kwargs) -> None:
    entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "paso": paso, "wr": round(wr, 4),
        "avg_score": round(avg_score, 2),
        "num_partidas": num_partidas,
    }
    if posiciones is not None:
        entry["posiciones"] = [int(p) for p in posiciones]
        total = sum(posiciones)
        if total > 0 and "top1_rate" not in kwargs:
            entry["top1_rate"] = round(posiciones[0] / total, 4)
            entry["top2_rate"] = round(
                (posiciones[0] + posiciones[1]) / total, 4)
    for key in ("top1_rate", "top2_rate", "formato"):
        if key in kwargs and kwargs[key] is not None:
            entry[key] = kwargs[key]
    if scores_por_jugador is not None:
        entry["scores_por_jugador"] = {
            k: round(v, 2) for k, v in scores_por_jugador.items()}
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Self-play — SOLO BOTS HEURÍSTICOS
# ------------------------------------------------------------------

def _cargar_snapshots(snaps_dir: str, min_steps: int, max_snapshots: int) -> list:
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
            snapshots.append(MaskablePPO.load(path, device="cpu"))
        except Exception:
            continue
    return snapshots


def crear_entorno_self_play_v4(
    version: str = "v4",
    agente_idx: int = 0,
    modelo_actual: Optional[Any] = None,
    fase: int = 0,
    oracle_buffer: Optional[object] = None,
    oracle_rng: Optional[np.random.Generator] = None,
):
    """Crea entorno con oponentes según fase. SOLO bots heurísticos.

    Fase 0 (calentamiento):     [Modelo, Bot, Bot, Bot]
    Fase 1 (self-play inicio):  [Modelo, Snapshot, Bot, Bot]
    Fase 2 (self-play total):   [Modelo, Modelo, Snapshot, Bot]

    Sin BotExperto en ningún momento. El modelo descubre estrategias
    avanzadas por sí mismo mediante self-play.
    """
    from src.agentes.politica_rl import PoliticaSB3
    from src.agentes.heuristicos import BOTS_DISPONIBLES
    from src.v4.entorno import CorazonesEnvV4

    snaps_dir = os.path.join("models", version, "snapshots")
    snapshots = _cargar_snapshots(
        snaps_dir, MIN_SNAPSHOT_STEPS, MAX_SNAPSHOTS_POOL)

    politicas: Dict[int, object] = {}
    rivales = [(agente_idx + offset) % 4 for offset in (1, 2, 3)]

    if fase == 0:
        # ── Fase 0: 3 bots heurísticos ──
        for r in rivales:
            politicas[r] = random.choice(BOTS_DISPONIBLES)

    elif fase == 1:
        # ── Fase 1: 1 snapshot + 2 bots ──
        if snapshots:
            politicas[rivales[0]] = PoliticaSB3(
                random.choice(snapshots), rivales[0])
        else:
            politicas[rivales[0]] = random.choice(BOTS_DISPONIBLES)
        politicas[rivales[1]] = random.choice(BOTS_DISPONIBLES)
        politicas[rivales[2]] = random.choice(BOTS_DISPONIBLES)

    else:
        # ── Fase 2: modelo actual + snapshot + bot ──
        if modelo_actual is not None:
            politicas[rivales[0]] = PoliticaSB3(modelo_actual, rivales[0])
        else:
            politicas[rivales[0]] = random.choice(BOTS_DISPONIBLES)
        if snapshots:
            politicas[rivales[1]] = PoliticaSB3(
                random.choice(snapshots), rivales[1])
        else:
            politicas[rivales[1]] = random.choice(BOTS_DISPONIBLES)
        politicas[rivales[2]] = random.choice(BOTS_DISPONIBLES)

    return CorazonesEnvV4(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
        oracle_buffer=oracle_buffer,
        oracle_rng=oracle_rng,
    )


# ------------------------------------------------------------------
# Guardado
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
# Evaluación (contra BotExperto para medir progreso real)
# ------------------------------------------------------------------

def _evaluar_estandar(modelo, num_partidas: int = 100, seed: int = 42):
    """Evalúa contra [Experto, Experto, BotRotativo]."""
    from src.v4.entorno import CorazonesEnvV4
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

        env = CorazonesEnvV4(agente_idx=agente_idx,
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
    from src.v4.entorno import CorazonesEnvV4
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

        env = CorazonesEnvV4(agente_idx=agente_idx,
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
    eval_partidas: int = 100,
    seed: int = 42,
    device: str = "cpu",
    resume_from: Optional[str] = None,
    output_dir: str = "models/v4",
    mcts_enabled: bool = False,
) -> int:
    """Entrenamiento autónomo v4 — Self-play puro.

    Args:
        total_steps: Pasos totales (default 10M).
        snapshot_every: Guardar snapshot cada N pasos.
        eval_every: Evaluar cada N pasos.
        eval_partidas: Partidas por evaluación.
        seed: Semilla aleatoria.
        device: Dispositivo (cpu, cuda, dml).
        resume_from: Snapshot para reanudar.
        output_dir: Directorio de salida.
        mcts_enabled: Activar oráculo MCTS.

    Returns:
        Paso final alcanzado.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.v3_1.train_mcts import MCTSBuffer
    from src.v3_1.red import obtener_policy_kwargs_v31

    dir_snapshots = os.path.join(output_dir, "snapshots")
    dir_vecnorm = os.path.join(output_dir, "vecnorm")
    os.makedirs(dir_snapshots, exist_ok=True)
    os.makedirs(dir_vecnorm, exist_ok=True)

    eval_log_path = os.path.join(output_dir, "eval_log.jsonl")
    version_name = os.path.basename(output_dir)
    step_counter = 0
    snapshot_count = 0

    # ── Creación de modelo ──
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
        print(f"🔄 REANUDANDO v4 desde snapshot paso {step_counter:,}")
        print("=" * 60)
        modelo = MaskablePPO.load(ruta, device=device)

        def _make_env():
            progress = step_counter / total_steps
            return crear_entorno_self_play_v4(
                version=version_name,
                agente_idx=random.randint(0, 3),
                modelo_actual=modelo if progress >= 0.30 else None,
                fase=(0 if progress < 0.10 else 1 if progress < 0.30 else 2),
            )
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        modelo.set_env(venv)
        snapshot_count = step_counter // snapshot_every
    else:
        print("=" * 60)
        print(f"🚀 ENTRENANDO v4 desde cero ({total_steps:,} pasos)")
        print(f"   228 dims | MLP [512,256,128] | Multi-position")
        print(f"   LR: cosine {LR_MAX:.0e}→{LR_MIN:.0e} | ent_coef=2e-3")
        print(f"   Terminal-only reward (26 - puntos)")
        print(f"   Fase 0 (0-10%): Bots heurísticos")
        print(f"   Fase 1 (10-30%): +Snapshots")
        print(f"   Fase 2 (30-100%): Self-play 4 posiciones")
        print(f"   SIN BotExperto | SIN BC pre-training | SIN BC fine-tune")
        print(f"   Device: {device} | Output: {output_dir}")
        print("=" * 60)

        policy_kwargs = obtener_policy_kwargs_v31(features_dim=128)

        def _make_env():
            return crear_entorno_self_play_v4(
                version=version_name, agente_idx=0, fase=0)

        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)

        modelo = MaskablePPO(
            "MlpPolicy", venv, policy_kwargs=policy_kwargs,
            verbose=0, device=device, seed=seed,
            tensorboard_log=os.path.join(output_dir, "logs"),
            learning_rate=cosine_lr_schedule,
            n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=2e-3,
            vf_coef=0.25, max_grad_norm=0.5,
        )

    # ── MCTS buffer (solo recolección, sin BC fine-tune) ──
    mcts_buffer = MCTSBuffer(capacity=100_000) if mcts_enabled else None

    # ── Training loop ──
    vecnorm_pkl = os.path.join(dir_vecnorm, "v4_vecnorm.pkl")
    pasos_restantes = total_steps - step_counter
    start_time = time.time()

    while step_counter < total_steps:
        progress = step_counter / total_steps
        if progress < 0.10:
            fase = 0
        elif progress < 0.30:
            fase = 1
        else:
            fase = 2

        agente_idx = random.randint(0, 3)
        enviar_oraculo = (
            mcts_enabled and mcts_buffer is not None
            and random.random() < MCTS_FREQUENCY
        )

        def _make_env_dyn():
            return crear_entorno_self_play_v4(
                version=version_name,
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

        elapsed = time.time() - start_time
        steps_per_sec = step_counter / elapsed if elapsed > 0 else 0
        bar_global = _progress_bar(
            step_counter, total_steps, steps_per_sec=steps_per_sec, prefix="  Total")
        print(f"\n{'─'*60}")
        print(f"📊 {bar_global}")
        print(f"{'─'*60}")

        # Guardar
        env_actual = modelo.get_env()
        if isinstance(env_actual, VecNormalize):
            os.makedirs(os.path.dirname(vecnorm_pkl), exist_ok=True)
            env_actual.save(vecnorm_pkl)
        _guardar_snapshot(modelo, step_counter, dir_snapshots,
                          vecnorm_pkl if os.path.exists(vecnorm_pkl) else None)

        # Evaluar
        if step_counter > 0 and step_counter % eval_every == 0:
            print(f"\n📈 Evaluando ({eval_partidas} manos)...")
            wr, score, pos, spj, t1, t2 = _evaluar_estandar(
                modelo, eval_partidas, seed + step_counter)
            log_eval(eval_log_path, step_counter, wr, score, eval_partidas,
                     posiciones=pos, scores_por_jugador=spj, top1_rate=t1, top2_rate=t2, formato="dificil")
            pos_str = " > ".join(f"{p}×{c}" for p, c in zip(
                ["1°", "2°", "3°", "4°"], pos))
            print(
                f"   DIFICIL [M,E,E,B]: WR≤8={wr:.0%} Top1={t1:.0%} Top2={t2:.0%} Score={score:.1f} | {pos_str}")

            wr2, score2, pos2, spj2, t1_2, t2_2 = _evaluar_facil(
                modelo, eval_partidas // 2, seed + step_counter + 1)
            log_eval(eval_log_path, step_counter, wr2, score2, eval_partidas // 2,
                     posiciones=pos2, scores_por_jugador=spj2, top1_rate=t1_2, top2_rate=t2_2, formato="facil")
            pos_str2 = " > ".join(f"{p}×{c}" for p, c in zip(
                ["1°", "2°", "3°", "4°"], pos2))
            print(
                f"   FACIL   [M,E,b,b]: WR≤8={wr2:.0%} Top1={t1_2:.0%} Top2={t2_2:.0%} Score={score2:.1f} | {pos_str2}")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Entrenamiento v4 COMPLETADO en {total_time/3600:.1f}h")
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
        description="Entrenamiento v4 — Self-play puro")
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--snapshot-every", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-partidas", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/v4")
    parser.add_argument("--mcts", action="store_true", default=False)

    args = parser.parse_args()
    entrenar_auto(
        total_steps=args.total_steps,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        eval_partidas=args.eval_partidas,
        seed=args.seed,
        device=args.device,
        resume_from=args.resume,
        output_dir=args.output_dir,
        mcts_enabled=args.mcts,
    )
