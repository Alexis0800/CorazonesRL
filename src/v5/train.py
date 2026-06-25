"""
Entrenamiento autónomo v5 — 4-model shared-weights self-play simultáneo.

Pipeline:
  - 4 entornos en paralelo (DummyVecEnv), uno por posición
  - El mismo modelo juega las 4 sillas simultáneamente
  - Cada posición recibe terminal-only reward (26 - puntos)
  - PPO actualiza con experiencias combinadas de las 4 posiciones
  - Zero-sum: rewards combinadas suman ~78 por mano
  - Cosine LR: 1e-4 → 1e-6
  - Curriculum 5 fases: destete gradual del BotExperto (ancla → snapshots → puro)

Ventajas vs v4:
  - 4× más datos por update (4 posiciones × 2048 steps)
  - El modelo aprende TODAS las estrategias (tirar Q♠, cazar Q♠, hacer pozo)
  - Sin necesidad de rotación manual

Uso:
    python -m src.v5.train --total-steps 20000000 --output-dir models/v5
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
# Constantes v5
# ------------------------------------------------------------------
EVAL_PARTIDAS: int = 200
MIN_SNAPSHOT_STEPS: int = 100_000
MAX_SNAPSHOTS_POOL: int = 50
MCTS_FREQUENCY: float = 0.60
EPSILON_MAX: float = 0.15   # Máxima exploración (inicio)
EPSILON_MIN: float = 0.01   # Mínima exploración (nunca 0)
EPSILON_DECAY: float = 3.0  # Rapidez de decaimiento exponencial
ENT_COEF: float = 1e-2       # Coeficiente de entropía (×3.3 vs v5 original)

LR_MAX: float = 1e-4
LR_MIN: float = 1e-6
N_STEPS: int = 2048  # total entre los 4 entornos (512 por posición)


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
    """Barra de progreso que muestra timesteps reales del modelo.

    Usa self.model.num_timesteps (contador oficial de SB3) en vez de
    contar llamadas a _on_step, porque _on_step se invoca por cada
    paso de entorno individual, no por iteración de PPO.
    """

    def __init__(self, total: int, label: str = "Chunk", log_every: int = 5):
        super().__init__()
        self._total = total
        self._label = label
        self._log_every = log_every
        self._start_time: float = 0.0
        self._prev_steps: int = 0
        self._last_log_steps: int = 0
        self._last_metrics: Dict[str, float] = {}
        # snapshot inicial para calcular solo los pasos de ESTE chunk
        self._snapshot_steps: int = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._snapshot_steps = self.model.num_timesteps
        self._prev_steps = 0
        self._last_log_steps = -self._log_every

    def _on_step(self) -> bool:
        current = self.model.num_timesteps - self._snapshot_steps
        self._prev_steps = current
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

        if current - self._last_log_steps >= self._log_every or current >= self._total:
            self._last_log_steps = current
            bar = _progress_bar(min(current, self._total), self._total,
                                steps_per_sec=steps_per_sec, prefix=f"  {self._label}")
            metrics_str = self._format_metrics()
            print(f"\r{bar}{metrics_str}", end="", flush=True)
        return True

    def _on_training_end(self) -> None:
        current = self.model.num_timesteps - self._snapshot_steps
        elapsed = time.time() - self._start_time
        steps_per_sec = current / elapsed if elapsed > 0 else 0
        bar = _progress_bar(min(current, self._total), self._total,
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

def log_eval(log_path: str, paso: int, resultados: List[Dict[str, Any]]) -> None:
    """Registra resultados de evaluación en JSONL (una línea por nivel)."""
    for r in resultados:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "paso": paso,
            "formato": r["tag"],
            "avg_score": round(r["avg_score"], 2),
            "top1_rate": round(r["top1_rate"], 4),
            "top2_rate": round(r["top2_rate"], 4),
            "num_partidas": 100,
            "posiciones": r["posiciones"],
            "scores_por_jugador": {k: round(v, 2) for k, v in r["scores"].items()},
        }
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Self-play — 4 entornos, shared model
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


def crear_entorno_self_play_v5(
    version: str = "v5",
    agente_idx: int = 0,
    modelo_actual: Optional[Any] = None,
    fase: int = 0,
    oracle_buffer: Optional[object] = None,
    oracle_rng: Optional[np.random.Generator] = None,
    epsilon: float = 0.0,
):
    """Crea UN entorno para una posición específica.

    Fases:
      0 (0-5%):    3 bots — warm-up mínimo
      1 (5-15%):   1 BotExperto + 2 bots — aprender rival fuerte
      2 (15-35%):  1 modelo + 1 snapshot + 1 BotExperto — ancla BotExperto
      3 (35-60%):  2 modelos + 1 snapshot — DESTETE: snapshots como ancla
      4 (60-100%): 3 modelos — self-play PURO sin BotExperto

    Destete gradual del BotExperto: se usa como ancla en Fases 1-2,
    luego se reemplaza por snapshots históricos en Fase 3, y finalmente
    self-play puro en Fase 4. Esto previene colapso sin imponer techo.

    Args:
        version: Nombre de versión para cargar snapshots.
        agente_idx: Posición del agente (0-3).
        modelo_actual: Modelo compartido (usado en fase 2/3 para oponentes).
        fase: 0-3 según tabla arriba.
        epsilon: Tasa de exploración epsilon-greedy.
    """
    from src.agentes.politica_rl import PoliticaSB3
    from src.agentes.heuristicos import BOTS_DISPONIBLES
    from src.agentes.bot_experto import BotExperto
    from src.v5.entorno import CorazonesEnvV5

    snaps_dir = os.path.join("models", version, "snapshots")
    snapshots = _cargar_snapshots(
        snaps_dir, MIN_SNAPSHOT_STEPS, MAX_SNAPSHOTS_POOL)

    politicas: Dict[int, object] = {}
    rivales = [(agente_idx + offset) % 4 for offset in (1, 2, 3)]

    if fase == 0:
        # ── 3 bots heurísticos (warm-up mínimo) ──
        for r in rivales:
            politicas[r] = random.choice(BOTS_DISPONIBLES)

    elif fase == 1:
        # ── 1 BotExperto + 2 bots (aprender de rival fuerte) ──
        politicas[rivales[0]] = BotExperto()
        politicas[rivales[1]] = random.choice(BOTS_DISPONIBLES)
        politicas[rivales[2]] = random.choice(BOTS_DISPONIBLES)

    elif fase == 2:
        # ── 1 modelo + 1 snapshot + 1 BotExperto (self-play con ancla) ──
        politicas[rivales[0]] = PoliticaSB3(modelo_actual, rivales[0])
        if snapshots:
            politicas[rivales[1]] = PoliticaSB3(
                random.choice(snapshots), rivales[1])
        else:
            politicas[rivales[1]] = PoliticaSB3(modelo_actual, rivales[1])
        politicas[rivales[2]] = BotExperto()  # ← ANCLA

    elif fase == 3:
        # ── DESTETE: 2 modelos + 1 snapshot (snapshots reemplazan BotExperto) ──
        politicas[rivales[0]] = PoliticaSB3(modelo_actual, rivales[0])
        politicas[rivales[1]] = PoliticaSB3(modelo_actual, rivales[1])
        if snapshots:
            politicas[rivales[2]] = PoliticaSB3(
                random.choice(snapshots), rivales[2])
        else:
            politicas[rivales[2]] = PoliticaSB3(modelo_actual, rivales[2])

    else:
        # ── Fase 4: self-play PURO — 3 modelos, sin BotExperto ──
        for r in rivales:
            politicas[r] = PoliticaSB3(modelo_actual, r)

    return CorazonesEnvV5(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
        oracle_buffer=oracle_buffer,
        oracle_rng=oracle_rng,
        epsilon=epsilon,
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
# Evaluación (contra BotExperto) — 3 niveles, 100 manos cada uno
# ------------------------------------------------------------------

def _evaluar_nivel(
    modelo,
    num_expertos: int,
    num_partidas: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evalúa contra N BotExperto + (3-N) bots heurísticos.

    Args:
        modelo: MaskablePPO entrenado.
        num_expertos: 1, 2, o 3 BotExpertos.
        num_partidas: Partidas a jugar (default 100).
        seed: Semilla base.

    Returns:
        Dict con: avg_score, top1_rate, top2_rate, posiciones, scores_por_jugador.
    """
    from src.v5.entorno import CorazonesEnvV5
    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import BOTS_DISPONIBLES

    posiciones = [0, 0, 0, 0]
    scores: Dict[str, List[float]] = {
        "modelo": [], "experto_1": [], "experto_2": [], "experto_3": [],
        "bot_1": [], "bot_2": [],
    }

    for i in range(num_partidas):
        agente_idx = i % 4
        rng = np.random.default_rng(seed + i)
        politicas: Dict[int, object] = {}
        oponentes = [(agente_idx + d) % 4 for d in (1, 2, 3)]

        # Asignar N BotExpertos + (3-N) bots
        for j, op in enumerate(oponentes):
            if j < num_expertos:
                politicas[op] = BotExperto()
            else:
                politicas[op] = random.choice(BOTS_DISPONIBLES)

        env = CorazonesEnvV5(agente_idx=agente_idx,
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
        scores["modelo"].append(puntos_agente)
        for j, op in enumerate(oponentes):
            key = f"experto_{j+1}" if j < num_expertos else f"bot_{j - num_expertos + 1}"
            scores[key].append(puntos_mano[op])

        ranking = sorted(enumerate(puntos_mano), key=lambda x: x[1])
        for pos, (jug, _) in enumerate(ranking):
            if jug == agente_idx:
                posiciones[pos] += 1
        env.close()

    avg_score = float(np.mean(scores["modelo"]))
    top1_rate = posiciones[0] / num_partidas
    top2_rate = (posiciones[0] + posiciones[1]) / num_partidas

    return {
        "avg_score": avg_score,
        "top1_rate": top1_rate,
        "top2_rate": top2_rate,
        "posiciones": [int(p) for p in posiciones],
        "scores": {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()},
    }


_EVAL_NIVELES = [
    (1, "E"),     # 1 Experto + 2 bots
    (2, "EE"),    # 2 Expertos + 1 bot
    (3, "EEE"),   # 3 Expertos
]


def _evaluar_completo(modelo, num_partidas: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Ejecuta los 3 niveles de evaluación y retorna resultados."""
    resultados = []
    for num_exp, tag in _EVAL_NIVELES:
        r = _evaluar_nivel(modelo, num_exp, num_partidas,
                           seed + num_exp * 1000)
        r["tag"] = tag
        resultados.append(r)
    return resultados


def _formatear_eval(resultados: List[Dict[str, Any]]) -> str:
    """Formatea los 3 resultados en una sola línea compacta."""
    parts = []
    for r in resultados:
        scores = r["scores"]
        # Promedio de oponentes
        opp_scores = [v for k, v in scores.items() if k != "modelo"]
        opp_avg = float(np.mean(opp_scores)) if opp_scores else 0.0
        parts.append(
            f"{r['tag']}: {r['avg_score']:.1f}pts "
            f"| T1={r['top1_rate']:.0%} T2={r['top2_rate']:.0%} "
            f"| Op={opp_avg:.1f}"
        )
    return " | ".join(parts)


# ------------------------------------------------------------------
# Entrenamiento principal — 4 entornos paralelos
# ------------------------------------------------------------------

def entrenar_auto(
    total_steps: int = 20_000_000,
    snapshot_every: int = 100_000,
    eval_every: int = 100_000,
    eval_partidas: int = 100,
    seed: int = 42,
    device: str = "cpu",
    resume_from: Optional[str] = None,
    output_dir: str = "models/v5",
    mcts_enabled: bool = False,
    ent_coef: float = ENT_COEF,
    epsilon_max: float = EPSILON_MAX,
    epsilon_min: float = EPSILON_MIN,
) -> int:
    """Entrenamiento autónomo v5 — 4 entornos paralelos shared-weights.

    Args:
        total_steps: Pasos totales (default 20M, 4× más datos que v4).
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

    # ── 4 entornos: uno por posición ──
    def _make_4_envs(modelo_actual=None, fase=0, epsilon=0.0):
        """Crea 4 entornos, uno por cada posición."""
        def _factory(pos):
            def _init():
                return crear_entorno_self_play_v5(
                    version=version_name,
                    agente_idx=pos,
                    modelo_actual=modelo_actual,
                    fase=fase,
                    epsilon=epsilon,
                )
            return _init
        return [_factory(0), _factory(1), _factory(2), _factory(3)]

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
        print(f"🔄 REANUDANDO v5 desde snapshot paso {step_counter:,}")
        print("=" * 60)
        modelo = MaskablePPO.load(ruta, device=device)

        progress = step_counter / total_steps
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * \
            math.exp(-EPSILON_DECAY * progress)
        if progress < 0.05:
            fase = 0
        elif progress < 0.15:
            fase = 1
        elif progress < 0.50:
            fase = 2
        else:
            fase = 3
        env_factories = _make_4_envs(
            modelo_actual=modelo if fase >= 2 else None, fase=fase, epsilon=epsilon)
        venv = DummyVecEnv(env_factories)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        modelo.set_env(venv)
        snapshot_count = step_counter // snapshot_every
    else:
        print("=" * 60)
        print(f"🚀 ENTRENANDO v5 desde cero ({total_steps:,} pasos)")
        print(f"   228 dims | MLP [512,256,128] | 4-env shared-weights")
        print(
            f"   LR: cosine {LR_MAX:.0e}→{LR_MIN:.0e} | ent_coef={ent_coef:.0e}")
        print(
            f"   Epsilon-greedy: {epsilon_max}→{epsilon_min} (decay={EPSILON_DECAY})")
        print(f"   Terminal-only reward (26 - puntos)")
        print(f"   4 entornos paralelos: 1 por posición")
        print(f"   Fase 0 ( 0- 5%): 3 Bots (warm-up mínimo)")
        print(f"   Fase 1 ( 5-15%): 1 BotExperto + 2 Bots (rival fuerte)")
        print(f"   Fase 2 (15-35%): 1 Modelo + 1 Snapshot + 1 BotExperto (ancla)")
        print(f"   Fase 3 (35-60%): 2 Modelos + 1 Snapshot (DESTETE — sin BotExperto)")
        print(f"   Fase 4 (60-100%): 3 Modelos (self-play PURO)")
        print(f"   Destete gradual: BotExperto → Snapshots → Self-play puro")
        print(f"   Device: {device} | Output: {output_dir}")
        print("=" * 60)

        policy_kwargs = obtener_policy_kwargs_v31(features_dim=128)

        epsilon = epsilon_max  # Máxima exploración al inicio
        env_factories = _make_4_envs(fase=0, epsilon=epsilon)
        venv = DummyVecEnv(env_factories)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)

        modelo = MaskablePPO(
            "MlpPolicy", venv, policy_kwargs=policy_kwargs,
            verbose=0, device=device, seed=seed,
            tensorboard_log=os.path.join(output_dir, "logs"),
            learning_rate=cosine_lr_schedule,
            n_steps=N_STEPS,   # 2048 total, 512 por posición
            batch_size=256, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=ent_coef,
            vf_coef=0.25, max_grad_norm=0.5,
        )

    # ── MCTS buffer (solo recolección, sin BC fine-tune) ──
    mcts_buffer = MCTSBuffer(capacity=100_000) if mcts_enabled else None

    # ── Training loop ──
    vecnorm_pkl = os.path.join(dir_vecnorm, "v5_vecnorm.pkl")
    pasos_restantes = total_steps - step_counter
    start_time = time.time()

    while step_counter < total_steps:
        progress = step_counter / total_steps
        if progress < 0.05:
            fase = 0
        elif progress < 0.15:
            fase = 1
        elif progress < 0.35:
            fase = 2
        elif progress < 0.60:
            fase = 3   # 35-60%: DESTETE — snapshots reemplazan BotExperto
        else:
            fase = 4   # 60-100%: self-play PURO

        # ── Calcular epsilon con decaimiento exponencial ──
        # epsilon nunca llega a 0: mínimo epsilon_min
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * \
            math.exp(-EPSILON_DECAY * progress)

        # ── Recrear los 4 entornos con la fase actual ──
        env_actual = modelo.get_env()
        if isinstance(env_actual, VecNormalize):
            env_factories = _make_4_envs(
                # fases 2, 3, 4 usan modelo_actual
                modelo_actual=modelo if fase >= 2 else None, fase=fase, epsilon=epsilon)
            env_actual.venv = DummyVecEnv(env_factories)
            _ = env_actual.reset()

        chunk = min(snapshot_every, pasos_restantes)
        pb_callback = _ProgressBarCallback(
            total=chunk, label=f"Snap {snapshot_count + 1:02d}", log_every=5000)
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
        print(f"📊 {bar_global} | fase={fase}")
        print(f"{'─'*60}")

        # Guardar
        env_actual = modelo.get_env()
        if isinstance(env_actual, VecNormalize):
            os.makedirs(os.path.dirname(vecnorm_pkl), exist_ok=True)
            env_actual.save(vecnorm_pkl)
        _guardar_snapshot(modelo, step_counter, dir_snapshots,
                          vecnorm_pkl if os.path.exists(vecnorm_pkl) else None)

        # Evaluar (3 niveles × 100 manos cada uno)
        if step_counter > 0 and step_counter % eval_every == 0:
            resultados = _evaluar_completo(
                modelo, eval_partidas, seed + step_counter)
            log_eval(eval_log_path, step_counter, resultados)
            linea = _formatear_eval(resultados)
            print(f"\n📈 {linea}")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Entrenamiento v5 COMPLETADO en {total_time/3600:.1f}h")
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
        description="Entrenamiento v5 — 4-env shared-weights")
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--snapshot-every", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-partidas", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/v5")
    parser.add_argument("--mcts", action="store_true", default=False)
    parser.add_argument("--ent-coef", type=float, default=ENT_COEF)
    parser.add_argument("--epsilon-max", type=float, default=EPSILON_MAX)
    parser.add_argument("--epsilon-min", type=float, default=EPSILON_MIN)

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
        ent_coef=args.ent_coef,
        epsilon_max=args.epsilon_max,
        epsilon_min=args.epsilon_min,
    )
