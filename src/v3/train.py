"""
Entrenamiento autónomo v3 — Transformer + MCTS + Fictitious Self-Play.

Pipeline de entrenamiento para el modelo v3 (250-dim, single-hand episodes):
  - TransformerFeatureExtractor (self-attention entre bloques de features)
  - MCTSBuffer con oráculo PIMC (Expert Iteration)
  - Fictitious Self-Play con bots heurísticos + BotExperto
  - Decaimiento progresivo de prob_bot (cosine decay)
  - Guardado de snapshots + VecNormalize
  - Evaluación periódica de win rate
  - Torneos Elo asíncronos

Diferencias clave vs v2 (train.py):
  - Observación de 250 dims (DIM_V3) en vez de 220
  - Episodios single-hand (no multi-mano)
  - Transformer en vez de MLP
  - MCTS oráculo integrado opcionalmente

Uso:
    python -m src.v3.train --total-steps 5000000 --output-dir models/v3
    python -m src.v3.train --resume models/v3/snapshots/snapshot_0010000000 --total-steps 15000000
"""

from __future__ import annotations
from src.v3.train_mcts import MCTSBuffer, evaluar_y_guardar_batch
from src.v3.red import obtener_policy_kwargs_transformer
from src.v3.entorno import CorazonesEnvV3
from src.entorno.dimensiones import DIM_V3
from src.agentes.heuristicos import BOTS_DISPONIBLES

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

# Asegurar que el directorio del proyecto está en el path
_proyecto = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _proyecto not in sys.path:
    sys.path.insert(0, _proyecto)


# ------------------------------------------------------------------
# Constantes v3
# ------------------------------------------------------------------
PROB_BOT_START: float = 0.50
PROB_BOT_END: float = 0.20
PROB_EXPERTO: float = 0.10
EVAL_PARTIDAS: int = 200
ELO_PARTIDAS: int = 30
ELO_MAX_SNAPSHOTS: int = 12
BEST_TOP: int = 2
MIN_SNAPSHOT_STEPS: int = 500_000
MAX_SNAPSHOTS_POOL: int = 50


# ------------------------------------------------------------------
# Utilidades: barra de progreso
# ------------------------------------------------------------------

def _progress_bar(
    current: int,
    total: int,
    width: int = 36,
    steps_per_sec: float = 0.0,
    prefix: str = "",
) -> str:
    """Renderiza una barra de progreso de texto.

    Args:
        current: Valor actual.
        total: Valor total.
        width: Ancho en caracteres de la barra.
        steps_per_sec: Pasos por segundo (0 = no mostrar ETA).
        prefix: Texto prefijo (ej: 'Chunk', 'Total').

    Returns:
        String con la barra de progreso.
    """
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
    """Callback que muestra barra de progreso durante learn().

    Reemplaza el verbose=1 de SB3 (tabla por iteración) con una barra
    compacta que se imprime cada N iteraciones.
    """

    def __init__(
        self,
        total: int,
        label: str = "Chunk",
        log_every: int = 5,
        verbose: int = 0,
    ):
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

        # Guardar últimas métricas
        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                self._last_metrics = {
                    k: v for k, v in self.model.logger.name_to_value.items()
                    if not k.startswith("time/")
                }
            except Exception:
                pass

        # Imprimir barra cada log_every iteraciones
        if self._iter_count % self._log_every == 0 or current >= self._total:
            bar = _progress_bar(
                min(current, self._total), self._total,
                steps_per_sec=steps_per_sec,
                prefix=f"  {self._label}",
            )
            # Mostrar métricas clave en la misma línea si hay
            metrics_str = self._format_metrics()
            print(f"\r{bar}{metrics_str}", end="", flush=True)

        return True

    def _on_training_end(self) -> None:
        # Barra final completa
        elapsed = time.time() - self._start_time
        current = self.model.num_timesteps - self._start_ts
        steps_per_sec = current / elapsed if elapsed > 0 else 0
        bar = _progress_bar(
            current, self._total,
            steps_per_sec=steps_per_sec,
            prefix=f"  {self._label}",
        )
        print(f"\r{bar}")

    def _format_metrics(self) -> str:
        """Formatea métricas clave en una línea compacta."""
        if not self._last_metrics:
            return ""
        key_metrics = [
            ("loss", "loss"),
            ("entropy_loss", "ent"),
            ("value_loss", "vloss"),
            ("approx_kl", "kl"),
            ("clip_fraction", "clip"),
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
    paso_actual: int,
    total_pasos: int,
    inicio: float = PROB_BOT_START,
    fin: float = PROB_BOT_END,
) -> float:
    """Calcula prob_bot con decaimiento coseno según el progreso.

    Fórmula: fin + 0.5 * (inicio - fin) * (1 + cos(pi * progreso))

    Args:
        paso_actual: Paso global actual.
        total_pasos: Pasos totales planeados.
        inicio: prob_bot inicial (default 0.50).
        fin: prob_bot final / piso (default 0.20).

    Returns:
        Probabilidad de usar un bot heurístico como oponente.
    """
    if total_pasos <= 0:
        return fin
    progreso = min(paso_actual / total_pasos, 1.0)
    return fin + 0.5 * (inicio - fin) * (1.0 + math.cos(math.pi * progreso))


# ------------------------------------------------------------------
# Logging JSONL
# ------------------------------------------------------------------

def log_eval(
    log_path: str,
    paso: int,
    wr: float,
    avg_score: float,
    num_partidas: int,
    prob_bot: float,
) -> None:
    """Registra resultado de evaluacion estandarizada en formato JSONL.

    Formato canonico de evaluacion:
        [Modelo, BotExperto, BotExperto, BotRotativo]

    Args:
        log_path: Ruta al archivo .jsonl.
        paso: Paso de entrenamiento.
        wr: Win rate estandar (fraccion manos con score <= 8).
        avg_score: Puntuacion promedio del agente.
        num_partidas: Numero de partidas por evaluacion.
        prob_bot: prob_bot usado en el entrenamiento.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "paso": paso,
        "wr": round(wr, 4),
        "avg_score": round(avg_score, 2),
        "num_partidas": num_partidas,
        "prob_bot": round(prob_bot, 4),
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Factory: self-play para v3
# ------------------------------------------------------------------

def crear_entorno_self_play_v3(
    version: str = "v3",
    prob_bot: float = 0.50,
    min_steps: int = MIN_SNAPSHOT_STEPS,
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
    agente_idx: int = 0,
    modelo_actual: Optional[Any] = None,
) -> CorazonesEnvV3:
    """Crea un entorno CorazonesEnvV3 con oponentes mixtos (self-play).

    Los oponentes son una mezcla de:
    - Bots heurísticos (prob_bot): conservador, agresivo, evasivo
    - BotExperto (10% fijo): para elevar el nivel de juego

    Args:
        version: Versión del modelo (ej: 'v3').
        prob_bot: Probabilidad de usar bot heurístico.
        min_steps: Pasos mínimos para un snapshot (no usado por ahora).
        max_snapshots: Máximo de snapshots en pool (no usado por ahora).
        agente_idx: Índice del agente en el entorno (0-3).
        modelo_actual: Instancia MaskablePPO actual (no usado por ahora).

    Returns:
        Instancia de CorazonesEnvV3 configurada para self-play.
    """
    from src.agentes.bot_experto import BotExperto

    # ── Seleccionar oponentes según prob_bot ──
    # prob_bot alta (inicio) → +bots fáciles
    # prob_bot baja (final) → +BotExperto (rivales fuertes)
    prob_experto_efectiva = max(0.05, min(0.80, 1.0 - prob_bot))

    politicas: Dict[int, object] = {}
    for offset in (1, 2, 3):
        rival_idx = (agente_idx + offset) % 4
        if random.random() < prob_experto_efectiva:
            politicas[rival_idx] = BotExperto()
        else:
            politicas[rival_idx] = random.choice(BOTS_DISPONIBLES)

    return CorazonesEnvV3(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
    )


# ------------------------------------------------------------------
# Guardado de snapshot + VecNormalize
# ------------------------------------------------------------------

def _guardar_snapshot(
    modelo: Any,
    paso: int,
    output_dir: str,
    vecnorm_path: Optional[str] = None,
) -> None:
    """Guarda snapshot .zip y opcionalmente VecNormalize .pkl.

    Args:
        modelo: Instancia de MaskablePPO.
        paso: Paso global al que corresponde el snapshot.
        output_dir: Directorio donde guardar.
        vecnorm_path: Ruta al VecNormalize a guardar (None = no guardar).
    """
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
# Evaluación simple (win rate vs bots + vs BotExperto)
# ------------------------------------------------------------------

def _evaluar_estandar(
    modelo: Any,
    num_partidas: int = EVAL_PARTIDAS,
    seed: int = 42,
) -> Tuple[float, float]:
    """Evalua win rate contra el campo estandar [Experto, Experto, BotRotativo].

    Formato canonico:
        Mesa: [Modelo, BotExperto, BotExperto, BotRotativo]
        BotRotativo alterna entre {conservador, agresivo, evasivo} cada mano.

    Args:
        modelo: Instancia MaskablePPO.
        num_partidas: Manos independientes a evaluar.
        seed: Semilla base.

    Returns:
        Tuple (wr, avg_score).
    """
    from src.v3.evaluacion import _construir_oponentes_estandar
    from src.v3.observacion import ObservacionBuilderV3, DIM_V3
    from src.dominio.carta import Carta
    import random as _random

    victorias = 0
    scores: list[float] = []
    builder = ObservacionBuilderV3(dim=DIM_V3)

    for i in range(num_partidas):
        agente_idx = i % 4  # rotar posicion
        rng = np.random.default_rng(seed + i)

        # Campo estandar: [Experto, Experto, BotRotativo]
        politicas = _construir_oponentes_estandar(agente_idx, seed + i)

        env = CorazonesEnvV3(
            agente_idx=agente_idx,
            politicas_oponentes=politicas,
        )
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        puntos_agente = info.get("puntos_agente", 26)
        scores.append(puntos_agente)
        if puntos_agente <= 8:
            victorias += 1

        env.close()

    wr = victorias / num_partidas
    avg_score = float(np.mean(scores))
    return wr, avg_score


# ------------------------------------------------------------------
# Entrenamiento autónomo principal
# ------------------------------------------------------------------

def entrenar_auto(
    total_steps: int = 5_000_000,
    snapshot_every: int = 100_000,
    eval_every: int = 50000,
    elo_every: int = 20,
    eval_partidas: int = EVAL_PARTIDAS,
    prob_bot_start: float = PROB_BOT_START,
    prob_bot_end: float = PROB_BOT_END,
    seed: int = 42,
    device: str = "cpu",
    resume_from: Optional[str] = None,
    output_dir: str = "models/v3",
    best_top: int = BEST_TOP,
    mcts_enabled: bool = False,
    mcts_frequency: float = 0.05,
) -> int:
    """Ejecuta entrenamiento autónomo v3 con Transformer + self-play.

    Args:
        total_steps: Pasos totales a entrenar.
        snapshot_every: Guardar snapshot cada N pasos.
        eval_every: Evaluar win rate cada N pasos (debe ser múltiplo de snapshot_every).
        elo_every: Lanzar torneo ELO cada N snapshots.
        eval_partidas: Partidas por evaluación.
        prob_bot_start: prob_bot inicial.
        prob_bot_end: prob_bot final.
        seed: Semilla aleatoria.
        device: Dispositivo de cómputo (cpu, cuda, dml).
        resume_from: Ruta a snapshot .zip para reanudar.
        output_dir: Directorio de salida.
        best_top: Cuántos mejores snapshots guardar.
        mcts_enabled: Si True, activa MCTS oracle guidance.
        mcts_frequency: Fracción de episodios con MCTS activo.

    Returns:
        Paso final alcanzado.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Directorios
    dir_snapshots = os.path.join(output_dir, "snapshots")
    dir_vecnorm = os.path.join(output_dir, "vecnorm")
    dir_best = os.path.join(output_dir, "best")
    os.makedirs(dir_snapshots, exist_ok=True)
    os.makedirs(dir_vecnorm, exist_ok=True)
    os.makedirs(dir_best, exist_ok=True)

    eval_log_path = os.path.join(output_dir, "eval_log.jsonl")

    step_counter = 0
    snapshot_count = 0

    # ── Modo resume ──
    if resume_from:
        if not resume_from.endswith(".zip"):
            resume_from += ".zip"
        ruta = resume_from if os.path.isabs(
            resume_from) else os.path.join(_proyecto, resume_from)
        if not os.path.exists(ruta):
            print(f"ERROR: No se encuentra {ruta}")
            sys.exit(1)

        step_counter = _extraer_paso_de_ruta(ruta)
        print("=" * 60)
        print(f"🔄 REANUDANDO v3 desde snapshot paso {step_counter:,}")
        print("=" * 60)

        modelo = MaskablePPO.load(ruta, device=device)
        vn_path = ruta.replace(".zip", "_vecnorm.pkl")

        def _make_env():
            return crear_entorno_self_play_v3(
                prob_bot=prob_bot_actual(
                    step_counter, total_steps, prob_bot_start, prob_bot_end),
                agente_idx=0,
            )

        if os.path.exists(vn_path):
            venv = VecNormalize.load(vn_path, DummyVecEnv([_make_env]))
        else:
            venv = DummyVecEnv([_make_env])
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                                clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        modelo.set_env(venv)
        snapshot_count = step_counter // snapshot_every

    else:
        print("=" * 60)
        print(f"🚀 ENTRENANDO v3 desde cero ({total_steps:,} pasos)")
        print(f"   Observación: {DIM_V3} dims | Transformer | Single-hand")
        print(f"   Device: {device}")
        print(f"   Output: {output_dir}")
        print("=" * 60)

        policy_kwargs = obtener_policy_kwargs_transformer(
            features_dim=256, d_model=128, n_heads=4, n_layers=2,
        )

        def _make_env():
            return crear_entorno_self_play_v3(prob_bot=prob_bot_start, agente_idx=0)

        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)

        modelo = MaskablePPO(
            "MlpPolicy",
            venv,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
            seed=seed,
            tensorboard_log=os.path.join(output_dir, "logs"),
            **HP_DEFAULT,
        )

    # ── MCTS buffer (si está habilitado) ──
    mcts_buffer = MCTSBuffer(capacity=100_000) if mcts_enabled else None

    # ── Training loop ──
    vecnorm_pkl = os.path.join(dir_vecnorm, "v3_vecnorm.pkl")
    pasos_restantes = total_steps - step_counter
    start_time = time.time()

    while step_counter < total_steps:
        # Actualizar prob_bot dinámicamente
        pb = prob_bot_actual(step_counter, total_steps,
                             prob_bot_start, prob_bot_end)

        # Recrear env interno con prob_bot actualizado, preservando VecNormalize
        # Reset explícito para sincronizar _last_obs con el nuevo entorno
        def _make_env_dyn():
            return crear_entorno_self_play_v3(prob_bot=pb, agente_idx=0)

        env_actual = modelo.get_env()
        if isinstance(env_actual, VecNormalize):
            env_actual.venv = DummyVecEnv([_make_env_dyn])
            # Sincronizar estado: forzar reset para alinear _last_obs
            _ = env_actual.reset()
        else:
            venv = DummyVecEnv([_make_env_dyn])
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                                clip_obs=10.0, clip_reward=10.0,
                                gamma=0.995, epsilon=1e-8)
            modelo.set_env(venv)

        # Entrenar un bloque con barra de progreso
        chunk = min(snapshot_every, pasos_restantes)
        pb_callback = _ProgressBarCallback(
            total=chunk, label=f"Snap {snapshot_count + 1:02d}", log_every=5,
        )
        modelo.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            log_interval=1,
            callback=pb_callback,
            progress_bar=False,
        )
        step_counter += chunk
        pasos_restantes -= chunk
        snapshot_count += 1

        elapsed = time.time() - start_time
        steps_per_sec = step_counter / elapsed if elapsed > 0 else 0

        # ── Barra de progreso GLOBAL ──
        bar_global = _progress_bar(
            step_counter, total_steps,
            steps_per_sec=steps_per_sec,
            prefix="  Total",
        )
        print(f"\n{'─'*60}")
        print(f"📊 {bar_global} | prob_bot={pb:.3f}")
        print(f"{'─'*60}")

        # ── Guardar VecNormalize (ANTES del snapshot para que se copie) ──
        env_actual = modelo.get_env()
        if isinstance(env_actual, VecNormalize):
            os.makedirs(os.path.dirname(vecnorm_pkl), exist_ok=True)
            env_actual.save(vecnorm_pkl)

        # ── Guardar snapshot ──
        _guardar_snapshot(modelo, step_counter, dir_snapshots,
                          vecnorm_pkl if os.path.exists(vecnorm_pkl) else None)

        # ── Evaluar win rate (formato estandar) ──
        if step_counter > 0 and step_counter % eval_every == 0:
            print(f"\n📈 Evaluando win rate ({eval_partidas} manos, "
                  f"formato [Modelo, Experto, Experto, Bot])...")
            wr, score = _evaluar_estandar(
                modelo, eval_partidas, seed + step_counter)
            log_eval(eval_log_path, step_counter,
                     wr, score, eval_partidas, pb)
            print(f"   WR estandar: {wr:.2%} | Score: {score:.1f}")

        # ── Torneo Elo (asíncrono) ──
        if snapshot_count % elo_every == 0 and snapshot_count > 0:
            print(f"\n🏆 Lanzando torneo Elo...")
            import subprocess
            elo_output = os.path.join(
                output_dir, "torneos", f"elo_paso_{step_counter:010d}.txt")
            os.makedirs(os.path.dirname(elo_output), exist_ok=True)

            try:
                subprocess.run([
                    sys.executable, "-m", "src.v3.elo",
                    "--directorio", dir_snapshots,
                    "--manos", str(ELO_PARTIDAS),
                    "--output", elo_output,
                ], cwd=_proyecto, timeout=600)
                print(f"   Resultados en: {elo_output}")
            except Exception as e:
                print(f"   ⚠️  Torneo Elo falló: {e}")

    # ── Fin ──
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Entrenamiento v3 COMPLETADO en {total_time/3600:.1f}h")
    print(f"   Último snapshot: paso {step_counter:,}")
    print(f"   Snapshots guardados: {dir_snapshots}")
    print(f"   Log evaluaciones: {eval_log_path}")
    print("=" * 60)

    modelo.save(os.path.join(dir_snapshots, f"snapshot_{step_counter:010d}"))
    return step_counter


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extraer_paso_de_ruta(ruta: str) -> int:
    """Extrae el número de paso de una ruta de snapshot."""
    import re
    match = re.search(r'snapshot_(\d+)', ruta)
    if match:
        return int(match.group(1))
    return 0


# ------------------------------------------------------------------
# Hiperparámetros default
# ------------------------------------------------------------------

HP_DEFAULT: Dict[str, Any] = {
    "learning_rate": 4e-5,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 1e-3,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrenamiento v3 — Transformer + Self-Play")
    parser.add_argument("--total-steps", type=int, default=5_000_000,
                        help="Pasos totales de entrenamiento (default: 5M)")
    parser.add_argument("--snapshot-every", type=int, default=100_000,
                        help="Guardar snapshot cada N pasos (default: 100K)")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluar cada N snapshots (default: 5)")
    parser.add_argument("--elo-every", type=int, default=20,
                        help="Torneo Elo cada N snapshots (default: 20)")
    parser.add_argument("--eval-partidas", type=int, default=EVAL_PARTIDAS,
                        help=f"Partidas por evaluación (default: {EVAL_PARTIDAS})")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "dml", "xpu"],
                        help="Dispositivo de cómputo (default: cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta a snapshot .zip para reanudar")
    parser.add_argument("--output-dir", type=str, default="models/v3",
                        help="Directorio de salida (default: models/v3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--mcts", action="store_true",
                        help="Activar MCTS oracle guidance")
    parser.add_argument("--mcts-frequency", type=float, default=0.05,
                        help="Fracción de episodios con MCTS (default: 0.05)")

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
    )
