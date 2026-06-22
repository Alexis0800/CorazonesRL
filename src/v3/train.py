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
from src.v3.train_mcts import MCTSBuffer, entrenar_bc_epoch
from src.v3.red import obtener_policy_kwargs_transformer
from src.v3.entorno import CorazonesEnvV3
from src.entorno.dimensiones import DIM_V3
from src.agentes.heuristicos import BOTS_DISPONIBLES

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

# Asegurar que el directorio del proyecto está en el path
_proyecto = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _proyecto not in sys.path:
    sys.path.insert(0, _proyecto)


# ------------------------------------------------------------------
# Constantes v3
# ------------------------------------------------------------------
PROB_BOT_START: float = 0.50
PROB_BOT_END: float = 0.05
PROB_EXPERTO: float = 0.10
EVAL_PARTIDAS: int = 200
ELO_PARTIDAS: int = 30
ELO_MAX_SNAPSHOTS: int = 12
BEST_TOP: int = 2
MIN_SNAPSHOT_STEPS: int = 100_000
MAX_SNAPSHOTS_POOL: int = 50
MIN_BC_SAMPLES: int = 100
MCTS_FREQUENCY: float = 0.40


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
    posiciones: Optional[List[int]] = None,
    scores_por_jugador: Optional[Dict[str, float]] = None,
    **kwargs,
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
        posiciones: Lista [1°, 2°, 3°, 4°] con conteo de posiciones del modelo.
        scores_por_jugador: Dict con avg score desglosado por tipo de jugador
            (modelo, experto_1, experto_2, bot).
    """
    entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "paso": paso,
        "wr": round(wr, 4),
        "avg_score": round(avg_score, 2),
        "num_partidas": num_partidas,
        "prob_bot": round(prob_bot, 4),
    }
    if posiciones is not None:
        entry["posiciones"] = [int(p) for p in posiciones]
        # Calcular top1 y top2 rates desde posiciones si no se pasaron
        total = sum(posiciones)
        if total > 0:
            if "top1_rate" not in kwargs:
                entry["top1_rate"] = round(posiciones[0] / total, 4)
            if "top2_rate" not in kwargs:
                entry["top2_rate"] = round(
                    (posiciones[0] + posiciones[1]) / total, 4)
    # Permitir override explícito de top1_rate y top2_rate + formato
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


def log_bc_loss(
    log_path: str,
    paso: int,
    bc_loss: float,
    buffer_size: int,
) -> None:
    """Registra pérdida de BC fine-tuning en el eval_log.

    Args:
        log_path: Ruta al archivo .jsonl de evaluación.
        paso: Paso de entrenamiento.
        bc_loss: Pérdida de cross-entropy del BC epoch.
        buffer_size: Tamaño del buffer MCTS usado.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tipo": "bc_finetune",
        "paso": paso,
        "bc_loss": round(bc_loss, 6),
        "buffer_size": buffer_size,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Utilidad: carga VecNormalize cross-dim (250 → 260)
# ------------------------------------------------------------------

def _pad_vecnorm_stats(stats: np.ndarray, target_dim: int, pad_value: float = 0.0) -> np.ndarray:
    """Asegura que las stats tengan target_dim elementos.

    Si son más cortas (modelo antiguo con menos dims),
    rellena con pad_value las nuevas dimensiones.
    """
    if stats.shape[0] >= target_dim:
        return stats[:target_dim].copy()
    padded = np.full(target_dim, pad_value, dtype=stats.dtype)
    padded[:stats.shape[0]] = stats
    return padded


def _cargar_vecnorm_compatible(
    pkl_path: str,
    make_env_fn,
    target_dim: int = DIM_V3,
):
    """Carga VecNormalize con compatibilidad cross-dim (250 → 260).

    Si el VecNormalize guardado tiene menos dimensiones que el entorno
    actual, rellena las nuevas dimensiones con mean=0, var=1.

    Args:
        pkl_path: Ruta al archivo .pkl de VecNormalize.
        make_env_fn: Función factory para crear el entorno (dim actual).
        target_dim: Dimensión objetivo de la observación.

    Returns:
        VecNormalize con stats compatibles.
    """
    import pickle
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Crear VecNormalize nuevo con el env actual (dim correcta)
    venv_new = DummyVecEnv([make_env_fn])
    vn = VecNormalize(
        venv_new, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
    )

    # Cargar stats guardadas del disco
    with open(pkl_path, "rb") as f:
        old_vn = pickle.load(f)

    # Transferir stats de recompensa (no cambian de dimensión)
    if hasattr(old_vn, "ret_rms"):
        vn.ret_rms = old_vn.ret_rms

    # Transferir stats de observación con padding
    old_mean = old_vn.obs_rms.mean
    old_var = old_vn.obs_rms.var
    vn.obs_rms.mean = _pad_vecnorm_stats(old_mean, target_dim, pad_value=0.0)
    vn.obs_rms.var = _pad_vecnorm_stats(old_var, target_dim, pad_value=1.0)
    # Preservar el contador para no reiniciar la normalización
    if hasattr(old_vn.obs_rms, "count"):
        vn.obs_rms.count = old_vn.obs_rms.count

    return vn


# ------------------------------------------------------------------
# Factory: self-play para v3 (con snapshots históricos)
# ------------------------------------------------------------------

def _cargar_snapshots_v3(
    snaps_dir: str,
    min_steps: int,
    max_snapshots: int,
) -> list:
    """Carga snapshots históricos para Fictitious Self-Play.

    Args:
        snaps_dir: Directorio de snapshots (ej: models/v3_mcts/snapshots).
        min_steps: Pasos mínimos para considerar un snapshot.
        max_snapshots: Máximo de snapshots en el pool.

    Returns:
        Lista de modelos MaskablePPO cargados en cpu.
    """
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

    # Ordenar por paso, tomar los max_snapshots más recientes
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


def crear_entorno_self_play_v3(
    version: str = "v3",
    prob_bot: float = 0.50,
    min_steps: int = MIN_SNAPSHOT_STEPS,
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
    agente_idx: int = 0,
    modelo_actual: Optional[Any] = None,
    oracle_buffer: Optional[object] = None,
    oracle_rng: Optional[np.random.Generator] = None,
) -> CorazonesEnvV3:
    """Crea un entorno CorazonesEnvV3 con oponentes mixtos (self-play).

    Los oponentes son una mezcla de:
    - Bots heurísticos (prob_bot): conservador, agresivo, evasivo
    - Snapshots históricos (1 - prob_bot): Fictitious Self-Play real
    - BotExperto (fallback): cuando no hay snapshots disponibles aún

    Args:
        version: Versión del modelo (ej: 'v3_mcts').
        prob_bot: Probabilidad de usar bot heurístico.
        min_steps: Pasos mínimos para un snapshot.
        max_snapshots: Máximo de snapshots en pool.
        agente_idx: Índice del agente en el entorno (0-3).
        modelo_actual: Instancia MaskablePPO actual (no usado).
        oracle_buffer: Buffer MCTS para recolectar datos del oráculo.
        oracle_rng: Generador aleatorio para el oráculo.

    Returns:
        Instancia de CorazonesEnvV3 configurada para self-play.
    """
    from src.agentes.bot_experto import BotExperto
    from src.agentes.politica_rl import PoliticaSB3

    # ── Cargar pool de snapshots históricos ──
    snaps_dir = os.path.join("models", version, "snapshots")
    snapshots = _cargar_snapshots_v3(snaps_dir, min_steps, max_snapshots)

    # ── Asignar oponentes ──
    politicas: Dict[int, object] = {}
    for offset in (1, 2, 3):
        rival_idx = (agente_idx + offset) % 4
        if random.random() < prob_bot:
            # Bot heurístico aleatorio
            politicas[rival_idx] = random.choice(BOTS_DISPONIBLES)
        elif snapshots:
            # Snapshot histórico (Fictitious Self-Play real)
            snap = random.choice(snapshots)
            politicas[rival_idx] = PoliticaSB3(snap, rival_idx)
        else:
            # Fallback: BotExperto (sin snapshots aún)
            politicas[rival_idx] = BotExperto()

    return CorazonesEnvV3(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
        oracle_buffer=oracle_buffer,
        oracle_rng=oracle_rng,
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
) -> Tuple[float, float, List[int], Dict[str, float], float, float]:
    """Evalua win rate contra el campo estandar [Experto, Experto, BotRotativo].

    Formato canonico:
        Mesa: [Modelo, BotExperto, BotExperto, BotRotativo]
        BotRotativo alterna entre {conservador, agresivo, evasivo} cada mano.

    Args:
        modelo: Instancia MaskablePPO.
        num_partidas: Manos independientes a evaluar.
        seed: Semilla base.

    Returns:
        Tuple (wr, avg_score, posiciones, scores_por_jugador, top1_rate, top2_rate).
        - wr: Win rate estandar (score modelo <= 8).
        - avg_score: Puntuacion promedio del modelo.
        - posiciones: [1°, 2°, 3°, 4°].
        - scores_por_jugador: Avg score por tipo.
        - top1_rate: Fraccion de 1° puesto.
        - top2_rate: Fraccion de 1° o 2° puesto.
    """
    from src.v3.evaluacion import _construir_oponentes_estandar
    from src.v3.observacion import ObservacionBuilderV3, DIM_V3
    from src.dominio.carta import Carta
    import random as _random

    victorias = 0
    scores: list[float] = []
    posiciones = [0, 0, 0, 0]  # 1st, 2nd, 3rd, 4th
    scores_por_tipo: Dict[str, list] = {
        "modelo": [], "experto_1": [], "experto_2": [], "bot": [],
    }
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

        # Puntuaciones de los 4 jugadores
        puntos_mano = env.motor.calcular_puntuacion_mano()
        puntos_agente = puntos_mano[agente_idx]
        scores.append(puntos_agente)
        if puntos_agente <= 8:
            victorias += 1

        # Determinar posicion (1° = menor puntuacion)
        ranking = sorted(enumerate(puntos_mano), key=lambda x: x[1])
        for pos, (jug, _) in enumerate(ranking):
            if jug == agente_idx:
                posiciones[pos] += 1

        # Acumular scores por tipo de jugador
        # Mapear indices a tipos: agente=modelo, oponentes=experto_1, experto_2, bot
        scores_por_tipo["modelo"].append(puntos_agente)
        # Los oponentes estan en politicas[offset]; el orden es experto_1, experto_2, bot
        oponentes_ordenados = sorted(politicas.keys())
        for j, op_idx in enumerate(oponentes_ordenados):
            if j == 0:
                scores_por_tipo["experto_1"].append(puntos_mano[op_idx])
            elif j == 1:
                scores_por_tipo["experto_2"].append(puntos_mano[op_idx])
            else:
                scores_por_tipo["bot"].append(puntos_mano[op_idx])

        env.close()

    wr = victorias / num_partidas
    avg_score = float(np.mean(scores))
    top1_rate = posiciones[0] / num_partidas
    top2_rate = (posiciones[0] + posiciones[1]) / num_partidas
    avg_por_tipo = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in scores_por_tipo.items()
    }
    return wr, avg_score, posiciones, avg_por_tipo, top1_rate, top2_rate


def _evaluar_facil(
    modelo: Any,
    num_partidas: int = 50,
    seed: int = 42,
) -> Tuple[float, float, List[int], Dict[str, float], float, float]:
    """Evalua contra campo facil [Experto, Bot, Bot].

    Formato:
        Mesa: [Modelo, BotExperto, BotHeuristico, BotHeuristico]
        Solo 1 BotExperto (rival fuerte). Los otros 2 son bots heuristicos.

    Util para medir progreso cuando el formato estandar (2 Expertos)
    es demasiado dificil y el modelo aun no llega a ese nivel.

    Returns:
        Igual que _evaluar_estandar: (wr, avg_score, pos, scores_pj, top1, top2).
    """
    from src.v3.evaluacion import _construir_oponentes_estandar
    from src.v3.observacion import ObservacionBuilderV3, DIM_V3
    from src.agentes.heuristicos import BOTS_DISPONIBLES
    from src.agentes.bot_experto import BotExperto
    import random as _random

    victorias = 0
    scores: list[float] = []
    posiciones = [0, 0, 0, 0]
    scores_por_tipo: Dict[str, list] = {
        "modelo": [], "experto": [], "bot_1": [], "bot_2": [],
    }
    builder = ObservacionBuilderV3(dim=DIM_V3)

    for i in range(num_partidas):
        agente_idx = i % 4
        rng = np.random.default_rng(seed + i)

        # Campo facil: [Experto, Bot, Bot]
        politicas: Dict[int, object] = {}
        oponentes = [(agente_idx + d) % 4 for d in (1, 2, 3)]
        politicas[oponentes[0]] = BotExperto()
        politicas[oponentes[1]] = _random.choice(BOTS_DISPONIBLES)
        politicas[oponentes[2]] = _random.choice(BOTS_DISPONIBLES)

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
    avg_por_tipo = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in scores_por_tipo.items()
    }
    return wr, avg_score, posiciones, avg_por_tipo, top1_rate, top2_rate


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
    # "v3_mcts" from "models/v3_mcts"
    version_name = os.path.basename(output_dir)

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
                version=version_name,
                prob_bot=prob_bot_actual(
                    step_counter, total_steps, prob_bot_start, prob_bot_end),
                agente_idx=0,
            )

        if os.path.exists(vn_path):
            venv = _cargar_vecnorm_compatible(vn_path, _make_env)
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
            return crear_entorno_self_play_v3(
                version=version_name,
                prob_bot=prob_bot_start, agente_idx=0)

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

        # ── Decidir si este chunk recolecta datos del oráculo ──
        enviar_oraculo = (
            mcts_enabled
            and mcts_buffer is not None
            and random.random() < mcts_frequency
        )

        # Recrear env interno con prob_bot actualizado, preservando VecNormalize
        # Reset explícito para sincronizar _last_obs con el nuevo entorno
        def _make_env_dyn():
            return crear_entorno_self_play_v3(
                version=version_name,
                prob_bot=pb, agente_idx=0,
                oracle_buffer=mcts_buffer if enviar_oraculo else None,
                oracle_rng=(
                    np.random.default_rng(seed + step_counter)
                    if enviar_oraculo else None
                ),
            )

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

        # ── BC fine-tuning con datos del oráculo ──
        if (mcts_enabled and mcts_buffer is not None
                and len(mcts_buffer) >= MIN_BC_SAMPLES):
            env_actual = modelo.get_env()
            vn = env_actual if isinstance(env_actual, VecNormalize) else None
            try:
                bc_loss = entrenar_bc_epoch(
                    modelo, mcts_buffer,
                    batch_size=256, lr=1e-3, vecnorm=vn,
                )
                buf_size = len(mcts_buffer)
                log_bc_loss(eval_log_path, step_counter, bc_loss, buf_size)
                if bc_loss > 0:
                    print(f"   🧠 BC fine-tune: loss={bc_loss:.4f} "
                          f"| buffer={buf_size}")
            except Exception as e:
                print(f"   ⚠️  BC fine-tune falló: {e}")

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

        # ── Evaluar win rate (formato estandar + facil) ──
        if step_counter > 0 and step_counter % eval_every == 0:
            print(f"\n📈 Evaluando ({eval_partidas} manos)...")

            # Formato dificil: [Modelo, Experto, Experto, Bot]
            wr, score, pos, spj, t1, t2 = _evaluar_estandar(
                modelo, eval_partidas, seed + step_counter)
            log_eval(eval_log_path, step_counter,
                     wr, score, eval_partidas, pb,
                     posiciones=pos, scores_por_jugador=spj,
                     top1_rate=t1, top2_rate=t2, formato="dificil")
            pos_str = " > ".join(f"{p}×{c}" for p, c in zip(
                ["1°", "2°", "3°", "4°"], pos))
            print(
                f"   DIFICIL [M,E,E,B]: WR≤8={wr:.0%} Top1={t1:.0%} Top2={t2:.0%} Score={score:.1f} | {pos_str}")

            # Formato facil: [Modelo, Experto, Bot, Bot]
            wr2, score2, pos2, spj2, t1_2, t2_2 = _evaluar_facil(
                modelo, eval_partidas // 2, seed + step_counter + 1)
            log_eval(eval_log_path, step_counter,
                     wr2, score2, eval_partidas // 2, pb,
                     posiciones=pos2, scores_por_jugador=spj2,
                     top1_rate=t1_2, top2_rate=t2_2, formato="facil")
            pos_str2 = " > ".join(f"{p}×{c}" for p, c in zip(
                ["1°", "2°", "3°", "4°"], pos2))
            print(
                f"   FACIL   [M,E,b,b]: WR≤8={wr2:.0%} Top1={t1_2:.0%} Top2={t2_2:.0%} Score={score2:.1f} | {pos_str2}")

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
    "learning_rate": 1e-4,
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
    parser.add_argument("--mcts-frequency", type=float, default=MCTS_FREQUENCY,
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
