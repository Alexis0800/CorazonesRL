"""
Helpers de entrenamiento para el agente Corazones.

Provee constantes, factorías de entorno y utilidades de snapshots
para el pipeline de entrenamiento autónomo (train.py).
"""

from __future__ import annotations
from src.red import obtener_policy_kwargs
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
from src.agentes.bot_experto import BotExperto
from src.agentes.politica_rl import PoliticaSB3
from src.entorno.single_agent import CorazonesEnv

import os
import sys
import glob
import random
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# Configuración global
# ------------------------------------------------------------------
DIRECTORIO_LOGS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs")
DIRECTORIO_MODELOS_V6 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "modelos", "v6")
DIRECTORIO_VECNORM_V6 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "vecnormalize", "v6")

PROB_BOT_V2: float = 0.30
MIN_SNAPSHOT_STEPS: int = 500_000
MAX_SNAPSHOTS_POOL: int = 50


# ------------------------------------------------------------------
# Gestión de snapshots
# ------------------------------------------------------------------
def listar_snapshots(directorio: str) -> List[str]:
    """Lista snapshots del directorio, ordenados por paso de entrenamiento.

    Args:
        directorio: Directorio donde buscar snapshots.

    Returns:
        Lista de rutas absolutas sin extensión .zip.
    """
    os.makedirs(directorio, exist_ok=True)
    snaps = glob.glob(os.path.join(directorio, "snapshot_*.zip"))
    snaps.sort(key=lambda p: int(
        os.path.basename(p).replace("snapshot_", "").replace(".zip", "")))
    return [p.replace(".zip", "") for p in snaps]


def _extraer_paso_de_ruta(ruta_snapshot: str) -> int:
    """Extrae el número de paso de una ruta de snapshot.

    Args:
        ruta_snapshot: Ruta como '.../snapshot_0001500000' o con .zip.

    Returns:
        Número de paso (int), o 0 si no se puede extraer.
    """
    nombre = os.path.basename(ruta_snapshot).replace(".zip", "")
    try:
        return int(nombre.replace("snapshot_", ""))
    except ValueError:
        return 0


def _filtrar_snapshots_por_calidad(
    snapshots: List[str],
    min_steps: int = MIN_SNAPSHOT_STEPS,
) -> List[str]:
    """Filtra snapshots que cumplen el umbral mínimo de pasos.

    Previene el colapso de política excluyendo snapshots tempranos
    que representan políticas inmaduras o aleatorias.

    Args:
        snapshots: Lista de rutas de snapshots (sin .zip).
        min_steps: Pasos mínimos de entrenamiento requeridos.

    Returns:
        Lista filtrada de snapshots que cumplen el umbral.
    """
    if not snapshots:
        return []
    return [s for s in snapshots if _extraer_paso_de_ruta(s) >= min_steps]


def _aplicar_snapshot_pruning(
    directorio: str,
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
) -> int:
    """Elimina snapshots más antiguos si el pool excede el máximo.

    Mantiene los snapshots con mayor número de paso (más recientes).

    Args:
        directorio: Directorio donde se almacenan los snapshots .zip.
        max_snapshots: Número máximo de snapshots a conservar.

    Returns:
        Número de snapshots eliminados.
    """
    snaps = glob.glob(os.path.join(directorio, "snapshot_*.zip"))
    if len(snaps) <= max_snapshots:
        return 0

    snaps.sort(key=lambda p: _extraer_paso_de_ruta(p), reverse=True)
    eliminados = 0
    for snap in snaps[max_snapshots:]:
        try:
            os.remove(snap)
            vecnorm_file = snap.replace(".zip", "_vecnorm.pkl")
            if os.path.exists(vecnorm_file):
                os.remove(vecnorm_file)
            eliminados += 1
        except OSError:
            pass
    return eliminados


# ------------------------------------------------------------------
# Factoría de entorno self-play
# ------------------------------------------------------------------
def crear_entorno_self_play(
    directorio: str,
    agente_idx: int = 0,
    seed: Optional[int] = None,
    prob_bot: float = PROB_BOT_V2,
    min_snapshot_steps: int = MIN_SNAPSHOT_STEPS,
    prob_experto: float = 0.0,
    obs_dim: int = 194,
) -> CorazonesEnv:
    """Crea entorno Self-Play con snapshots del directorio especificado.

    Mezcla de oponentes por orden de preferencia:
      1. Con prob_bot: bot heurístico (conservador / agresivo / evasivo).
      2. Con prob_experto: BotExperto (sistema de reglas avanzado).
      3. El resto: snapshot histórico ponderado exponencialmente.
      Si no hay snapshots suficientes, usa bot heurístico como fallback.

    Args:
        directorio: Directorio de snapshots .zip.
        agente_idx: Índice del agente controlado por RL (0-3).
        seed: Semilla aleatoria opcional.
        prob_bot: Probabilidad de usar un bot heurístico como oponente.
        min_snapshot_steps: Pasos mínimos para considerar un snapshot.
        prob_experto: Probabilidad de usar BotExperto como oponente.
        obs_dim: Dimensión del vector de observación (194 o 220).

    Returns:
        Entorno CorazonesEnv configurado con oponentes mixtos.
    """
    todos_snapshots = listar_snapshots(directorio)
    snapshots = _filtrar_snapshots_por_calidad(
        todos_snapshots, min_snapshot_steps)

    bots_simples = [bot_evasivo, bot_evasivo, bot_conservador, bot_agresivo]
    random.shuffle(bots_simples)
    politicas: Dict[int, object] = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        r = random.random()
        usar_bot = r < prob_bot or len(snapshots) < 2
        usar_experto = not usar_bot and r < prob_bot + prob_experto

        if usar_experto:
            politicas[i] = BotExperto()
            continue

        if not usar_bot:
            pesos = np.exp(np.linspace(0, 2, len(snapshots)))
            pesos /= pesos.sum()
            snap_elegido = np.random.choice(snapshots, p=pesos)

            from sb3_contrib import MaskablePPO
            try:
                modelo_oponente = MaskablePPO.load(snap_elegido, device="cpu")
                vecnorm_path = snap_elegido + "_vecnorm.pkl"
                politicas[i] = PoliticaSB3(modelo_oponente, i, vecnorm_path)
                continue
            except Exception:
                pass

        politicas[i] = bots_simples[bot_idx % len(bots_simples)]
        bot_idx += 1

    env = CorazonesEnv(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
        obs_dim=obs_dim,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


# ------------------------------------------------------------------
# Hiperparámetros PPO
# ------------------------------------------------------------------
def obtener_hiperparametros_v2(logdir: str, device: str) -> Dict:
    """Hiperparámetros PPO optimizados para prevenir colapso de política.

    Cambios respecto a v1:
        - learning_rate: 3e-5 → 1e-4 (reactivar aprendizaje).
        - ent_coef: 0.05 → 0.10 (forzar más exploración, v11).
        - n_epochs: 10 → 8 (reducir sobreajuste por batch).
        - max_grad_norm: 0.5 → 1.0 (permitir gradientes más grandes).

    Args:
        logdir: Directorio para logs de TensorBoard.
        device: Dispositivo de cómputo ('cpu' o 'cuda').

    Returns:
        Diccionario con hiperparámetros para MaskablePPO.
    """
    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": 8,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.15,
        "normalize_advantage": True,
        "ent_coef": 0.10,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "target_kl": 0.02,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
    }


def obtener_hiperparametros_v4(
    logdir: str,
    device: str,
    paso_actual: int = 0,
    total_pasos: int = 20_000_000,
) -> Dict:
    """Hiperparámetros PPO corregidos para v19 — critic forte + self-play agresivo.

    Correcciones respecto a v3/v18 (análisis a 7.5M):
        - vf_coef: 1.0 → 2.0→1.5 (forzar critic, explained_variance estancado en ~0.60).
        - ent_coef: 0.12→0.06 → 0.16→0.08 (más exploración, policy_gradient_loss casi plano).
        - prob_bot_end: 0.30 → 0.15 (menos sobreajuste a bots simples en etapas tardías).
        - prob_experto default: 0.05 → 0.15 (más exposure al benchmark real).
        - max_grad_norm: 0.8 → 1.0 (permitir gradientes más grandes con critic fuerte).
        - n_steps: 4096 → 2048 (actualizaciones más frecuentes, batch_size ajustado a 256).

    Schedule lineal:
        Inicio (0%):  lr=3e-4, ent=0.16, vf_coef=2.0, clip=0.20, epochs=4
        Mitad (50%):  lr=2e-4, ent=0.12, vf_coef=1.75, clip=0.175, epochs=3
        Final (100%): lr=1e-4, ent=0.08, vf_coef=1.5, clip=0.15, epochs=3

    Args:
        logdir: Directorio para logs de TensorBoard.
        device: Dispositivo de cómputo ('cpu' o 'cuda').
        paso_actual: Paso global actual para calcular el progreso.
        total_pasos: Pasos totales planeados (default: 20M).

    Returns:
        Diccionario con hiperparámetros para MaskablePPO.
    """
    progreso = min(paso_actual / total_pasos, 1.0)

    # v19: lr (sin cambios)
    lr = 3e-4 + (1e-4 - 3e-4) * progreso
    # v19: ent_coef aumentado para combatir policy_gradient_loss plano
    ent = 0.16 + (0.08 - 0.16) * progreso
    # v19: clip (sin cambios)
    clip = 0.20 + (0.15 - 0.20) * progreso
    # v19: epochs (sin cambios)
    epochs = int(4 + (3 - 4) * progreso)
    # v19: vf_coef fuerte para forzar al critic (explained_variance estancado ~0.60)
    vf_coef = 2.0 + (1.5 - 2.0) * progreso
    # v19: grad_norm aumentado
    grad_norm = 1.0
    target_kl = 0.03 + (0.02 - 0.03) * progreso

    if progreso < 0.25:
        fase = "1 (exploración)"
    elif progreso < 0.75:
        fase = "2 (consolidación)"
    else:
        fase = "3 (fine-tuning)"

    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": lr,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": epochs,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": clip,
        "normalize_advantage": True,
        "ent_coef": ent,
        "vf_coef": vf_coef,
        "max_grad_norm": grad_norm,
        "target_kl": target_kl,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
        "_fase": fase,
    }


def obtener_hiperparametros_v5(
    logdir: str,
    device: str,
    paso_actual: int = 0,
    total_pasos: int = 20_000_000,
) -> Dict:
    """Hiperparámetros PPO corregidos para v19b — correcciones conservadoras sobre v19.

    v19 fracasó por 3 cambios simultáneos agresivos:
        - vf_coef=2.0 → critic inestable (explained_variance colapsos a ~0.16).
        - n_steps=2048 → solo 8 mini-batches, updates ruidosos.
        - ent_coef=0.16 inicial → política ruidosa early.

    Correcciones v19b:
        - vf_coef: 1.5→1.0 (moderado, estable, similar a v18 con leve boost inicial).
        - n_steps: 4096, batch_size: 512 (restaurar 8 epochs × 8 mini-batches → 64 updates).
        - ent_coef: 0.10→0.06 (similar a v18, era adecuado; 0.16 era excesivo).
        - prob_bot_end: 0.20 (compromiso entre 0.30 y 0.15).
        - prob_experto: 0.10 (más que v18, sin la sobreexposición de v19).

    Schedule lineal:
        Inicio (0%):  lr=3e-4, ent=0.10, vf_coef=1.5, clip=0.20, epochs=4
        Mitad (50%):  lr=2e-4, ent=0.08, vf_coef=1.25, clip=0.175, epochs=3
        Final (100%): lr=1e-4, ent=0.06, vf_coef=1.0, clip=0.15, epochs=3

    Args:
        logdir: Directorio para logs de TensorBoard.
        device: Dispositivo de cómputo ('cpu' o 'cuda').
        paso_actual: Paso global actual para calcular el progreso.
        total_pasos: Pasos totales planeados (default: 20M).

    Returns:
        Diccionario con hiperparámetros para MaskablePPO.
    """
    progreso = min(paso_actual / total_pasos, 1.0)

    lr = 3e-4 + (1e-4 - 3e-4) * progreso
    ent = 0.10 + (0.06 - 0.10) * progreso
    clip = 0.20 + (0.15 - 0.20) * progreso
    epochs = int(4 + (3 - 4) * progreso)
    vf_coef = 1.5 + (1.0 - 1.5) * progreso
    max_grad_norm = 0.8
    target_kl = 0.03 + (0.02 - 0.03) * progreso

    if progreso < 0.25:
        fase = "1 (exploración)"
    elif progreso < 0.75:
        fase = "2 (consolidación)"
    else:
        fase = "3 (fine-tuning)"

    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": lr,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": epochs,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": clip,
        "normalize_advantage": True,
        "ent_coef": ent,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "target_kl": target_kl,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
        "_fase": fase,
    }


def obtener_hiperparametros_v6(
    logdir: str,
    device: str,
    paso_actual: int = 0,
    total_pasos: int = 20_000_000,
) -> Dict:
    """Hiperparámetros PPO para v20 — ent_coef constante + LR con floor.

    v19b alcanzó 1636 Elo a 5.2M pero colapsó después por:
        - ent_coef decay 0.10→0.06 → entropía colapsó (0.70), KL→0.011
        - LR bajó demasiado rápido → actor dejó de aprender
        - La política se volvió determinística prematuramente.

    Correcciones v20:
        - ent_coef: 0.12 CONSTANTE (sin decay) — previene colapso de entropía.
        - LR en 2 fases: 3e-4→2e-4 (0-10M), 2e-4→1e-4 (10M-20M).
        - vf_coef: 1.5→1.0 (mismo decay estable de v19b, funcionó bien).
        - clip_range, epochs, target_kl: mismo schedule conservador de v19b.
        - n_steps=4096, batch_size=512 (sin cambios).

    Comparativa vs v19b:
        v19b: ent 0.10→0.06,  LR 3e-4→1e-4 (lineal)
        v20:  ent 0.12 fijo,   LR 3e-4→2e-4 (50%)→1e-4 (100%)

    Args:
        logdir: Directorio para logs de TensorBoard.
        device: Dispositivo de cómputo ('cpu' o 'cuda').
        paso_actual: Paso global actual para calcular el progreso.
        total_pasos: Pasos totales planeados (default: 20M).

    Returns:
        Diccionario con hiperparámetros para MaskablePPO.
    """
    progreso = min(paso_actual / total_pasos, 1.0)

    # LR en 2 fases: no baja de 2e-4 antes de 10M pasos
    if progreso < 0.5:
        lr = 3e-4 + (2e-4 - 3e-4) * (progreso / 0.5)
    else:
        lr = 2e-4 + (1e-4 - 2e-4) * ((progreso - 0.5) / 0.5)

    # ent_coef constante — la clave para evitar colapso de entropía
    ent = 0.12

    clip = 0.20 + (0.15 - 0.20) * progreso
    epochs = int(4 + (3 - 4) * progreso)
    vf_coef = 1.5 + (1.0 - 1.5) * progreso
    max_grad_norm = 0.8
    target_kl = 0.03 + (0.02 - 0.03) * progreso

    if progreso < 0.25:
        fase = "1 (exploración)"
    elif progreso < 0.75:
        fase = "2 (consolidación)"
    else:
        fase = "3 (fine-tuning)"

    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": lr,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": epochs,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": clip,
        "normalize_advantage": True,
        "ent_coef": ent,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "target_kl": target_kl,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
        "_fase": fase,
    }


def obtener_hiperparametros_v3(
    logdir: str,
    device: str,
    paso_actual: int = 0,
    total_pasos: int = 20_000_000,
) -> Dict:
    """Hiperparámetros PPO con learning rate schedule lineal.

    v18 — mismo ent_coef que v13 (exitosa a 1M) pero menos epochs:
        Inicio (0%):  lr=3e-4, ent=0.12, clip=0.20, epochs=4, batch=512, kl=0.03
        Mitad (50%):  lr=2e-4, ent=0.09, clip=0.175, epochs=3
        Final (100%): lr=1e-4, ent=0.06, clip=0.15, epochs=3, batch=512

    Calibración:
        - v13 con ent=0.12 tuvo WR=0.21 a 1M pero colapsó por epochs=8 (overfitting).
        - v18 reduce epochs 50% (8→4) para prevenir colapso con misma exploración.

    Args:
        logdir: Directorio para logs de TensorBoard.
        device: Dispositivo de cómputo ('cpu' o 'cuda').
        paso_actual: Paso global actual para calcular el progreso.
        total_pasos: Pasos totales planeados (default: 20M).

    Returns:
        Diccionario con hiperparámetros para MaskablePPO.
    """
    progreso = min(paso_actual / total_pasos, 1.0)

    # v18: lr suave
    lr = 3e-4 + (1e-4 - 3e-4) * progreso
    # v18: ent_coef = v13 (probado: WR=0.21 a 1M)
    ent = 0.12 + (0.06 - 0.12) * progreso
    # v18: clip
    clip = 0.20 + (0.15 - 0.20) * progreso
    # v18: epochs reducidas (menos overfitting que v13)
    epochs = int(4 + (3 - 4) * progreso)
    grad_norm = 0.8
    target_kl = 0.03 + (0.02 - 0.03) * progreso

    if progreso < 0.25:
        fase = "1 (exploración)"
    elif progreso < 0.75:
        fase = "2 (consolidación)"
    else:
        fase = "3 (fine-tuning)"

    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": lr,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": epochs,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": clip,
        "normalize_advantage": True,
        "ent_coef": ent,
        "vf_coef": 1.0,
        "max_grad_norm": grad_norm,
        "target_kl": target_kl,  # v18: decae con progreso
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
        "_fase": fase,
    }
