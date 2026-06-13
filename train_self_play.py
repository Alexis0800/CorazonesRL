"""
Pipeline de Entrenamiento con Self-Play real (v4).

CORRECCIÓN: El modelo se estancó en ~47% porque entrenaba contra 3 bots fijos.
Self-Play real entrena contra snapshots históricos del propio agente, creando
un currículum de dificultad creciente.

Fase 1 (hecha): 3.6M pasos contra bots → ~47% win rate
Fase 2 (AHORA): Self-Play contra los 74 snapshots existentes

VecNormalize: Cada snapshot guarda sus propias stats de normalización.
Al cargar un snapshot con MaskablePPO.load(), SB3 restaura sus stats.
Al llamar model.predict(obs), las observaciones se normalizan automáticamente.

Uso:
    python train_self_play.py --self-play --steps 5000000 --snapshot-every 50000
"""

from __future__ import annotations
from src.red import obtener_policy_kwargs
from src.carta import Carta
from src.bots import bot_conservador, bot_agresivo, bot_evasivo
from src.entorno import CorazonesEnv

import os
import sys
import glob
import random
import argparse
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# Configuración global
# ------------------------------------------------------------------
DIRECTORIO_MODELOS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "modelos_historicos")
DIRECTORIO_MODELOS_V2 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "modelos_historicos", "v2")
DIRECTORIO_MODELOS_V5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "modelos_historicos", "v5")
DIRECTORIO_LOGS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs")
DIRECTORIO_VECNORM = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "vecnormalize")
DIRECTORIO_VECNORM_V5 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "vecnormalize", "v5")
DIRECTORIO_MODELOS_V6 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "modelos_historicos", "v6")
DIRECTORIO_VECNORM_V6 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "vecnormalize", "v6")

# ------------------------------------------------------------------
# Constantes Fase 5B (Self-Play agresivo, mínimo anclaje a bots)
# ------------------------------------------------------------------
# 30% bots (50% evasivo para cerrar punto ciego)
PROB_BOT_V2: float = 0.30
MIN_SNAPSHOT_STEPS: int = 500_000  # Solo snapshots maduros (≥500K pasos)
# Máximo de snapshots en el pool activo (subido a 50 para evitar pruning prematuro)
MAX_SNAPSHOTS_POOL: int = 50


# ------------------------------------------------------------------
# Política SB3 con normalización de observaciones
# ------------------------------------------------------------------
class PoliticaSB3:
    """Adaptador que envuelve un snapshot como política de oponente.

    El modelo se cargó con MaskablePPO.load() que restaura sus stats de
    VecNormalize. Al llamar predict(), las observaciones se normalizan
    automáticamente usando esas stats.
    """

    def __init__(self, model: Any, agente_idx: int, vecnorm_path: Optional[str] = None):
        self.model = model
        self.agente_idx = agente_idx
        self._obs_rms = None

        # Cargar stats de VecNormalize asociadas al snapshot
        if vecnorm_path and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    data = pickle.load(f)
                self._obs_rms = data.get("obs_rms", None)
            except Exception:
                self._obs_rms = None

    def __call__(self, motor: Any, jugador_idx: int, legales: List[Carta]) -> Carta:
        # Construir observación desde perspectiva de este oponente
        env = motor  # El motor expone suficiente info
        obs = self._construir_obs_desde_motor(motor, jugador_idx)

        # Normalizar si tenemos stats
        if self._obs_rms is not None:
            mean = np.array(self._obs_rms.mean)
            var = np.array(self._obs_rms.var)
            obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -
                          10.0, 10.0).astype(np.float32)

        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True
        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True)
        return Carta._TODAS[int(action)]

    def _construir_obs_desde_motor(self, motor: Any, jugador_idx: int) -> np.ndarray:
        """Construye observación de 194 dims desde la perspectiva de jugador_idx."""
        obs = np.zeros(194, dtype=np.float32)
        a = jugador_idx

        # Mano del jugador
        for c in motor.jugadores[a].mano:
            obs[c.id] = 1.0

        # Mesa actual
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0

        # Bazas ganadas (cementerio)
        for i in range(4):
            for c in motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        # Vacíos (no disponibles desde el motor) — quedan en 0
        # Puntajes históricos (no disponibles desde el motor) — quedan en 0
        # Features estratégicas v5 (no disponibles desde el motor) — quedan en 0
        # Features all_void v6 (no disponibles desde el motor) — quedan en 0
        return obs


# ------------------------------------------------------------------
# Transferencia de pesos entre versiones de espacio de observación
# ------------------------------------------------------------------

def transferir_pesos(
    ruta_origen: str,
    ruta_destino: Optional[str] = None,
    old_dim: int = 190,
    new_dim: int = 194,
    device: str = "cpu",
    directorio_destino: Optional[str] = None,
) -> Any:
    """Transfiere pesos de un modelo con input_dim=old_dim a uno con input_dim=new_dim.

    Las nuevas columnas de la primera capa se rellenan con ceros, lo que
    significa que las nuevas features empiezan "apagadas" y PPO aprende
    progresivamente a usarlas. El bias de la primera capa y todos los
    pesos de capas posteriores se copian tal cual.

    Args:
        ruta_origen: Ruta al snapshot .zip del modelo fuente.
        ruta_destino: Ruta donde guardar el modelo transferido (opcional).
        old_dim: Dimensión de entrada del modelo fuente (default 190).
        new_dim: Dimensión de entrada del modelo destino (default 194).
        device: Dispositivo de cómputo ('cpu' o 'cuda').
        directorio_destino: Directorio donde guardar (alternativa a ruta_destino).

    Returns:
        Modelo MaskablePPO con pesos transferidos.
    """
    import torch
    from sb3_contrib import MaskablePPO

    # 1. Cargar modelo fuente
    print(
        f"📦 Cargando modelo fuente ({old_dim}→{new_dim} dims): {ruta_origen}")
    modelo_origen = MaskablePPO.load(ruta_origen, device=device)

    # 2. Extraer el paso de entrenamiento del nombre
    paso = _extraer_paso_de_ruta(ruta_origen)

    # 3. Crear modelo destino con el espacio correcto (194 dims)
    #    Usamos un entorno temporal 194-dim para que SB3 cree la red correcta
    env_temp = CorazonesEnv(agente_idx=0)
    env_temp.reset(seed=42)

    # Crear VecNormalize temporal para el constructor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    venv_temp = DummyVecEnv([lambda: env_temp])
    venv_temp = VecNormalize(venv_temp, norm_obs=True, norm_reward=True)

    hp = obtener_hiperparametros_v3(
        directorio_destino or ".", device, paso)
    policy_kwargs = hp.pop("policy_kwargs", None) or obtener_policy_kwargs()
    hp.pop("_fase", None)
    hp.pop("tensorboard_log", None)

    modelo_destino = MaskablePPO(
        policy=hp["policy"],
        env=venv_temp,
        **{k: v for k, v in hp.items()
           if k in ["learning_rate", "n_steps", "batch_size",
                    "n_epochs", "gamma", "gae_lambda",
                    "clip_range", "normalize_advantage",
                    "ent_coef", "vf_coef", "max_grad_norm",
                    "target_kl", "verbose", "device"]},
        policy_kwargs=policy_kwargs,
    )
    venv_temp.close()

    # 4. Transferir pesos
    state_origen = modelo_origen.policy.state_dict()
    state_destino = modelo_destino.policy.state_dict()

    for key in state_destino:
        if key in state_origen:
            if "weight" in key and state_origen[key].shape != state_destino[key].shape:
                # Expandir la primera capa: (H, old_dim) → (H, new_dim)
                old_w = state_origen[key]  # (256, 190)
                new_w = torch.zeros_like(state_destino[key])  # (256, 194)
                new_w[:, :old_dim] = old_w   # copiar columnas existentes
                new_w[:, old_dim:] = 0.0     # nuevas columnas en cero
                state_destino[key] = new_w
                print(
                    f"  [Transfer] {key}: {list(old_w.shape)} → {list(new_w.shape)} (padding con ceros)")
            else:
                state_destino[key] = state_origen[key]

    modelo_destino.policy.load_state_dict(state_destino, strict=False)
    modelo_destino._total_timesteps = int(paso)
    print(f"  [OK] Pesos transferidos. Paso inicial: {paso:,}")

    # 5. Guardar modelo transferido
    if ruta_destino:
        os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
        modelo_destino.save(ruta_destino)
        print(f"  [OK] Modelo transferido guardado en: {ruta_destino}.zip")

    return modelo_destino


# ------------------------------------------------------------------
# Factoría de entornos
# ------------------------------------------------------------------
def crear_entorno_con_bots(agente_idx=0, seed=None, shuffle_bots=True):
    bots = [bot_evasivo, bot_evasivo, bot_conservador, bot_agresivo]
    if shuffle_bots:
        random.shuffle(bots)
    politicas = {}
    bot_idx = 0
    for i in range(4):
        if i != agente_idx:
            politicas[i] = bots[bot_idx % len(bots)]
            bot_idx += 1
    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


def listar_snapshots():
    if not os.path.isdir(DIRECTORIO_MODELOS):
        return []
    snaps = glob.glob(os.path.join(DIRECTORIO_MODELOS, "snapshot_*.zip"))
    snaps.sort(key=lambda p: int(os.path.basename(
        p).replace("snapshot_", "").replace(".zip", "")))
    return [p.replace(".zip", "") for p in snaps]


def listar_snapshots_v2() -> List[str]:
    """Lista snapshots del directorio v2, ordenados por paso de entrenamiento.

    Returns:
        Lista de rutas absolutas sin extensión .zip.
    """
    os.makedirs(DIRECTORIO_MODELOS_V2, exist_ok=True)
    snaps = glob.glob(os.path.join(DIRECTORIO_MODELOS_V2, "snapshot_*.zip"))
    snaps.sort(key=lambda p: int(os.path.basename(
        p).replace("snapshot_", "").replace(".zip", "")))
    return [p.replace(".zip", "") for p in snaps]


def listar_snapshots_v5() -> List[str]:
    """Lista snapshots del directorio v5 (190 dims), ordenados por paso.

    Returns:
        Lista de rutas absolutas sin extensión .zip.
    """
    os.makedirs(DIRECTORIO_MODELOS_V5, exist_ok=True)
    snaps = glob.glob(os.path.join(DIRECTORIO_MODELOS_V5, "snapshot_*.zip"))
    snaps.sort(key=lambda p: int(os.path.basename(
        p).replace("snapshot_", "").replace(".zip", "")))
    return [p.replace(".zip", "") for p in snaps]


def listar_snapshots_v6() -> List[str]:
    """Lista snapshots del directorio v6 (194 dims), ordenados por paso."""
    os.makedirs(DIRECTORIO_MODELOS_V6, exist_ok=True)
    snaps = glob.glob(os.path.join(DIRECTORIO_MODELOS_V6, "snapshot_*.zip"))
    snaps.sort(key=lambda p: int(os.path.basename(
        p).replace("snapshot_", "").replace(".zip", "")))
    return [p.replace(".zip", "") for p in snaps]


def _extraer_paso_de_ruta(ruta_snapshot: str) -> int:
    """Extrae el número de paso de una ruta de snapshot.

    Args:
        ruta_snapshot: Ruta como '.../snapshot_0001500000' o '.../snapshot_0001500000.zip'.

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
    """Filtra snapshots que cumplen el umbral mínimo de pasos de entrenamiento.

    Previene el colapso de política excluyendo snapshots demasiado tempranos
    que representan políticas inmaduras o aleatorias.

    Args:
        snapshots: Lista de rutas de snapshots (sin .zip).
        min_steps: Pasos mínimos de entrenamiento requeridos.

    Returns:
        Lista filtrada de snapshots que cumplen el umbral.
    """
    if not snapshots:
        return []
    return [
        s for s in snapshots
        if _extraer_paso_de_ruta(s) >= min_steps
    ]


def crear_entorno_self_play(agente_idx=0, seed=None, prob_bot=0.15):
    """Crea entorno Self-Play: 85% snapshots, 15% bots."""
    snapshots = listar_snapshots()
    bots = [bot_evasivo, bot_evasivo, bot_conservador, bot_agresivo]
    random.shuffle(bots)
    politicas = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        usar_bot = random.random() < prob_bot or len(snapshots) < 2

        if not usar_bot:
            # Elegir snapshot aleatorio con peso hacia recientes
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
                pass  # Fallback a bot

        politicas[i] = bots[bot_idx % len(bots)]
        bot_idx += 1

    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


def crear_entorno_self_play_v2(
    agente_idx: int = 0,
    seed: Optional[int] = None,
    prob_bot: float = PROB_BOT_V2,
    min_snapshot_steps: int = MIN_SNAPSHOT_STEPS,
) -> CorazonesEnv:
    """Crea entorno Self-Play v2 con quality-filter y mayor anclaje a bots.

    Cambios respecto a v1:
        - Solo usa snapshots con >= min_snapshot_steps pasos (quality filter).
        - 40% de oponentes son bots heurísticos (vs 15% en v1).
        - Los snapshots se eligen con pesos exponenciales hacia los más
          recientes (políticas más maduras).
        - Si no hay snapshots de calidad, todos los oponentes son bots.

    Args:
        agente_idx: Índice del agente controlado por RL (0-3).
        seed: Semilla aleatoria opcional.
        prob_bot: Probabilidad de usar un bot heurístico como oponente.
        min_snapshot_steps: Pasos mínimos para considerar un snapshot.

    Returns:
        Entorno CorazonesEnv configurado con oponentes mixtos.
    """
    todos_snapshots = listar_snapshots_v2()
    snapshots = _filtrar_snapshots_por_calidad(
        todos_snapshots, min_snapshot_steps)

    bots = [bot_evasivo, bot_evasivo, bot_conservador, bot_agresivo]
    random.shuffle(bots)
    politicas: Dict[int, object] = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        # Usar bot si: probabilidad dicta O no hay suficientes snapshots de calidad
        usar_bot = random.random() < prob_bot or len(snapshots) < 2

        if not usar_bot:
            # Pesos exponenciales: favorecer snapshots más recientes (mayor paso)
            pasos = np.array([_extraer_paso_de_ruta(s) for s in snapshots])
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
                pass  # Fallback a bot si falla la carga

        politicas[i] = bots[bot_idx % len(bots)]
        bot_idx += 1

    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


def crear_entorno_self_play_v5(
    agente_idx: int = 0,
    seed: Optional[int] = None,
    prob_bot: float = PROB_BOT_V2,
    min_snapshot_steps: int = MIN_SNAPSHOT_STEPS,
) -> CorazonesEnv:
    """Crea entorno Self-Play v5 (190 dims) con snapshots del directorio v5.

    Idéntico a v2 pero usa listar_snapshots_v5() y el directorio v5.
    """
    todos_snapshots = listar_snapshots_v5()
    snapshots = _filtrar_snapshots_por_calidad(
        todos_snapshots, min_snapshot_steps)

    bots = [bot_evasivo, bot_evasivo, bot_conservador, bot_agresivo]
    random.shuffle(bots)
    politicas: Dict[int, object] = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        usar_bot = random.random() < prob_bot or len(snapshots) < 2

        if not usar_bot:
            pasos = np.array([_extraer_paso_de_ruta(s) for s in snapshots])
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

        politicas[i] = bots[bot_idx % len(bots)]
        bot_idx += 1

    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


def crear_entorno_self_play_v6(
    agente_idx: int = 0,
    seed: Optional[int] = None,
    prob_bot: float = PROB_BOT_V2,
    min_snapshot_steps: int = MIN_SNAPSHOT_STEPS,
) -> CorazonesEnv:
    """Crea entorno Self-Play v6 (194 dims) con snapshots del directorio v6."""
    todos_snapshots = listar_snapshots_v6()
    snapshots = _filtrar_snapshots_por_calidad(
        todos_snapshots, min_snapshot_steps)

    bots = [bot_evasivo, bot_evasivo, bot_conservador, bot_agresivo]
    random.shuffle(bots)
    politicas: Dict[int, object] = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        usar_bot = random.random() < prob_bot or len(snapshots) < 2

        if not usar_bot:
            pasos = np.array([_extraer_paso_de_ruta(s) for s in snapshots])
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

        politicas[i] = bots[bot_idx % len(bots)]
        bot_idx += 1

    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


# ------------------------------------------------------------------
# VecNormalize
# ------------------------------------------------------------------
def crear_entorno_vecnormalizado(env_base, vecnorm_path=None):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    venv = DummyVecEnv([lambda: env_base])
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        print(f"  VecNormalize cargado de {vecnorm_path}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        print("  VecNormalize nuevo (norm_obs=True, norm_reward=True)")
    return venv


# ------------------------------------------------------------------
# Hiperparámetros PPO
# ------------------------------------------------------------------
def obtener_hiperparametros_ppo(logdir, device):
    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": 3e-5,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.15,
        "normalize_advantage": True,
        "ent_coef": 0.05,
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
    }


def obtener_hiperparametros_v2(logdir: str, device: str) -> Dict:
    """Hiperparámetros PPO optimizados para prevenir colapso de política.

    Cambios respecto a v1:
        - learning_rate: 3e-5 → 1e-4 (reactivar aprendizaje).
        - ent_coef: 0.05 → 0.08 (forzar más exploración).
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
        "ent_coef": 0.08,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "target_kl": 0.02,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
    }


def obtener_hiperparametros_v3(
    logdir: str,
    device: str,
    paso_actual: int = 0,
) -> Dict:
    """Hiperparámetros PPO con learning rate schedule en 3 fases.

    El agente necesita exploración agresiva al inicio y fine-tuning
    conservador al madurar. Un lr fijo de 1e-4 en modelos >5M pasos
    causa divergencia KL (early stopping frecuente) y oscilación
    de política.

    Schedule de 3 fases:
        Fase 1 (0 – 2M pasos):   lr=1e-4, ent=0.08, clip=0.15, epochs=8
        Fase 2 (2M – 5M pasos):  lr=5e-5, ent=0.05, clip=0.12, epochs=6
        Fase 3 (5M+ pasos):      lr=2e-5, ent=0.03, clip=0.10, epochs=4

    Args:
        logdir: Directorio para logs de TensorBoard.
        device: Dispositivo de cómputo ('cpu' o 'cuda').
        paso_actual: Paso global actual para seleccionar la fase.

    Returns:
        Diccionario con hiperparámetros para MaskablePPO.
    """
    if paso_actual < 2_000_000:
        # Fase 1: exploración agresiva
        lr = 1e-4
        ent = 0.08
        clip = 0.15
        epochs = 8
        grad_norm = 1.0
        fase = "1 (exploración)"
    elif paso_actual < 5_000_000:
        # Fase 2: consolidación
        lr = 5e-5
        ent = 0.05
        clip = 0.12
        epochs = 6
        grad_norm = 0.8
        fase = "2 (consolidación)"
    else:
        # Fase 3: fine-tuning conservador
        lr = 2e-5
        ent = 0.03
        clip = 0.10
        epochs = 4
        grad_norm = 0.5
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
        "target_kl": 0.02,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
        "_fase": fase,
    }


def _aplicar_snapshot_pruning(
    directorio: str,
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
) -> int:
    """Elimina los snapshots más antiguos si el pool excede el máximo.

    Mantiene los snapshots con mayor número de paso (más recientes).
    Los snapshots eliminados son los de menor paso (más antiguos/inmaduros).

    Args:
        directorio: Directorio donde se almacenan los snapshots .zip.
        max_snapshots: Número máximo de snapshots a conservar.

    Returns:
        Número de snapshots eliminados.
    """
    snaps = glob.glob(os.path.join(directorio, "snapshot_*.zip"))
    if len(snaps) <= max_snapshots:
        return 0

    # Ordenar por paso (mayor primero) y eliminar los excedentes
    snaps.sort(key=lambda p: _extraer_paso_de_ruta(p), reverse=True)
    eliminados = 0
    for snap in snaps[max_snapshots:]:
        try:
            os.remove(snap)
            # También eliminar vecnorm asociado si existe
            vecnorm_file = snap.replace(".zip", "_vecnorm.pkl")
            if os.path.exists(vecnorm_file):
                os.remove(vecnorm_file)
            eliminados += 1
        except OSError:
            pass
    return eliminados


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------
def entrenar(
    modelo,
    venv,
    env_fn,
    total_steps: int,
    snapshot_every: int = 50000,
    inicio_paso: int = 0,
    vecnorm_path: str = "",
    directorio_snapshots: str = "",
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
    eval_every: int = 0,
    eval_partidas: int = 100,
    min_win_rate: float = 0.0,
    eval_log_dir: str = "",
):
    """Bucle de entrenamiento con snapshots periódicos, pruning y auto-evaluación.

    Args:
        modelo: Instancia de MaskablePPO.
        venv: Entorno vectorizado con VecNormalize.
        env_fn: Función factoría que crea un nuevo entorno base.
        total_steps: Pasos totales a entrenar en esta sesión.
        snapshot_every: Guardar snapshot cada N pasos.
        inicio_paso: Paso global de inicio (para nombrar snapshots).
        vecnorm_path: Ruta para guardar VecNormalize.
        directorio_snapshots: Directorio donde guardar snapshots.
        max_snapshots: Máximo de snapshots a conservar (pruning).
        eval_every: Evaluar cada N snapshots (0 = no evaluar).
        eval_partidas: Partidas por evaluación.
        min_win_rate: Si WR baja de este umbral, detener entrenamiento (0=no parar).
        eval_log_dir: Directorio para guardar eval_log.jsonl.
    """
    if not directorio_snapshots:
        directorio_snapshots = DIRECTORIO_MODELOS_V5

    os.makedirs(directorio_snapshots, exist_ok=True)

    eval_log = ""
    if eval_every > 0 and eval_log_dir:
        os.makedirs(eval_log_dir, exist_ok=True)
        eval_log = os.path.join(eval_log_dir, "eval_log.jsonl")

    steps_restantes = total_steps
    paso_actual = inicio_paso
    snapshot_count = 0

    while steps_restantes > 0:
        bloque = min(steps_restantes, snapshot_every)
        modelo.learn(
            total_timesteps=bloque,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        paso_actual += bloque
        steps_restantes -= bloque
        snapshot_count += 1

        # Guardar VecNormalize
        if vecnorm_path:
            os.makedirs(os.path.dirname(vecnorm_path), exist_ok=True)
            venv.save(vecnorm_path)

        # Guardar snapshot v5
        nombre = f"snapshot_{paso_actual:010d}"
        ruta = os.path.join(directorio_snapshots, nombre)
        modelo.save(ruta)
        # Guardar también VecNormalize asociado a este snapshot
        venv.save(ruta + "_vecnorm.pkl")
        print(
            f"  [Snapshot] {ruta}.zip | "
            f"Progreso: {paso_actual}/{inicio_paso + total_steps}"
        )

        # Pruning: eliminar snapshots antiguos si exceden el máximo
        eliminados = _aplicar_snapshot_pruning(
            directorio_snapshots, max_snapshots)
        if eliminados > 0:
            print(
                f"  [Pruning] {eliminados} snapshot(s) antiguo(s) eliminado(s)")

        # Auto-evaluación periódica
        if eval_every > 0 and snapshot_count % eval_every == 0:
            from src.evaluacion import evaluar_snapshot_callback
            resultado = evaluar_snapshot_callback(
                ruta_snapshot=ruta,
                paso=paso_actual,
                log_path=eval_log,
                num_partidas=eval_partidas,
                min_win_rate=min_win_rate,
            )
            if (min_win_rate > 0
                    and resultado["win_rate_bots"] < min_win_rate
                    and paso_actual > 500_000):
                print(
                    f"\n  🛑 Early stopping: WR {resultado['win_rate_bots']:.1%} "
                    f"< umbral {min_win_rate:.1%} (paso {paso_actual:,})")
                break

    return paso_actual


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Self-Play Corazones RL v6 (194 dims, all_void)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Reanudar desde snapshot específico (.zip)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Modelo base .zip para iniciar (ej: modelos_historicos/v5/snapshot_*.zip)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Crear un modelo nuevo desde cero (sin cargar snapshots previos)")
    parser.add_argument("--transfer-from", type=str, default=None,
                        help="Transferir pesos desde snapshot 190-dim a 194-dim (padding ceros)")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--snapshot-every", type=int, default=100_000)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--prob-bot", type=float, default=PROB_BOT_V2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--logdir", type=str, default=DIRECTORIO_LOGS)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluar win rate cada N snapshots (0=no evaluar, default=5)")
    parser.add_argument("--eval-partidas", type=int, default=100,
                        help="Partidas por evaluación (default=100)")
    parser.add_argument("--min-win-rate", type=float, default=0.0,
                        help="Detener si WR < umbral (0=sin early stopping). Ej: 0.5")
    parser.add_argument("--v6", action="store_true",
                        help="Entrenar en directorio v6 (paralelo a v5, sin interferir)")
    args = parser.parse_args()

    # Directorios según modo
    if args.v6:
        dir_snapshots = DIRECTORIO_MODELOS_V6
        dir_vecnorm = DIRECTORIO_VECNORM_V6
        modo_label = "v6 (paralelo)"
    else:
        dir_snapshots = DIRECTORIO_MODELOS_V5
        dir_vecnorm = DIRECTORIO_VECNORM_V5
        modo_label = "v5"

    print("=" * 60)
    print(
        f"Self-Play Corazones RL {modo_label} (194 dims) — LR Schedule 3 fases")
    print("=" * 60)
    print(f"  Self-Play: {args.self_play}")
    print(f"  Pasos: {args.steps:,}")
    print(f"  Snapshot cada: {args.snapshot_every:,}")
    print(f"  prob_bot: {args.prob_bot:.0%}")
    print(f"  Quality filter min steps: {MIN_SNAPSHOT_STEPS:,}")
    print(f"  Max snapshots pool: {MAX_SNAPSHOTS_POOL}")
    print(f"  Snapshots → {dir_snapshots}")
    print(f"  VecNormalize → {dir_vecnorm}")
    print(f"  TensorBoard → {args.logdir}")
    if args.eval_every > 0:
        print(
            f"  Auto-Eval: cada {args.eval_every} snapshots ({args.eval_partidas} partidas)")
        if args.min_win_rate > 0:
            print(f"  Early stopping: WR < {args.min_win_rate:.0%}")
    print("-" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)

    from sb3_contrib import MaskablePPO

    os.makedirs(dir_snapshots, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(dir_vecnorm, exist_ok=True)

    vecnorm_path = os.path.join(dir_vecnorm, "v5_vecnorm.pkl")
    inicio_paso = 0
    from_scratch = args.from_scratch

    # --- Transferencia de pesos (190 → 194 dims) ---
    if args.transfer_from:
        ruta_origen = args.transfer_from
        if not ruta_origen.endswith(".zip"):
            ruta_origen += ".zip"
        if not os.path.exists(ruta_origen):
            print(f"ERROR: No se encuentra el modelo fuente: {ruta_origen}")
            sys.exit(1)

        ruta_transferido = os.path.join(
            dir_snapshots,
            f"snapshot_{_extraer_paso_de_ruta(ruta_origen):010d}"
        )
        print(f"Transferencia 190→194 desde: {ruta_origen}")
        modelo = transferir_pesos(
            ruta_origen=ruta_origen,
            ruta_destino=ruta_transferido,
            directorio_destino=dir_snapshots,
        )
        inicio_paso = _extraer_paso_de_ruta(ruta_origen)
        ruta_modelo = ruta_transferido + ".zip"
        from_scratch = False  # Ya tenemos modelo, no crear desde cero
        print(f"  [OK] Modelo 194-dim guardado. Continuando entrenamiento...")

    # --- Determinar el modelo base ---
    if args.transfer_from:
        pass  # Ya manejado arriba
    elif args.resume:
        ruta_modelo = args.resume if args.resume.endswith(
            ".zip") else args.resume + ".zip"
        print(f"Reanudando desde: {ruta_modelo}")
        inicio_paso = _extraer_paso_de_ruta(ruta_modelo)
    elif args.base_model:
        ruta_modelo = args.base_model if args.base_model.endswith(
            ".zip") else args.base_model + ".zip"
        print(f"Modelo base: {ruta_modelo}")
        inicio_paso = _extraer_paso_de_ruta(ruta_modelo)
    elif from_scratch:
        ruta_modelo = None  # se crea uno nuevo después
        print("Desde cero: creando modelo nuevo con pesos aleatorios")
    else:
        # Auto-reanudar desde el último snapshot
        listar_fn = listar_snapshots_v6 if args.v6 else listar_snapshots_v5
        snaps_auto = listar_fn()
        if snaps_auto:
            ruta_modelo = snaps_auto[-1] + ".zip"
            inicio_paso = _extraer_paso_de_ruta(ruta_modelo)
            print(
                f"Reanudando desde último snapshot {modo_label}: {ruta_modelo}")
        else:
            print(
                f"ERROR: No hay snapshots {modo_label}. Usa --from-scratch o --base-model.")
            print(f"  Los snapshots se guardan en: {dir_snapshots}")
            sys.exit(1)

    # --- Modo de entrenamiento ---
    if args.self_play:
        if args.v6:
            snaps_pool = listar_snapshots_v6()
            crear_sp = crear_entorno_self_play_v6
            sp_label = "SELF-PLAY v6"
        else:
            snaps_pool = listar_snapshots_v5()
            crear_sp = crear_entorno_self_play_v5
            sp_label = "SELF-PLAY v5"

        snaps_calidad = _filtrar_snapshots_por_calidad(
            snaps_pool, MIN_SNAPSHOT_STEPS)
        if len(snaps_calidad) >= 2:
            print(
                f"Modo {sp_label}: {len(snaps_calidad)} snapshots de calidad")
            def env_fn(): return crear_sp(
                seed=None, prob_bot=args.prob_bot)
        else:
            print(
                f"Modo BOTS ({len(snaps_calidad)} snapshots de calidad, necesitas ≥2 para self-play)")

            def env_fn(): return crear_entorno_con_bots(seed=None, shuffle_bots=True)
    else:
        print("Modo BOTS: entrenando contra heurísticos")
        def env_fn(): return crear_entorno_con_bots(seed=None, shuffle_bots=True)

    env_base = env_fn()

    # --- VecNormalize ---
    if ruta_modelo is not None:
        vecnorm_snap = ruta_modelo.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vecnorm_snap):
            venv = crear_entorno_vecnormalizado(env_base, vecnorm_snap)
            vecnorm_path = vecnorm_snap
            print(f"  VecNormalize cargado de snapshot: {vecnorm_snap}")
        elif os.path.exists(vecnorm_path):
            venv = crear_entorno_vecnormalizado(env_base, vecnorm_path)
            print(f"  VecNormalize cargado de: {vecnorm_path}")
        else:
            venv = crear_entorno_vecnormalizado(env_base)
            print("  VecNormalize nuevo (desde cero)")

        # --- Cargar modelo ---
        print(f"Cargando modelo: {ruta_modelo}")
        modelo = MaskablePPO.load(
            ruta_modelo,
            env=venv,
            device=args.device,
            tensorboard_log=args.logdir,
        )
    else:
        # --from-scratch: crear modelo nuevo
        venv = crear_entorno_vecnormalizado(env_base)
        print("  VecNormalize nuevo (desde cero)")
        hp = obtener_hiperparametros_v3(args.logdir, args.device, 0)
        policy_kwargs = hp.pop(
            "policy_kwargs", None) or obtener_policy_kwargs()
        hp.pop("_fase", None)
        modelo = MaskablePPO(
            policy=hp["policy"],
            env=venv,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            n_epochs=hp["n_epochs"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            clip_range=hp["clip_range"],
            normalize_advantage=hp["normalize_advantage"],
            ent_coef=hp["ent_coef"],
            vf_coef=hp["vf_coef"],
            max_grad_norm=hp["max_grad_norm"],
            target_kl=hp["target_kl"],
            policy_kwargs=policy_kwargs,
            verbose=hp["verbose"],
            device=hp["device"],
            tensorboard_log=hp["tensorboard_log"],
        )
        print("  [OK] Modelo nuevo creado (pesos aleatorios)")

    # --- Aplicar hiperparámetros v3 (con learning rate schedule) ---
    hp = obtener_hiperparametros_v3(args.logdir, args.device, inicio_paso)
    fase = hp.pop("_fase", "?")
    hp.pop("policy", None)
    hp.pop("policy_kwargs", None)
    hp.pop("verbose", None)
    hp.pop("device", None)
    hp.pop("tensorboard_log", None)
    for key in ["learning_rate", "ent_coef", "vf_coef", "gamma",
                "gae_lambda", "target_kl", "max_grad_norm",
                "n_steps", "batch_size", "n_epochs"]:
        setattr(modelo, key, hp[key])

    # sb3-contrib MaskablePPO espera clip_range como callable (función)
    modelo.clip_range = lambda _: hp["clip_range"]

    # Corregir: setattr no actualiza el optimizador de PyTorch
    if hasattr(modelo.policy, "optimizer") and modelo.policy.optimizer is not None:
        for param_group in modelo.policy.optimizer.param_groups:
            param_group["lr"] = hp["learning_rate"]
        print(f"  [OK] Optimizer lr actualizado a {hp['learning_rate']}")

    print(f"  Fase: {fase}")
    print(f"  lr={modelo.learning_rate}, ent_coef={modelo.ent_coef}")
    print(f"  vf_coef={modelo.vf_coef}, n_epochs={modelo.n_epochs}")
    print(
        f"  max_grad_norm={modelo.max_grad_norm}, clip_range={modelo.clip_range}")
    print("-" * 60)
    print("Métricas clave:")
    print("  train/value_loss           -> <1.0 (normalizado)")
    print("  train/entropy_loss         -> Negativo = explorando")
    print("  train/explained_variance   -> Subiendo hacia >0.8")
    print("  rollout/ep_rew_mean        -> Subiendo")
    print("-" * 60)

    try:
        paso_final = entrenar(
            modelo, venv, env_fn,
            args.steps,
            snapshot_every=args.snapshot_every,
            inicio_paso=inicio_paso,
            vecnorm_path=vecnorm_path,
            directorio_snapshots=dir_snapshots,
            max_snapshots=MAX_SNAPSHOTS_POOL,
            eval_every=args.eval_every,
            eval_partidas=args.eval_partidas,
            min_win_rate=args.min_win_rate,
            eval_log_dir=dir_snapshots,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando...")
        if vecnorm_path:
            venv.save(vecnorm_path)
        modelo.save(os.path.join(dir_snapshots, "modelo_final_v5"))
        print("Modelo guardado.")
        env_base.close()
        return

    # Guardado final
    modelo.save(os.path.join(dir_snapshots, "modelo_final_v5"))
    if vecnorm_path:
        venv.save(os.path.join(dir_vecnorm, "v5_vecnorm_final.pkl"))
    print(f"Entrenamiento v5 completado. Paso final: {paso_final}")
    env_base.close()


if __name__ == "__main__":
    main()
