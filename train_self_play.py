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
DIRECTORIO_LOGS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs")
DIRECTORIO_VECNORM = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "vecnormalize")


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
        """Construye observación de 187 dims desde la perspectiva de jugador_idx."""
        obs = np.zeros(187, dtype=np.float32)
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

        # Vacíos (no disponibles desde el motor)
        # Puntajes históricos (no disponibles desde el motor)
        return obs


# ------------------------------------------------------------------
# Factoría de entornos
# ------------------------------------------------------------------
def crear_entorno_con_bots(agente_idx=0, seed=None, shuffle_bots=True):
    bots = [bot_conservador, bot_agresivo, bot_evasivo]
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


def crear_entorno_self_play(agente_idx=0, seed=None, prob_bot=0.15):
    """Crea entorno Self-Play: 85% snapshots, 15% bots."""
    snapshots = listar_snapshots()
    bots = [bot_conservador, bot_agresivo, bot_evasivo]
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


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------
def entrenar(modelo, venv, env_fn, total_steps, snapshot_every=50000, inicio_paso=0, vecnorm_path=""):
    steps_restantes = total_steps
    paso_actual = inicio_paso
    while steps_restantes > 0:
        bloque = min(steps_restantes, snapshot_every)
        modelo.learn(total_timesteps=bloque,
                     reset_num_timesteps=False, progress_bar=True)
        paso_actual += bloque
        steps_restantes -= bloque
        if vecnorm_path:
            os.makedirs(os.path.dirname(vecnorm_path), exist_ok=True)
            venv.save(vecnorm_path)
        ruta = os.path.join(DIRECTORIO_MODELOS, f"snapshot_{paso_actual:010d}")
        modelo.save(ruta)
        print(
            f"  [Snapshot] {ruta}.zip | Progreso: {paso_actual}/{inicio_paso + total_steps}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Self-Play Corazones RL v4")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--snapshot-every", type=int, default=50_000)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--prob-bot", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--logdir", type=str, default=DIRECTORIO_LOGS)
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Play Corazones RL v4")
    print("=" * 60)
    print(f"  Self-Play: {args.self_play}")
    print(f"  Pasos: {args.steps:,}")
    print(f"  Snapshot cada: {args.snapshot_every:,}")
    snapshots = listar_snapshots()
    print(f"  Snapshots disponibles: {len(snapshots)}")
    print("-" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)

    from sb3_contrib import MaskablePPO

    vecnorm_path = os.path.join(DIRECTORIO_VECNORM, "vecnorm.pkl")

    if args.self_play and len(snapshots) >= 2:
        print("Modo SELF-PLAY: oponentes = snapshots históricos")
        def env_fn(): return crear_entorno_self_play(seed=None, prob_bot=args.prob_bot)
    else:
        print("Modo BOTS: entrenando contra heurísticos")
        def env_fn(): return crear_entorno_con_bots(seed=None, shuffle_bots=True)

    env_base = env_fn()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(DIRECTORIO_VECNORM, exist_ok=True)

    # Cargar modelo existente (el último snapshot)
    if args.resume:
        ruta_modelo = args.resume if args.resume.endswith(
            ".zip") else args.resume + ".zip"
    else:
        # Usar el snapshot más reciente como base
        ruta_modelo = snapshots[-1] + ".zip" if snapshots else None
        if not ruta_modelo:
            print("ERROR: No hay snapshots. Entrena primero sin --self-play")
            sys.exit(1)

    print(f"Cargando modelo base: {ruta_modelo}")

    # Cargar VecNormalize si existe
    vecnorm_snap = ruta_modelo.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_snap):
        venv = crear_entorno_vecnormalizado(env_base, vecnorm_snap)
        vecnorm_path = vecnorm_snap
    else:
        venv = crear_entorno_vecnormalizado(env_base)

    modelo = MaskablePPO.load(ruta_modelo, env=venv,
                              device=args.device, tensorboard_log=args.logdir)

    # Aplicar hiperparámetros de Self-Play
    hp = obtener_hiperparametros_ppo(args.logdir, args.device)
    for key in ["learning_rate", "ent_coef", "vf_coef", "gamma",
                "gae_lambda", "target_kl", "max_grad_norm", "n_steps", "batch_size"]:
        setattr(modelo, key, hp[key])

    print(f"  lr={modelo.learning_rate}, ent_coef={modelo.ent_coef}")
    print(f"  vf_coef={modelo.vf_coef}, clip_range={modelo.clip_range}")
    print("-" * 60)
    print("Métricas clave:")
    print("  train/value_loss           → <1.0 (normalizado)")
    print("  train/entropy_loss         → Negativo = explorando")
    print("  train/explained_variance   → Subiendo hacia >0.8")
    print("  rollout/ep_rew_mean        → Subiendo")
    print("-" * 60)

    try:
        entrenar(modelo, venv, env_fn, args.steps,
                 args.snapshot_every, vecnorm_path=vecnorm_path)
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando...")
        if vecnorm_path:
            venv.save(vecnorm_path)
        modelo.save(os.path.join(DIRECTORIO_MODELOS, "modelo_final"))

    modelo.save(os.path.join(DIRECTORIO_MODELOS, "modelo_final"))
    if vecnorm_path:
        venv.save(os.path.join(DIRECTORIO_VECNORM, "vecnorm_final.pkl"))
    print("Entrenamiento Self-Play completado.")
    env_base.close()


if __name__ == "__main__":
    main()
