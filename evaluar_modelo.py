"""
Script de evaluación para el modelo entrenado de Corazones.
Mide el win rate contra 3 bots heurísticos en N partidas.

IMPORTANTE: Carga el modelo sin VecNormalize para máxima compatibilidad.
Si el modelo fue entrenado con norm_obs, normaliza observaciones manualmente.

Uso:
    python evaluar_modelo.py [--ruta RUTA] [--partidas N]
"""

import argparse
import os
import sys
import pickle
import numpy as np
from src.entorno import CorazonesEnv
from src.bots import bot_conservador, bot_agresivo, bot_evasivo


def normalizar_obs_si_hay_stats(obs: np.ndarray, vecnorm_path: str) -> np.ndarray:
    """Intenta normalizar observación con stats de VecNormalize. Si falla, devuelve raw."""
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return obs
    try:
        with open(vecnorm_path, "rb") as f:
            data = pickle.load(f)
        obs_rms = data.get("obs_rms", None)
        if obs_rms is None:
            return obs
        mean = np.array(obs_rms.mean)
        var = np.array(obs_rms.var)
        return np.clip((obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0).astype(np.float32)
    except Exception:
        return obs


def evaluar(ruta_modelo: str, num_partidas: int, vecnorm_path: str) -> float:
    """Evalúa el modelo contra 3 bots heurísticos.

    Args:
        ruta_modelo: Ruta al modelo .zip (sin extensión).
        num_partidas: Número de partidas.
        vecnorm_path: Ruta al .pkl de VecNormalize (opcional).

    Returns:
        Win rate (0.0 a 1.0).
    """
    # Añadir .zip si no lo tiene
    if not ruta_modelo.endswith(".zip"):
        ruta_modelo += ".zip"

    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No se encuentra {ruta_modelo}")
        sys.exit(1)

    # Cargar modelo sin entorno para evitar recursión con VecNormalize
    from sb3_contrib import MaskablePPO
    try:
        # Forzar device y evitar cargar el env
        import torch
        modelo = MaskablePPO.load(ruta_modelo, device="cpu")
    except Exception as e:
        print(f"ERROR al cargar modelo: {e}")
        print("Intentando con device=None...")
        modelo = MaskablePPO.load(ruta_modelo, device=None, env=None)

    bots = [bot_conservador, bot_agresivo, bot_evasivo]
    victorias = 0

    for seed in range(num_partidas):
        env = CorazonesEnv(
            agente_idx=0,
            politicas_oponentes={1: bots[0], 2: bots[1], 3: bots[2]},
        )
        obs_raw, _ = env.reset(seed=seed)
        obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
        done = False

        while not done:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True)
            obs_raw, reward, terminated, truncated, _ = env.step(int(action))
            obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
            done = terminated or truncated

        punt_agente = env._puntuacion_historica[0]
        punt_rivales = [env._puntuacion_historica[i] for i in range(1, 4)]

        if punt_agente < min(punt_rivales):
            victorias += 1

        env.close()

        if (seed + 1) % 25 == 0:
            print(
                f"  {seed + 1}/{num_partidas} partidas... ({victorias} victorias)")

    return victorias / num_partidas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluar modelo de Corazones RL")
    parser.add_argument(
        "--ruta", type=str, default="modelos_historicos/modelo_final",
        help="Ruta al modelo (con o sin .zip)"
    )
    parser.add_argument(
        "--partidas", type=int, default=100,
        help="Número de partidas"
    )
    args = parser.parse_args()

    # Detectar VecNormalize asociado
    ruta_base = args.ruta.replace(".zip", "")
    vecnorm_path = None
    for candidato in [ruta_base + "_vecnorm.pkl", "vecnormalize/vecnorm.pkl"]:
        if os.path.exists(candidato):
            vecnorm_path = candidato
            break

    print("=" * 55)
    print(f"Evaluando: {ruta_base}.zip")
    print(f"Partidas: {args.partidas}")
    print(f"Oponentes: conservador, agresivo, evasivo")
    print(f"VecNormalize: {vecnorm_path or 'NO (usando obs crudas)'}")
    print("-" * 55)

    win_rate = evaluar(ruta_base, args.partidas, vecnorm_path)

    victorias = int(win_rate * args.partidas)
    print("-" * 55)
    print(f"Win rate: {victorias}/{args.partidas} = {win_rate:.1%}")
    print("=" * 55)
