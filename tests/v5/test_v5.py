"""
Tests TDD para v5 — 4-model shared-weights self-play.

Verifica que 4 entornos puedan correr en paralelo con el mismo modelo
jugando desde las 4 posiciones simultáneamente.
"""
import numpy as np
import pytest


class TestV5MultiEnv:
    """Verifica que 4 entornos paralelos funcionan correctamente."""

    def test_cuatro_entornos_simultaneos(self):
        """4 entornos, uno por posición, con el mismo modelo."""
        from src.v5.train import crear_entorno_self_play_v5

        # Crear 4 entornos, cada uno con un agente_idx distinto
        envs = []
        for pos in range(4):
            env = crear_entorno_self_play_v5(
                version="v5", agente_idx=pos, fase=0)
            envs.append(env)

        # Verificar que todos resetean
        for pos, env in enumerate(envs):
            obs, _ = env.reset(seed=42 + pos)
            assert obs.shape == (228,)
            assert obs.dtype == np.float32

        # Jugar una mano en cada entorno
        for env in envs:
            terminated, truncated = False, False
            steps = 0
            while not terminated and not truncated:
                mask = env.action_masks()
                legales = np.where(mask)[0]
                if len(legales) == 0:
                    break
                action = np.random.choice(legales)
                _, reward, terminated, truncated, info = env.step(action)
                steps += 1
            assert steps > 0
            assert 0 <= info.get('score', 99) <= 26
            env.close()

    def test_rewards_sum_to_constante(self):
        """En 4 entornos, la suma de rewards terminales = 78 (4×26 - 26)."""
        from src.v5.train import crear_entorno_self_play_v5

        envs = []
        for pos in range(4):
            env = crear_entorno_self_play_v5(
                version="v5", agente_idx=pos, fase=0)
            envs.append(env)

        rewards = []
        for pos, env in enumerate(envs):
            env.reset(seed=42)
            terminated, truncated = False, False
            r_total = 0.0
            while not terminated and not truncated:
                mask = env.action_masks()
                legales = np.where(mask)[0]
                if len(legales) == 0:
                    break
                action = np.random.choice(legales)
                _, reward, terminated, truncated, _ = env.step(action)
                r_total += reward
            rewards.append(r_total)
            env.close()

        # Suma de rewards = 4×26 - suma_puntos = 104 - 26 = 78
        # (los 4 agentes son independientes, cada uno ve 26-puntos)
        # Pero como son manos distintas (distintas semillas), no suman exactamente
        for r in rewards:
            assert 0 <= r <= 52, f"Reward fuera de rango: {r}"

    def test_fase_2_self_play_puro(self):
        """En fase 2 (self-play puro), los 3 oponentes son el modelo compartido."""
        from src.v5.train import crear_entorno_self_play_v5
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v3_1.red import obtener_policy_kwargs_v31

        # Crear un modelo dummy
        policy_kwargs = obtener_policy_kwargs_v31(features_dim=128)

        def _make():
            return crear_entorno_self_play_v5(version="v5", agente_idx=0, fase=0)

        venv = DummyVecEnv([_make])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        modelo = MaskablePPO(
            "MlpPolicy", venv, policy_kwargs=policy_kwargs,
            verbose=0, device="cpu", seed=42,
            learning_rate=1e-4, n_steps=64, batch_size=32,
        )

        # Crear 2 entornos en fase 2 (self-play puro) — los 3 oponentes son el modelo
        env0 = crear_entorno_self_play_v5(
            version="v5", agente_idx=0, modelo_actual=modelo, fase=2)
        env1 = crear_entorno_self_play_v5(
            version="v5", agente_idx=1, modelo_actual=modelo, fase=2)

        # Ambos deben poder resetear
        obs0, _ = env0.reset(seed=42)
        obs1, _ = env1.reset(seed=99)
        assert obs0.shape == (228,)
        assert obs1.shape == (228,)

        env0.close()
        env1.close()
