"""
Tests TDD para el pipeline de entrenamiento v2_ronda.

Verifica que el entorno single-hand es compatible con:
  - VecNormalize (norm_obs + norm_reward)
  - MaskablePPO (creacion y learn)
  - Self-play con oponentes mixtos (bots + snapshots)
"""
import os
import tempfile
import numpy as np
import pytest

from src.entorno.dimensiones import DIM_ENTORNO as dim_entorno


class TestV2VecNormalize:
    """Verifica compatibilidad CorazonesEnvSingleHand + VecNormalize."""

    def test_venv_wraps_single_hand(self):
        """VecNormalize envuelve CorazonesEnvSingleHand sin errores."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand

        def _make_env():
            return CorazonesEnvSingleHand(agente_idx=0)
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        obs = venv.reset()
        assert obs.shape == (1, dim_entorno)

    def test_venv_step_retorna_reward_normalizado(self):
        """VecNormalize produce rewards finitos."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand

        def _make_env():
            return CorazonesEnvSingleHand(agente_idx=0)
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        venv.reset()
        for _ in range(10):
            # Usar action_masks del env subyacente
            mask = venv.venv.envs[0].action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            action = [legal[0]]
            obs, reward, done, info = venv.step(action)
            assert np.isfinite(
                reward[0]), f"Reward={reward[0]} debe ser finito"


class TestV2ModelCreation:
    """Verifica que MaskablePPO se crea con el entorno v2."""

    def test_crear_modelo_ppo(self):
        """MaskablePPO se crea sin errores con CorazonesEnvSingleHand."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand

        def _make_env():
            return CorazonesEnvSingleHand(agente_idx=0)
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

        modelo = MaskablePPO(
            "MlpPolicy", venv,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            verbose=0,
            device="cpu",
        )
        assert modelo is not None

    def test_learn_pocos_pasos(self):
        """MaskablePPO.learn() ejecuta sin errores en v2."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand

        def _make_env():
            return CorazonesEnvSingleHand(agente_idx=0)
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        venv.reset()

        modelo = MaskablePPO(
            "MlpPolicy", venv,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            verbose=0,
            device="cpu",
        )
        modelo.learn(total_timesteps=256, progress_bar=False)
        # Si llega aqui sin excepcion, el test pasa


class TestV2SelfPlay:
    """Verifica que el entorno v2 funciona con oponentes mixtos."""

    def test_oponentes_mixtos(self):
        """Entorno single-hand con bots + BotExperto funciona."""
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        from src.agentes.heuristicos import bot_evasivo, bot_conservador, bot_agresivo
        from src.agentes.bot_experto import BotExperto

        politicas = {
            1: bot_evasivo,
            2: bot_conservador,
            3: BotExperto(),
        }
        env = CorazonesEnvSingleHand(
            agente_idx=0, politicas_oponentes=politicas)
        obs, _ = env.reset(seed=42)

        terminated = False
        steps = 0
        total_reward = 0.0
        while not terminated and steps < 60:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            _, reward, terminated, truncated, _ = env.step(legal[0])
            total_reward += reward
            steps += 1

        assert terminated, f"Mano deberia terminar, steps={steps}"
        assert np.isfinite(total_reward)

    def test_save_and_load_snapshot(self):
        """Guardar y cargar un snapshot de MaskablePPO entrenado en v2."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand

        def _make_env():
            return CorazonesEnvSingleHand(agente_idx=0)
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        venv.reset()

        modelo = MaskablePPO(
            "MlpPolicy", venv,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            verbose=0,
            device="cpu",
        )
        modelo.learn(total_timesteps=256, progress_bar=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test_model")
            modelo.save(zip_path)

            # Recargar
            loaded = MaskablePPO.load(zip_path, device="cpu")
            assert loaded is not None
            # Verificar que predice sin errores
            obs = venv.reset()
            action, _ = loaded.predict(
                obs, action_masks=venv.venv.envs[0].action_masks())
            assert 0 <= action[0] <= 51


class TestV2Evaluacion:
    """Verifica que las funciones de evaluacion cargan VecNormalize correctamente."""

    def test_evaluar_score_promedio_sin_venv(self):
        """evaluar_score_promedio funciona sin VecNormalize."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        from src.v2_ronda.train import evaluar_score_promedio

        def _make_env():
            return CorazonesEnvSingleHand()
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        venv.reset()

        modelo = MaskablePPO("MlpPolicy", venv, verbose=0, device="cpu")
        modelo.learn(total_timesteps=256, progress_bar=False)

        resultado = evaluar_score_promedio(
            modelo, venv_stats_path=None, num_hands=5)
        assert "score_promedio" in resultado
        assert resultado["total_hands"] == 5
        assert 0 <= resultado["score_promedio"] <= 26

    def test_evaluar_score_promedio_con_venv(self):
        """evaluar_score_promedio funciona con VecNormalize stats cargadas."""
        import tempfile
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        from src.v2_ronda.train import evaluar_score_promedio

        def _make_env():
            return CorazonesEnvSingleHand()
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        venv.reset()

        modelo = MaskablePPO("MlpPolicy", venv, verbose=0, device="cpu")
        modelo.learn(total_timesteps=512, progress_bar=False)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            venv.save(f.name)
            vn_path = f.name

        try:
            resultado = evaluar_score_promedio(
                modelo, venv_stats_path=vn_path, num_hands=5)
            assert "score_promedio" in resultado
            assert resultado["total_hands"] == 5
            assert 0 <= resultado["score_promedio"] <= 26
        finally:
            os.unlink(vn_path)

    def test_evaluar_vs_experto_v2_funciona(self):
        """evaluar_vs_experto_v2 no lanza excepciones."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        from src.v2_ronda.train import evaluar_vs_experto_v2

        def _make_env():
            return CorazonesEnvSingleHand()
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        venv.reset()

        modelo = MaskablePPO("MlpPolicy", venv, verbose=0, device="cpu")
        modelo.learn(total_timesteps=256, progress_bar=False)

        resultado = evaluar_vs_experto_v2(
            modelo, venv_stats_path=None, num_hands=3)
        assert "score_promedio" in resultado
        assert resultado["total_hands"] == 3
