"""
Tests del entorno RLlib (CorazonesEnvRLlib) — PARTIDA COMPLETA (v10).

Cubre:
  - Espacios de observación y acción correctos.
  - reset() devuelve obs válida.
  - step() con acción legal no crashea.
  - Un episodio = una partida completa: termina cuando alguien llega a 100.
  - El marcador persiste entre manos (no se reinicia cada 13 steps).
  - Recompensa terminal por puesto + info con puesto/top2.
  - Action mask tiene al menos 1 acción legal en cada step.
  - Compatibilidad con gymnasium check_env.
"""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.entorno.corazones_rllib import CorazonesEnvRLlib
from src.entorno.dimensiones import DIM_ENTORNO


@pytest.fixture
def env():
    e = CorazonesEnvRLlib({"obs_dim": DIM_ENTORNO, "agente_idx": 0})
    yield e
    e.close()


class TestEspacios:
    def test_observation_space_keys(self, env):
        assert "obs" in env.observation_space.spaces
        assert "action_mask" in env.observation_space.spaces

    def test_obs_shape(self, env):
        assert env.observation_space["obs"].shape == (DIM_ENTORNO,)

    def test_mask_shape(self, env):
        assert env.observation_space["action_mask"].shape == (52,)

    def test_action_space_es_discrete_52(self, env):
        from gymnasium import spaces
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 52


class TestReset:
    def test_reset_devuelve_obs_valida(self, env):
        obs, info = env.reset()
        assert "obs" in obs
        assert "action_mask" in obs
        assert obs["obs"].shape == (DIM_ENTORNO,)
        assert obs["action_mask"].shape == (52,)

    def test_reset_mask_tiene_al_menos_una_accion(self, env):
        obs, _ = env.reset()
        assert obs["action_mask"].sum() >= 1

    def test_reset_obs_dtype_float32(self, env):
        obs, _ = env.reset()
        assert obs["obs"].dtype == np.float32
        assert obs["action_mask"].dtype == np.float32

    def test_reset_obs_en_rango_01(self, env):
        obs, _ = env.reset()
        assert np.all(obs["obs"] >= 0.0)
        assert np.all(obs["obs"] <= 1.0)
        assert np.all((obs["action_mask"] == 0.0) | (obs["action_mask"] == 1.0))


class TestStep:
    def _accion_legal(self, obs: dict) -> int:
        mask = obs["action_mask"]
        legales = np.where(mask == 1.0)[0]
        assert len(legales) > 0, "Sin acciones legales"
        return int(legales[0])

    def test_step_con_accion_legal_no_crashea(self, env):
        obs, _ = env.reset()
        action = self._accion_legal(obs)
        obs2, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert truncated is False

    def test_step_mask_siempre_tiene_legal(self, env):
        obs, _ = env.reset()
        for _ in range(500):
            action = self._accion_legal(obs)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
            assert obs["action_mask"].sum() >= 1, "Máscara vacía en step no terminal"

    def test_partida_completa_termina_y_alguien_llega_a_100(self, env):
        obs, _ = env.reset()
        pasos = 0
        terminated = False
        info = {}
        while not terminated:
            action = self._accion_legal(obs)
            obs, _, terminated, _, info = env.step(action)
            pasos += 1
            assert pasos <= 5000, "Partida no terminó en un número razonable de pasos"
        assert terminated
        # Una partida dura varias manos → bastante más de 13 steps
        assert pasos > 13
        assert max(info["scores_finales"]) >= 100

    def test_info_terminal_tiene_puesto_y_top2(self, env):
        obs, _ = env.reset()
        terminated = False
        info = {}
        while not terminated:
            action = self._accion_legal(obs)
            obs, _, terminated, _, info = env.step(action)
        assert info["puesto"] in (1, 2, 3, 4)
        assert info["top2"] == (info["puesto"] <= 2)
        assert info["gano_partida"] == (info["puesto"] == 1)
        assert info["manos_jugadas"] >= 1

    def test_reward_terminal_en_rango_esperado(self, env):
        obs, _ = env.reset()
        terminated = False
        ultimo_reward = 0.0
        while not terminated:
            action = self._accion_legal(obs)
            obs, ultimo_reward, terminated, _, _ = env.step(action)
        # Reward terminal = R_terminal (±1, ±0.3) + shaping acotado. Holgura amplia.
        assert -3.0 <= ultimo_reward <= 3.0

    def test_reward_es_escalar_finito(self, env):
        obs, _ = env.reset()
        for _ in range(500):
            action = self._accion_legal(obs)
            obs, reward, terminated, _, _ = env.step(action)
            assert np.isfinite(reward), f"Reward no finita: {reward}"
            if terminated:
                break


class TestMarcadorPersistente:
    def test_marcador_no_se_reinicia_cada_mano(self, env):
        """Tras superar 13 steps (≥1 mano completa), debe haber puntos acumulados
        en el marcador (no todo en cero como pasaría con episodios de 1 mano)."""
        obs, _ = env.reset()
        terminated = False
        scores_intermedios = None
        pasos = 0
        while not terminated and pasos < 60:
            action = np.where(obs["action_mask"] == 1.0)[0][0]
            obs, _, terminated, _, _ = env.step(int(action))
            pasos += 1
            scores_intermedios = env._motor.puntuaciones_historicas()
        # Tras ~60 steps (varias manos) el marcador acumulado debe ser > 0
        assert sum(scores_intermedios) > 0


class TestDiferentesPositions:
    @pytest.mark.parametrize("agente_idx", [0, 1, 2, 3])
    def test_todas_posiciones_juegan_partida_completa(self, agente_idx):
        env = CorazonesEnvRLlib({"obs_dim": DIM_ENTORNO, "agente_idx": agente_idx})
        obs, _ = env.reset()
        terminated = False
        pasos = 0
        while not terminated:
            mask = obs["action_mask"]
            legales = np.where(mask == 1.0)[0]
            assert len(legales) > 0
            obs, _, terminated, _, _ = env.step(int(legales[0]))
            pasos += 1
            assert pasos <= 5000
        assert pasos > 13
        env.close()


class TestCheckEnv:
    def test_check_env_pasa(self):
        env = CorazonesEnvRLlib({"obs_dim": DIM_ENTORNO, "agente_idx": 0})
        check_env(env, warn=True, skip_render_check=True)
        env.close()
