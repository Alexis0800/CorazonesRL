"""
Tests TDD para el entorno single-hand (v2_ronda).

CorazonesEnvSingleHand: una mano = un episodio.
- Sin puntajes historicos (todos empiezan en 0)
- Sin contexto de partida multi-mano
- Reward score-based (CalculadoraRecompensasScore)
- Action masking via action_masks()
- Auto-play de oponentes
"""
import numpy as np
import pytest
from src.entorno.dimensiones import DIM_ENTRENAMIENTO


class TestCorazonesEnvSingleHand:
    """Tests basicos del entorno single-hand."""

    @pytest.fixture
    def env(self):
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        return CorazonesEnvSingleHand()

    def test_observation_space(self, env):
        """El espacio de observacion es Box(220,) float32."""
        assert env.observation_space.shape == (DIM_ENTRENAMIENTO,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self, env):
        """El espacio de accion es Discrete(52)."""
        assert env.action_space.n == 52

    def test_reset_retorna_obs_e_info(self, env):
        """reset() retorna (obs, info)."""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (DIM_ENTRENAMIENTO,)
        assert isinstance(info, dict)

    def test_reset_con_seed_reproducible(self):
        """Dos resets con misma seed producen la misma observacion."""
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        env1 = CorazonesEnvSingleHand()
        env2 = CorazonesEnvSingleHand()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert np.array_equal(obs1, obs2)

    def test_step_retorna_tuple(self, env):
        """step() retorna (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        mask = env.action_masks()
        legal_actions = np.where(mask)[0]
        action = legal_actions[0]
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_mask_legales(self, env):
        """action_masks() retorna array con al menos una accion legal."""
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.sum() >= 1, "Debe haber al menos una accion legal"
        assert mask.dtype == np.bool_

    def test_terminated_after_full_hand(self, env):
        """Despues de 52 jugadas (13 bazas), terminated=True."""
        env.reset(seed=42)
        steps = 0
        terminated = False
        while not terminated and steps < 60:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            action = legal[0]  # siempre la primera legal
            _, reward, terminated, truncated, _ = env.step(action)
            steps += 1
        assert terminated, f"El episodio deberia terminar, steps={steps}"
        assert steps <= 56, f"Max 52 + auto-play extra, fueron {steps}"

    def test_reward_acumulado_es_finito(self, env):
        """El reward acumulado en una mano es finito (no NaN, no inf)."""
        env.reset(seed=42)
        total_reward = 0.0
        terminated = False
        while not terminated:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            action = legal[0]
            _, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward), f"Reward no finito: {reward}"
            total_reward += reward
        assert np.isfinite(total_reward)

    def test_info_contiene_metadatos(self, env):
        """info contiene los metadatos esperados."""
        env.reset(seed=42)
        mask = env.action_masks()
        action = np.where(mask)[0][0]
        _, _, _, _, info = env.step(action)
        assert "agente_idx" in info
        assert "puntos_mano" in info
        assert "numero_baza" in info

    def test_no_hay_puntuacion_historica_en_obs(self, env):
        """Las dimensiones de puntaje historico [172:176] deben ser cero."""
        obs, _ = env.reset(seed=42)
        assert np.all(obs[172:176] == 0.0), \
            "No debe haber puntaje historico en v2 (single-hand)"


class TestCorazonesEnvSingleHandMultiAgent:
    """Tests con diferentes agentes y oponentes."""

    def test_agente_idx_1_funciona(self):
        """El agente puede ser el jugador 1."""
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        env = CorazonesEnvSingleHand(agente_idx=1)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (DIM_ENTRENAMIENTO,)

    def test_agente_idx_3_funciona(self):
        """El agente puede ser el jugador 3."""
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        env = CorazonesEnvSingleHand(agente_idx=3)
        obs, _ = env.reset(seed=42)
        mask = env.action_masks()
        assert mask.sum() >= 1

    def test_oponentes_heuristicos(self):
        """Se pueden asignar bots heuristicos como oponentes."""
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        from src.agentes.heuristicos import bot_evasivo, bot_conservador, bot_agresivo
        politicas = {
            1: bot_evasivo,
            2: bot_conservador,
            3: bot_agresivo,
        }
        env = CorazonesEnvSingleHand(
            agente_idx=0, politicas_oponentes=politicas)
        obs, _ = env.reset(seed=42)
        # Ejecutar una mano completa
        terminated = False
        steps = 0
        while not terminated and steps < 60:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            _, _, terminated, truncated, _ = env.step(legal[0])
            steps += 1
        assert terminated

    def test_agente_idx_invalido(self):
        """agente_idx fuera de rango lanza ValueError."""
        from src.v2_ronda.entorno import CorazonesEnvSingleHand
        with pytest.raises(ValueError):
            CorazonesEnvSingleHand(agente_idx=4)
        with pytest.raises(ValueError):
            CorazonesEnvSingleHand(agente_idx=-1)
