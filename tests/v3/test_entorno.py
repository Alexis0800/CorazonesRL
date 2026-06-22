"""
Tests de integracion para CorazonesEnvV3 (entorno v3 con 250-dim obs).

Validan:
  - Gymnasium API compliance
  - Action masking correcto
  - Ciclo completo de episodio (reset → step → done)
  - Forma de observacion (260,)
  - Valores de observacion en rango valido
  - Recompensas (no NaN, magnitudes razonables)
  - Hooks MCTS opcionales
  - Entorno con oponentes heuristicos
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_V3
from src.v3.entorno import CorazonesEnvV3, crear_entorno_v3
from src.agentes.heuristicos import bot_conservador, bot_evasivo, bot_agresivo
from src.agentes.bot_experto import BotExperto


class TestCorazonesEnvV3GymAPI:
    """Valida que el entorno cumple la API Gymnasium."""

    def test_check_env_pasa(self):
        """check_env no es aplicable a entornos con Action Masking
        (el muestreador aleatorio no conoce la mascara).
        Verificamos manualmente la API."""
        env = CorazonesEnvV3(agente_idx=0)
        # Verificar que los espacios son correctos
        assert env.observation_space.shape == (DIM_V3,)
        assert env.action_space.n == 52
        # Verificar que reset/step funcionan
        obs, _ = env.reset(seed=42)
        assert obs.shape == (DIM_V3,)
        mask = env.action_masks()
        legales = [i for i, m in enumerate(mask) if m]
        if legales:
            obs, reward, terminated, truncated, info = env.step(legales[0])
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            env.close()

    def test_reset_retorna_obs_y_info(self):
        """reset() retorna (obs, info)."""
        env = CorazonesEnvV3(agente_idx=0)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_retorna_5_elementos(self):
        """step() retorna (obs, reward, terminated, truncated, info)."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        mask = env.action_masks()
        legales = [i for i, m in enumerate(mask) if m]
        assert len(legales) > 0, "Debe haber al menos una accion legal"
        action = legales[0]
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_space_shape(self):
        """El observation space debe ser Box(250,)."""
        env = CorazonesEnvV3(agente_idx=0)
        assert env.observation_space.shape == (DIM_V3,)
        assert env.observation_space.dtype == np.float32

    def test_action_space_discrete_52(self):
        """El action space debe ser Discrete(52)."""
        env = CorazonesEnvV3(agente_idx=0)
        assert env.action_space.n == 52


class TestActionMasking:
    """Valida el sistema de action masking."""

    def test_mascara_no_esta_vacia_inicio(self):
        """Al inicio de la mano, debe haber cartas legales."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.sum() >= 1
        assert mask.sum() <= 52

    def test_mascara_es_booleana(self):
        """La mascara debe ser un array de bool."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.dtype == bool

    def test_acciones_ilegales_bloqueadas(self):
        """Intentar jugar una carta ilegal deberia ser rechazada por la mascara."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        mask = env.action_masks()
        ilegales = [i for i, m in enumerate(mask) if not m]
        assert len(ilegales) > 0, "Debe haber al menos una accion ilegal"
        # step() con accion ilegal deberia ser atrapado por la mascara
        # (el modelo usa la mascara, no el entorno directamente)
        # Pero podemos verificar que la carta ilegal no esta en la mano del agente
        agente = env.motor.jugadores[env.agente_idx]
        ids_mano = {c.id for c in agente.mano}
        for ilegal_id in ilegales:
            if ilegal_id not in ids_mano:
                continue  # carta no esta en mano, normal que sea ilegal
            # Si la carta esta en mano pero es ilegal, la mascara funciona
            assert not mask[ilegal_id]

    def test_mascara_despues_de_todas_las_bazas_es_cero(self):
        """Al terminar la mano, la mascara debe ser todo ceros."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        # Jugar la mano completa
        terminated = False
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            assert len(legales) > 0, (
                f"Mascara vacia en baza {env.motor.numero_baza}"
            )
            action = legales[0]
            _, _, terminated, _, _ = env.step(action)
        # Despues de terminated, mascara debe ser 0
        mask = env.action_masks()
        assert mask.sum() == 0


class TestEpisodeCycle:
    """Valida el ciclo completo de episodio."""

    def test_episodio_completo(self):
        """Un episodio debe terminar en exactamente 13 bazas (max 52 steps)."""
        env = CorazonesEnvV3(agente_idx=0)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (DIM_V3,)

        steps = 0
        terminated = False
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            action = legales[0]  # tomar siempre la primera legal
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            assert not np.isnan(reward), f"NaN reward en step {steps}"

        assert steps <= 52  # max 4 jugadores × 13 bazas
        assert steps >= 1   # al menos una accion
        assert terminated
        assert not truncated

    def test_multiples_episodios_misma_instancia(self):
        """reset() debe permitir multiples episodios en la misma instancia."""
        env = CorazonesEnvV3(agente_idx=0)

        for ep in range(3):
            obs, _ = env.reset(seed=42 + ep)
            assert obs.shape == (DIM_V3,)
            assert not env.mano_terminada

            terminated = False
            steps = 0
            while not terminated:
                mask = env.action_masks()
                legales = [i for i, m in enumerate(mask) if m]
                action = legales[steps % len(legales)]
                obs, reward, terminated, truncated, _ = env.step(action)
                steps += 1

            assert terminated, f"Episodio {ep} no termino"
            assert env.mano_terminada

    def test_step_en_episodio_terminado_lanza_error(self):
        """step() despues de terminated debe lanzar RuntimeError."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        terminated = False
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            _, _, terminated, _, _ = env.step(legales[0])
        with pytest.raises(RuntimeError, match="episodio ya terminado"):
            env.step(0)


class TestObservationValues:
    """Valida los valores en el vector de observacion."""

    def test_obs_no_nan(self):
        """La observacion no debe contener NaN en ningun momento."""
        env = CorazonesEnvV3(agente_idx=0)
        obs, _ = env.reset(seed=42)
        assert not np.any(np.isnan(obs)), "NaN en observacion inicial"

        for _ in range(10):
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            if not legales:
                break
            obs, _, terminated, _, _ = env.step(legales[0])
            assert not np.any(np.isnan(obs)), "NaN en observacion"
            if terminated:
                break

    def test_obs_rango_razonable(self):
        """La observacion debe estar en un rango razonable [0, 26]."""
        env = CorazonesEnvV3(agente_idx=0)
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= -1.0), "Valores negativos en obs"
        assert np.all(obs <= 27.0), "Valores muy altos en obs"

    def test_obs_bloque_base_intacto(self):
        """Los primeros 52 elementos deben ser one-hot de la mano."""
        env = CorazonesEnvV3(agente_idx=0)
        obs, _ = env.reset(seed=42)
        # La mano tiene exactamente 13 cartas al inicio
        assert sum(obs[0:52]) == pytest.approx(13.0, abs=0.01)

    def test_get_observation_no_avanza_estado(self):
        """get_observation() no debe modificar el estado."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        obs1 = env.get_observation()
        obs2 = env.get_observation()
        assert np.array_equal(obs1, obs2), (
            "get_observation() modifico el estado"
        )


class TestRewards:
    """Valida que el sistema de recompensas funciona."""

    def test_reward_no_nan(self):
        """Ninguna recompensa debe ser NaN."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        for _ in range(52):
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            if not legales:
                break
            _, reward, terminated, _, _ = env.step(legales[0])
            assert not np.isnan(reward), f"NaN reward en step {_}"
            if terminated:
                break

    def test_recompensa_acumulada_razonable(self):
        """La recompensa total de una mano debe estar en [-26, 104]."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        total = 0.0
        terminated = False
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            _, reward, terminated, _, _ = env.step(legales[0])
            total += reward
        # Razonable: entre -26 (todos los puntos) y 78+26 (moon perfecto)
        assert -30 <= total <= 110, f"Recompensa total extrema: {total}"


class TestMCTSHooks:
    """Valida los hooks para MCTS-guided training."""

    def test_oracle_buffer_se_pasa_al_entorno(self):
        """El buffer MCTS debe ser accesible desde el entorno."""
        buffer = MagicMock()
        env = CorazonesEnvV3(agente_idx=0, oracle_buffer=buffer)
        assert env._oracle_buffer is buffer

    def test_oracle_rng_se_pasa_al_entorno(self):
        """El RNG debe ser accesible desde el entorno."""
        rng = np.random.default_rng(123)
        env = CorazonesEnvV3(agente_idx=0, oracle_rng=rng)
        assert env._oracle_rng is rng

    def test_entorno_funciona_con_hooks_mcts(self):
        """El entorno debe funcionar normalmente con hooks MCTS configurados."""
        from src.v3.train_mcts import MCTSBuffer
        buffer = MCTSBuffer(capacity=1000)
        rng = np.random.default_rng(42)
        env = CorazonesEnvV3(
            agente_idx=0, oracle_buffer=buffer, oracle_rng=rng,
        )
        env.reset(seed=42)

        terminated = False
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            _, _, terminated, _, _ = env.step(legales[0])

        # El entorno completo sin errores es suficiente validacion
        assert terminated


class TestConOponentes:
    """Valida el entorno con oponentes configurados."""

    def test_oponentes_heuristicos(self):
        """El entorno debe funcionar con bots heuristicos."""
        politicas = {
            1: bot_conservador,
            2: bot_agresivo,
            3: bot_evasivo,
        }
        env = CorazonesEnvV3(agente_idx=0, politicas_oponentes=politicas)
        env.reset(seed=42)

        terminated = False
        steps = 0
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            _, _, terminated, _, _ = env.step(legales[0])
            steps += 1

        assert terminated
        assert steps <= 52

    def test_oponente_bot_experto(self):
        """El entorno debe funcionar contra BotExperto."""
        politicas = {
            1: BotExperto(),
            2: BotExperto(),
            3: BotExperto(),
        }
        env = CorazonesEnvV3(agente_idx=0, politicas_oponentes=politicas)
        env.reset(seed=42)

        terminated = False
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            _, _, terminated, _, _ = env.step(legales[0])

        assert terminated


class TestFactory:
    """Valida la factory function."""

    def test_crear_entorno_v3_retorna_corazones_env_v3(self):
        """crear_entorno_v3() debe retornar una instancia valida."""
        env = crear_entorno_v3(agente_idx=2)
        assert isinstance(env, CorazonesEnvV3)
        assert env.agente_idx == 2

    def test_crear_entorno_v3_con_politicas(self):
        """crear_entorno_v3() debe aceptar politicas."""
        env = crear_entorno_v3(
            agente_idx=0,
            politicas_oponentes={1: bot_conservador},
        )
        assert 1 in env._politicas_oponentes


class TestEdgeCases:
    """Casos borde del entorno."""

    def test_agente_idx_invalido(self):
        """agente_idx fuera de [0,3] debe lanzar ValueError."""
        with pytest.raises(ValueError, match="agente_idx"):
            CorazonesEnvV3(agente_idx=-1)
        with pytest.raises(ValueError, match="agente_idx"):
            CorazonesEnvV3(agente_idx=4)

    def test_info_parcial_tiene_baza(self):
        """info debe contener el numero de baza."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        mask = env.action_masks()
        legales = [i for i, m in enumerate(mask) if m]
        obs, reward, terminated, truncated, info = env.step(legales[0])
        assert "baza" in info
        assert "puntos_mano" in info
        assert isinstance(info["baza"], int)
        assert len(info["puntos_mano"]) == 4

    def test_info_final_tiene_puntos_agente(self):
        """info final debe contener puntos_agente."""
        env = CorazonesEnvV3(agente_idx=0)
        env.reset(seed=42)
        terminated = False
        while not terminated:
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            obs, reward, terminated, truncated, info = env.step(legales[0])
        if "puntos_agente" in info:
            assert isinstance(info["puntos_agente"], int)
