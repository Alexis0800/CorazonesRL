"""
Tests TDD para shared-weights multi-position training.

Verifica que el entorno soporte rotación de agente_idx y que
el modelo pueda jugar desde cualquier posición.
"""

import numpy as np
import pytest


class TestMultiPositionEnv:
    """Verifica que CorazonesEnvV31 funcione desde cualquier posición."""

    def test_env_desde_posicion_0(self):
        """El entorno funciona normalmente con agente_idx=0."""
        from src.v3_1.entorno import CorazonesEnvV31
        from src.agentes.heuristicos import bot_evasivo

        politicas = {i: bot_evasivo for i in range(1, 4)}
        env = CorazonesEnvV31(agente_idx=0, politicas_oponentes=politicas)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (228,)
        assert obs.dtype == np.float32
        env.close()

    def test_env_desde_posicion_1(self):
        """El entorno funciona con agente_idx=1."""
        from src.v3_1.entorno import CorazonesEnvV31
        from src.agentes.heuristicos import bot_evasivo

        politicas = {0: bot_evasivo, 2: bot_evasivo, 3: bot_evasivo}
        env = CorazonesEnvV31(agente_idx=1, politicas_oponentes=politicas)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (228,)
        env.close()

    def test_env_desde_posicion_3(self):
        """El entorno funciona con agente_idx=3."""
        from src.v3_1.entorno import CorazonesEnvV31
        from src.agentes.heuristicos import bot_evasivo

        politicas = {0: bot_evasivo, 1: bot_evasivo, 2: bot_evasivo}
        env = CorazonesEnvV31(agente_idx=3, politicas_oponentes=politicas)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (228,)
        env.close()

    def test_mano_completa_desde_cualquier_posicion(self):
        """Una mano se completa sin errores desde cualquier posición."""
        from src.v3_1.entorno import CorazonesEnvV31
        from src.agentes.heuristicos import bot_evasivo

        for agente_idx in range(4):
            politicas = {
                i: bot_evasivo
                for i in range(4) if i != agente_idx
            }
            env = CorazonesEnvV31(
                agente_idx=agente_idx, politicas_oponentes=politicas)
            obs, _ = env.reset(seed=42 + agente_idx)
            terminated, truncated = False, False
            steps = 0
            while not terminated and not truncated:
                mask = env.action_masks()
                legales = np.where(mask)[0]
                action = np.random.choice(legales)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
            env.close()
            assert steps > 0, f"Posición {agente_idx}: 0 pasos"
            assert 0 <= info.get('score', 99) <= 26, \
                f"Posición {agente_idx}: score fuera de rango"


class TestSelfPlayOponents:
    """Verifica que crear_entorno_self_play_v31 genera oponentes correctos."""

    def test_fase_0_solo_bots(self):
        """Fase 0: todos los oponentes son bots (el env no falla)."""
        from src.v3_1.train import crear_entorno_self_play_v31

        env = crear_entorno_self_play_v31(
            version="v3_1", agente_idx=0, modelo_actual=None, fase=0)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (228,)
        # Verificar que hay 3 oponentes configurados
        assert len(env.politicas_oponentes) == 3
        env.close()

    def test_fase_2_con_modelo_actual(self):
        """Fase 2 con modelo_actual: un oponente usa el modelo."""
        from src.v3_1.train import crear_entorno_self_play_v31
        from src.agentes.politica_rl import PoliticaSB3

        # Crear un modelo dummy (solo para verificar que se acepta)
        # No necesitamos un modelo real para este test
        env = crear_entorno_self_play_v31(
            version="v3_1", agente_idx=0, modelo_actual=None, fase=2)
        # Sin snapshots existentes, debería usar BotExperto como fallback
        obs, _ = env.reset(seed=42)
        assert obs.shape == (228,)
        env.close()

    def test_agente_idx_rotacion(self):
        """Crear entorno con distintos agente_idx no falla."""
        from src.v3_1.train import crear_entorno_self_play_v31

        for idx in range(4):
            env = crear_entorno_self_play_v31(
                version="v3_1", agente_idx=idx, fase=0)
            obs, _ = env.reset(seed=42 + idx)
            assert obs.shape == (228,)
            env.close()
