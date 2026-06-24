"""Tests TDD para v4 — self-play puro con bots heurísticos."""
import numpy as np
import pytest


class TestV4Entorno:
    """Verifica que el entorno v4 funciona desde cualquier posición."""

    def test_env_v4_existe(self):
        from src.v4.entorno import CorazonesEnvV4
        from src.agentes.heuristicos import bot_evasivo
        politicas = {i: bot_evasivo for i in range(1, 4)}
        env = CorazonesEnvV4(agente_idx=0, politicas_oponentes=politicas)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (228,)
        env.close()

    def test_terminal_only_reward(self):
        """Recompensa solo al final de la mano, no intermedia."""
        from src.v4.entorno import CorazonesEnvV4
        from src.agentes.heuristicos import bot_evasivo
        politicas = {i: bot_evasivo for i in range(1, 4)}
        env = CorazonesEnvV4(agente_idx=0, politicas_oponentes=politicas)
        obs, _ = env.reset(seed=42)
        terminated, truncated = False, False
        recompensas = []
        while not terminated and not truncated:
            mask = env.action_masks()
            legales = np.where(mask)[0]
            action = np.random.choice(legales)
            obs, reward, terminated, truncated, info = env.step(action)
            recompensas.append(reward)
        env.close()
        # Solo la última recompensa debe ser no-cero
        intermedias = recompensas[:-1]
        assert all(r == 0.0 for r in intermedias), \
            f"Recompensas intermedias no son cero: {intermedias}"
        assert recompensas[-1] != 0.0, "Recompensa terminal es cero"
        assert 0 <= recompensas[-1] <= 52, \
            f"Recompensa terminal fuera de rango: {recompensas[-1]}"


class TestV4SelfPlay:
    """Verifica que el self-play v4 solo usa bots heurísticos."""

    def test_fase_0_solo_bots(self):
        """Fase 0: solo bots heurísticos, sin BotExperto."""
        from src.v4.train import crear_entorno_self_play_v4
        env = crear_entorno_self_play_v4(version="v4", agente_idx=0, fase=0)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (228,)
        assert len(env.politicas_oponentes) == 3
        env.close()

    def test_rotacion_posiciones(self):
        """El entorno soporta las 4 posiciones."""
        from src.v4.train import crear_entorno_self_play_v4
        for idx in range(4):
            env = crear_entorno_self_play_v4(
                version="v4", agente_idx=idx, fase=0)
            obs, _ = env.reset(seed=42 + idx)
            assert obs.shape == (228,)
            env.close()
