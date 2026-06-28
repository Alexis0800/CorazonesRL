"""
Tests del entorno con fase de PASE habilitada (v10b).

Cubre: obs de 228 dims, que la mano 1 entra en fase de pase, que las 3 primeras
sub-decisiones son selección de cartas a pasar, la transición a juego, que una
partida completa con pase termina, y compatibilidad con check_env.
"""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.entorno.corazones_rllib import CorazonesEnvRLlib
from src.entorno.dimensiones import DIM_V12


def _env():
    return CorazonesEnvRLlib({"obs_dim": DIM_V12, "agente_idx": 0,
                              "random_position": False, "con_pase": True})


def _legal(obs):
    return int(np.where(obs["action_mask"] == 1.0)[0][0])


class TestConfig:
    def test_requiere_dim_228(self):
        with pytest.raises(ValueError):
            CorazonesEnvRLlib({"obs_dim": 224, "con_pase": True})

    def test_obs_dim_228(self):
        env = _env()
        obs, _ = env.reset()
        assert obs["obs"].shape == (DIM_V12,)


class TestFasePase:
    def test_mano_1_entra_en_fase_pase(self):
        env = _env()
        obs, _ = env.reset()
        # Mano 1 = pase a la izquierda → feature de fase de pase activa.
        assert obs["obs"][224] == 1.0
        assert obs["obs"][225] == pytest.approx(0.33, abs=0.01)  # izquierda
        # La máscara de pase = 13 cartas de la mano.
        assert obs["action_mask"].sum() == 13

    def test_tres_subdecisiones_de_pase(self):
        env = _env()
        obs, _ = env.reset()
        # Selección 1 y 2: sigue en fase de pase, con menos cartas seleccionables.
        for esperado_sel in (1, 2):
            obs, r, term, trunc, _ = env.step(_legal(obs))
            assert r == 0.0 and not term
            assert obs["obs"][224] == 1.0
            assert obs["obs"][226] == pytest.approx(esperado_sel / 3.0, abs=0.01)
            assert obs["action_mask"].sum() == 13 - esperado_sel
        # Selección 3: se ejecuta el pase → pasamos a fase de juego.
        obs, r, term, trunc, _ = env.step(_legal(obs))
        assert not term
        assert obs["obs"][224] == 0.0  # ya no es fase de pase
        # Ahora la máscara son jugadas legales (la mano sigue con 13 tras el intercambio).
        assert obs["action_mask"].sum() >= 1

    def test_mano_sin_pase_no_entra_en_fase(self):
        # Forzamos el estado a una mano sin pase (numero_mano múltiplo de 4).
        env = _env()
        env.reset()
        # mano 4 = sin pase: re-iniciar manualmente el motor en esa mano.
        env._motor.numero_mano = 4
        env._iniciar_mano()
        obs = env._build_obs()
        assert obs["obs"][224] == 0.0


class TestPartidaCompletaConPase:
    def test_partida_termina(self):
        env = _env()
        obs, _ = env.reset()
        terminated = False
        pasos = 0
        info = {}
        while not terminated:
            obs, _, terminated, _, info = env.step(_legal(obs))
            pasos += 1
            assert pasos <= 8000
        assert max(info["scores_finales"]) >= 100
        assert info["puesto"] in (1, 2, 3, 4)

    def test_obs_siempre_228_y_mask_no_vacia(self):
        env = _env()
        obs, _ = env.reset()
        for _ in range(300):
            assert obs["obs"].shape == (DIM_V12,)
            assert obs["action_mask"].sum() >= 1
            obs, _, term, _, _ = env.step(_legal(obs))
            if term:
                break


class TestCheckEnv:
    def test_check_env_con_pase(self):
        env = _env()
        check_env(env, skip_render_check=True)
        env.close()
