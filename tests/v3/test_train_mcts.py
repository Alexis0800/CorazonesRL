"""
Tests unitarios para MCTS-guided training (v3).

Verifica:
  - MCTSBuffer: add, sample, len, clear
  - evaluar_con_oraculo: retorna acción óptima según PIMC
  - Integración con MotorCorazones
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.v3.train_mcts import (
    MCTSBuffer,
    evaluar_con_oraculo,
)


class TestMCTSBuffer:
    """Suite de tests para el buffer circular MCTS."""

    def test_buffer_vacio(self) -> None:
        """Un buffer nuevo debe estar vacío."""
        buf = MCTSBuffer(capacity=100)
        assert len(buf) == 0
        assert buf.is_empty()

    def test_add_y_len(self) -> None:
        """Añadir elementos incrementa len."""
        buf = MCTSBuffer(capacity=100)
        obs = np.zeros(250, dtype=np.float32)
        buf.add(obs, 3)
        assert len(buf) == 1
        buf.add(obs, 7)
        assert len(buf) == 2

    def test_sample_retorna_tuplas(self) -> None:
        """sample() debe retornar arrays de obs y actions."""
        buf = MCTSBuffer(capacity=100)
        for i in range(10):
            obs = np.ones(250, dtype=np.float32) * i
            buf.add(obs, i % 52)

        obs_batch, act_batch = buf.sample(5)
        assert obs_batch.shape == (5, 250)
        assert act_batch.shape == (5,)
        assert obs_batch.dtype == np.float32
        assert act_batch.dtype == np.int64

    def test_sample_no_excede_buffer(self) -> None:
        """sample(n) con n > len(buffer) debe samplear todo el buffer."""
        buf = MCTSBuffer(capacity=100)
        for i in range(3):
            buf.add(np.zeros(250, dtype=np.float32), i)

        obs_batch, act_batch = buf.sample(10)
        assert len(obs_batch) == 3  # solo hay 3

    def test_circular_overwrite(self) -> None:
        """Cuando se excede la capacidad, debe sobrescribir."""
        buf = MCTSBuffer(capacity=5)
        for i in range(7):
            buf.add(np.ones(250, dtype=np.float32) * i, i)
        assert len(buf) == 5  # max capacity
        # El elemento más antiguo debe haberse perdido
        obs_batch, act_batch = buf.sample(5)
        # Los valores deben ser de los últimos 5 (2,3,4,5,6)
        assert np.all(act_batch >= 2)

    def test_clear(self) -> None:
        """clear() debe vaciar el buffer."""
        buf = MCTSBuffer(capacity=100)
        for i in range(10):
            buf.add(np.zeros(250, dtype=np.float32), i)
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty()

    def test_sample_sin_reposicion_dentro_del_batch(self) -> None:
        """Un mismo batch no debe tener duplicados."""
        buf = MCTSBuffer(capacity=100)
        for i in range(50):
            buf.add(np.ones(250, dtype=np.float32) * i, i)

        obs_batch, act_batch = buf.sample(10)
        # Verificar que no hay duplicados en las acciones del batch
        assert len(set(act_batch.tolist())) == 10


class TestEvaluarConOráculo:
    """Suite de tests para la evaluación con oráculo PIMC."""

    @pytest.fixture
    def motor_avanzado(self) -> MotorCorazones:
        """Motor con varias bazas ya jugadas (baza ≥ 8)."""
        import random
        random.seed(42)  # determinismo en repartir()

        m = MotorCorazones()
        m.repartir()
        # Forzar bazas rápidas jugando cartas legales
        for _ in range(7 * 4):  # 7 bazas completas
            actual = m.obtener_jugador_actual()
            legales = m.obtener_jugadas_legales(actual)
            if legales:
                m.jugar_carta(actual, legales[0])
        return m

    def test_evaluar_retorna_carta_legal(self, motor_avanzado) -> None:
        """La acción óptima debe ser una carta legal."""
        rng = np.random.default_rng(42)
        legales = motor_avanzado.obtener_jugadas_legales(0)
        if len(legales) >= 2:
            mejor, _ = evaluar_con_oraculo(
                motor_avanzado, 0, legales,
                num_mundos=50, rng=rng,
            )
            assert mejor in legales, \
                f"La carta {mejor} no es legal. Legales: {[c.id for c in legales]}"

    def test_evaluar_retorna_score_dict(self, motor_avanzado) -> None:
        """Debe retornar scores para cada carta legal."""
        rng = np.random.default_rng(42)
        legales = motor_avanzado.obtener_jugadas_legales(0)
        if len(legales) >= 2:
            _, scores = evaluar_con_oraculo(
                motor_avanzado, 0, legales,
                num_mundos=50, rng=rng,
            )
            assert len(scores) == len(legales)
            for c in legales:
                assert c.id in scores

    def test_mejor_carta_minimiza_score(self, motor_avanzado) -> None:
        """La carta óptima debe tener el menor score esperado."""
        rng = np.random.default_rng(42)
        legales = motor_avanzado.obtener_jugadas_legales(0)
        if len(legales) >= 2:
            mejor, scores = evaluar_con_oraculo(
                motor_avanzado, 0, legales,
                num_mundos=100, rng=rng,
            )
            mejor_score = scores[mejor.id]
            for c in legales:
                assert scores[c.id] >= mejor_score - 0.01, \
                    f"Carta {c.id} tiene score {scores[c.id]} < {mejor_score}"
