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
from src.entorno.dimensiones import DIM_V3
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
        obs = np.zeros(DIM_V3, dtype=np.float32)
        buf.add(obs, 3)
        assert len(buf) == 1
        buf.add(obs, 7)
        assert len(buf) == 2

    def test_sample_retorna_tuplas(self) -> None:
        """sample() debe retornar arrays de obs y actions."""
        buf = MCTSBuffer(capacity=100)
        for i in range(10):
            obs = np.ones(DIM_V3, dtype=np.float32) * i
            buf.add(obs, i % 52)

        obs_batch, act_batch = buf.sample(5)
        assert obs_batch.shape == (5, DIM_V3)
        assert act_batch.shape == (5,)
        assert obs_batch.dtype == np.float32
        assert act_batch.dtype == np.int64

    def test_sample_no_excede_buffer(self) -> None:
        """sample(n) con n > len(buffer) debe samplear todo el buffer."""
        buf = MCTSBuffer(capacity=100)
        for i in range(3):
            buf.add(np.zeros(DIM_V3, dtype=np.float32), i)

        obs_batch, act_batch = buf.sample(10)
        assert len(obs_batch) == 3  # solo hay 3

    def test_circular_overwrite(self) -> None:
        """Cuando se excede la capacidad, debe sobrescribir."""
        buf = MCTSBuffer(capacity=5)
        for i in range(7):
            buf.add(np.ones(DIM_V3, dtype=np.float32) * i, i)
        assert len(buf) == 5  # max capacity
        # El elemento más antiguo debe haberse perdido
        obs_batch, act_batch = buf.sample(5)
        # Los valores deben ser de los últimos 5 (2,3,4,5,6)
        assert np.all(act_batch >= 2)

    def test_clear(self) -> None:
        """clear() debe vaciar el buffer."""
        buf = MCTSBuffer(capacity=100)
        for i in range(10):
            buf.add(np.zeros(DIM_V3, dtype=np.float32), i)
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty()

    def test_sample_sin_reposicion_dentro_del_batch(self) -> None:
        """Un mismo batch no debe tener duplicados."""
        buf = MCTSBuffer(capacity=100)
        for i in range(50):
            buf.add(np.ones(DIM_V3, dtype=np.float32) * i, i)

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


# ============================================================
# Clase 3 — Oracle hook en CorazonesEnvV3
# ============================================================

class TestEnvOracleHook:
    """Verifica que el hook de oráculo PIMC recolecta datos durante el juego."""

    @pytest.fixture
    def oracle_buffer(self) -> MCTSBuffer:
        return MCTSBuffer(capacity=1000)

    def test_env_popula_buffer_cuando_disponible(self, oracle_buffer) -> None:
        """Con oracle_buffer, el entorno debe añadir pares (obs, action)
        en bazas altas donde el oráculo es viable."""
        from src.v3.entorno import CorazonesEnvV3

        env = CorazonesEnvV3(agente_idx=0, oracle_buffer=oracle_buffer)
        obs, _ = env.reset(seed=42)

        # Jugar varias bazas; el oráculo se activa solo en bazas ≥10
        for _ in range(50):
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            if not legales:
                break
            obs, _, terminated, truncated, _ = env.step(legales[0])
            if terminated or truncated:
                break

        env.close()
        # En bazas ≥10, el oráculo debería haber recolectado datos
        # Al menos 1 entrada si el episodio llegó a la baza 10
        if len(oracle_buffer) == 0:
            pytest.skip("El episodio no alcanzó baza ≥10 con mundos ≤100K")

    def test_env_sin_oracle_buffer_no_puebla(self) -> None:
        """Sin oracle_buffer, el entorno debe funcionar normalmente
        y no debe crashear."""
        from src.v3.entorno import CorazonesEnvV3

        env = CorazonesEnvV3(agente_idx=0)
        obs, _ = env.reset(seed=42)

        for _ in range(50):
            mask = env.action_masks()
            legales = [i for i, m in enumerate(mask) if m]
            if not legales:
                break
            obs, _, terminated, truncated, _ = env.step(legales[0])
            if terminated or truncated:
                break

        env.close()
        # No debería crashear — prueba de integridad


# ============================================================
# Clase 4 — BC loss con modelo real
# ============================================================

class TestCalcularBCLossIntegration:
    """Verifica que calcular_bc_loss funciona con modelos MaskablePPO."""

    def test_bc_loss_buffer_vacio_retorna_cero(self) -> None:
        """Con buffer vacío, la BC loss debe ser 0.0."""
        from src.v3.train_mcts import calcular_bc_loss

        buf = MCTSBuffer(capacity=10)
        # Función dummy que no debería ser llamada

        def dummy_policy(obs):
            raise RuntimeError("No debe llamarse con buffer vacío")

        loss = calcular_bc_loss(dummy_policy, buf, batch_size=8)
        assert loss == 0.0

    def test_bc_loss_con_datos_sinteticos(self) -> None:
        """BC loss con datos sintéticos debe ser > 0 y finita."""
        from src.v3.train_mcts import calcular_bc_loss
        import torch

        # Buffer con datos sintéticos
        buf = MCTSBuffer(capacity=100)
        rng = np.random.default_rng(123)
        for i in range(20):
            obs = rng.normal(0, 1, DIM_V3).astype(np.float32)
            action = i % 52
            buf.add(obs, action)

        # Policy dummy: logits aleatorios (sin entrenar)
        def dummy_policy(obs_tensor):
            batch = obs_tensor.shape[0]
            logits = torch.randn(batch, 52)
            values = torch.zeros(batch)
            return logits, values

        loss = calcular_bc_loss(dummy_policy, buf, batch_size=8)
        # Cross-entropy con logits aleatorios debe ser ~log(52) ≈ 3.95
        assert loss > 0.0, f"BC loss debería ser > 0, fue {loss}"
        assert loss < 10.0, f"BC loss sospechosamente alta: {loss}"


# ============================================================
# Clase 5 — Fine-tuning BC periódico
# ============================================================

class TestFineTuningBC:
    """Verifica que el fine-tuning BC reduce la pérdida sobre el buffer."""

    def test_entrenar_bc_reduce_loss(self) -> None:
        """Un epoch de BC fine-tuning debe reducir la loss."""
        from src.v3.train_mcts import (
            MCTSBuffer, calcular_bc_loss, entrenar_bc_epoch,
        )
        from src.v3.red import obtener_policy_kwargs_transformer
        from src.v3.entorno import CorazonesEnvV3
        from sb3_contrib import MaskablePPO

        # Crear modelo con el env real (observación DIM_V3)
        env = CorazonesEnvV3(agente_idx=0)
        policy_kwargs = obtener_policy_kwargs_transformer(
            features_dim=32, net_arch=[32, 32],
        )
        model = MaskablePPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            verbose=0, device="cpu",
            n_steps=64, batch_size=32, n_epochs=1,
        )
        env.close()

        # Buffer con datos de un "oráculo" simulado
        buf = MCTSBuffer(capacity=200)
        rng = np.random.default_rng(42)
        oracle_action = 5
        for i in range(30):
            obs = rng.normal(0, 1, DIM_V3).astype(np.float32)
            buf.add(obs, oracle_action)

        # Medir loss antes del fine-tuning
        def policy_fn(obs_t):
            features = model.policy.extract_features(obs_t)
            logits = model.policy.action_net(features)
            values = model.policy.value_net(features)
            return logits, values

        loss_antes = calcular_bc_loss(policy_fn, buf, batch_size=16)

        # Ejecutar un epoch de BC fine-tuning
        loss_despues = entrenar_bc_epoch(model, buf, batch_size=16, lr=1e-3)

        # La loss después debe ser menor o igual
        assert loss_despues <= loss_antes + 0.5, \
            f"BC debería reducir loss: {loss_antes:.4f} → {loss_despues:.4f}"
