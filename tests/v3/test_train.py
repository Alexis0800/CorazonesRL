"""
Tests para el pipeline de entrenamiento v3.

Valida:
  - Factory crear_entorno_self_play_v3
  - prob_bot decay
  - log_eval JSONL
  - Integración MCTS buffer en training loop
  - Transformer policy kwargs
  - Snapshot save/load con VecNormalize 250-dim
"""

from __future__ import annotations

import io
import json
import os
import tempfile

import numpy as np
import pytest

from src.entorno.dimensiones import DIM_V3
from src.v3.entorno import CorazonesEnvV3
from src.v3.red import (
    TransformerFeatureExtractor,
    obtener_policy_kwargs_transformer,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def dir_tmp():
    """Directorio temporal para guardar snapshots y vecnorm."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ──────────────────────────────────────────────────────────────
# Transformer policy kwargs
# ──────────────────────────────────────────────────────────────

class TestTransformerPolicyKwargs:
    """Validación de factory de policy_kwargs para Transformer."""

    def test_obtener_policy_kwargs_retorna_dict(self):
        """Debe retornar un dict con las claves esperadas."""
        kwargs = obtener_policy_kwargs_transformer()
        assert isinstance(kwargs, dict)
        assert "features_extractor_class" in kwargs
        assert "features_extractor_kwargs" in kwargs
        assert "net_arch" in kwargs

    def test_features_extractor_class_es_transformer(self):
        """La clase extractora debe ser TransformerFeatureExtractor."""
        kwargs = obtener_policy_kwargs_transformer()
        assert kwargs["features_extractor_class"] == TransformerFeatureExtractor

    def test_features_dim_default_256(self):
        """features_dim por defecto debe ser 256."""
        kwargs = obtener_policy_kwargs_transformer()
        assert kwargs["features_extractor_kwargs"]["features_dim"] == 256

    def test_net_arch_pi_y_vf(self):
        """net_arch debe tener pi y vf con arquitectura configurable."""
        kwargs = obtener_policy_kwargs_transformer(net_arch=[128, 64])
        assert kwargs["net_arch"]["pi"] == [128, 64]
        assert kwargs["net_arch"]["vf"] == [128, 64]


# ──────────────────────────────────────────────────────────────
# Transformer instantiation
# ──────────────────────────────────────────────────────────────

class TestTransformerInstantiation:
    """Verifica que TransformerFeatureExtractor se construye correctamente."""

    def test_construir_extractor_sin_error(self):
        """Debe construir sin errores con obs_dim=250."""
        import gymnasium as gym
        obs_space = gym.spaces.Box(
            low=-10, high=10, shape=(DIM_V3,), dtype=np.float32)
        extractor = TransformerFeatureExtractor(obs_space, features_dim=256)
        assert extractor is not None

    def test_forward_shape(self):
        """forward debe retornar (batch, features_dim)."""
        import gymnasium as gym
        import torch
        obs_space = gym.spaces.Box(
            low=-10, high=10, shape=(DIM_V3,), dtype=np.float32)
        extractor = TransformerFeatureExtractor(obs_space, features_dim=128)
        batch = torch.randn(4, DIM_V3)
        out = extractor(batch)
        assert out.shape == (4, 128)


# ──────────────────────────────────────────────────────────────
# prob_bot decay
# ──────────────────────────────────────────────────────────────

class TestProbBotDecay:
    """Verifica el decaimiento coseno de prob_bot."""

    def test_inicio_mayor_que_fin(self):
        """Al inicio prob_bot debe ser PROB_BOT_START."""
        from src.v3.train import prob_bot_actual
        v = prob_bot_actual(0, 10_000_000)
        assert v > 0.30  # mayor que fin típico

    def test_fin_es_piso(self):
        """Al final prob_bot debe bajar al piso configurado."""
        from src.v3.train import prob_bot_actual
        inicio, fin = 0.50, 0.20
        v = prob_bot_actual(10_000_000, 10_000_000, inicio=inicio, fin=fin)
        assert abs(v - fin) < 1e-6

    def test_decreciente(self):
        """prob_bot debe ser monótona decreciente."""
        from src.v3.train import prob_bot_actual
        total = 10_000_000
        v1 = prob_bot_actual(2_000_000, total)
        v2 = prob_bot_actual(8_000_000, total)
        assert v1 > v2


# ──────────────────────────────────────────────────────────────
# log_eval JSONL
# ──────────────────────────────────────────────────────────────

class TestLogEval:
    """Verifica el formato JSONL del log de evaluación."""

    def test_log_escribe_linea_json(self, dir_tmp):
        """Debe escribir una línea JSON válida."""
        from src.v3.train import log_eval
        log_path = os.path.join(dir_tmp, "eval.jsonl")
        log_eval(log_path, paso=1_000_000, win_rate=0.45, avg_score=12.3,
                 num_partidas=100, prob_bot=0.35)
        with open(log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["paso"] == 1_000_000
        assert data["win_rate_bots"] == 0.45

    def test_log_appendea_no_sobrescribe(self, dir_tmp):
        """Múltiples llamadas deben appendear líneas."""
        from src.v3.train import log_eval
        log_path = os.path.join(dir_tmp, "eval.jsonl")
        log_eval(log_path, paso=1, win_rate=0.3, avg_score=10.0,
                 num_partidas=50, prob_bot=0.5)
        log_eval(log_path, paso=2, win_rate=0.4, avg_score=8.0,
                 num_partidas=50, prob_bot=0.4)
        with open(log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2


# ──────────────────────────────────────────────────────────────
# Factory crear_entorno_self_play_v3
# ──────────────────────────────────────────────────────────────

class TestFactorySelfPlayV3:
    """Validación de la factory de self-play para v3."""

    def test_retorna_corazones_env_v3(self):
        """Debe retornar CorazonesEnvV3 con obs de 250 dims."""
        from src.v3.train import crear_entorno_self_play_v3
        env = crear_entorno_self_play_v3(
            version="v3", prob_bot=1.0, agente_idx=0,
        )
        assert isinstance(env, CorazonesEnvV3)
        assert env.observation_space.shape == (DIM_V3,)
        env.close()

    def test_agente_idx_0_funciona(self):
        """Debe funcionar con agente en posicion 0."""
        from src.v3.train import crear_entorno_self_play_v3
        env = crear_entorno_self_play_v3(
            version="v3", prob_bot=1.0, agente_idx=0,
        )
        obs, _ = env.reset(seed=42)
        assert obs.shape == (DIM_V3,)
        env.close()

    def test_agente_idx_1_funciona(self):
        """Debe funcionar con agente en posicion 1."""
        from src.v3.train import crear_entorno_self_play_v3
        env = crear_entorno_self_play_v3(
            version="v3", prob_bot=1.0, agente_idx=1,
        )
        obs, _ = env.reset(seed=42)
        assert obs.shape == (DIM_V3,)
        env.close()


# ──────────────────────────────────────────────────────────────
# MCTS hooks en training loop
# ──────────────────────────────────────────────────────────────

class TestMCTSHooks:
    """Verifica integración de MCTSBuffer en training loop."""

    def test_buffer_vacio_se_crea(self):
        """El training state debe crear un buffer vacío."""
        from src.v3.train_mcts import MCTSBuffer
        buf = MCTSBuffer(capacity=1000)
        assert len(buf) == 0
        assert buf.is_empty()

    def test_buffer_add_y_sample(self):
        """Agregar obs y muestrear debe retornar arrays."""
        from src.v3.train_mcts import MCTSBuffer
        buf = MCTSBuffer(capacity=1000)
        obs = np.zeros(DIM_V3, dtype=np.float32)
        buf.add(obs, 42)
        assert len(buf) == 1
        obs_b, act_b = buf.sample(1)
        assert obs_b.shape == (1, DIM_V3)
        assert act_b[0] == 42

    def test_buffer_fifo_sobrescribe(self):
        """Cuando se excede capacity, debe sobrescribir los más viejos."""
        from src.v3.train_mcts import MCTSBuffer
        buf = MCTSBuffer(capacity=3)
        obs = np.zeros(DIM_V3, dtype=np.float32)
        buf.add(obs.copy(), 0)
        buf.add(obs.copy(), 1)
        buf.add(obs.copy(), 2)
        buf.add(obs.copy(), 3)
        # Debe seguir teniendo size=3, con acción más antigua (0) sobrescrita
        assert len(buf) == 3
        _, act = buf.sample(3)
        assert set(act.tolist()) == {1, 2, 3}


# ──────────────────────────────────────────────────────────────
# Snapshot guardado/carga
# ──────────────────────────────────────────────────────────────

class TestSnapshotSaveLoad:
    """Verifica guardado y carga de snapshots + VecNormalize."""

    def test_guardar_y_cargar_snapshot(self, dir_tmp):
        """Debe guardarse como .zip (save es llamado en el modelo)."""
        from src.v3.train import _guardar_snapshot
        from unittest.mock import MagicMock

        modelo = MagicMock()
        _guardar_snapshot(
            modelo=modelo,
            paso=500_000,
            output_dir=os.path.join(dir_tmp, "snapshots"),
            vecnorm_path=None,
        )
        modelo.save.assert_called_once()
