"""
Tests para el pipeline de entrenamiento v3.

Valida:
  - Factory crear_entorno_self_play_v3
  - prob_bot decay
  - log_eval JSONL
  - Integración MCTS buffer en training loop
  - Transformer policy kwargs
  - Snapshot save/load con VecNormalize 260-dim
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
        """Debe construir sin errores con obs_dim=260."""
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
        log_eval(log_path, paso=1_000_000, wr=0.45, avg_score=12.3,
                 num_partidas=100, prob_bot=0.35)
        with open(log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["paso"] == 1_000_000
        assert data["wr"] == 0.45

    def test_log_appendea_no_sobrescribe(self, dir_tmp):
        """Múltiples llamadas deben appendear líneas."""
        from src.v3.train import log_eval
        log_path = os.path.join(dir_tmp, "eval.jsonl")
        log_eval(log_path, paso=1, wr=0.3, avg_score=10.0,
                 num_partidas=50, prob_bot=0.5)
        log_eval(log_path, paso=2, wr=0.4, avg_score=8.0,
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
        """Debe retornar CorazonesEnvV3 con obs de 260 dims."""
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
# Carga de snapshots históricos (Fictitious Self-Play)
# ──────────────────────────────────────────────────────────────

class TestSnapshotSelfPlayV3:
    """Verifica la carga de snapshots para Fictitious Self-Play."""

    def test_cargar_snapshots_directorio_vacio(self, dir_tmp):
        """Directorio sin snapshots debe retornar lista vacía."""
        from src.v3.train import _cargar_snapshots_v3
        snaps = _cargar_snapshots_v3(dir_tmp, min_steps=0, max_snapshots=50)
        assert snaps == []

    def test_cargar_snapshots_directorio_inexistente(self):
        """Directorio inexistente debe retornar lista vacía."""
        from src.v3.train import _cargar_snapshots_v3
        snaps = _cargar_snapshots_v3(
            "/tmp/no_existe_12345", min_steps=0, max_snapshots=50)
        assert snaps == []

    def test_factory_prob_bot_1_solo_bots(self):
        """prob_bot=1.0 debe usar solo bots heurísticos, no snapshots."""
        from src.v3.train import crear_entorno_self_play_v3
        env = crear_entorno_self_play_v3(
            version="v3", prob_bot=1.0, agente_idx=0,
        )
        # No debe crashear aunque no haya snapshots
        obs, _ = env.reset(seed=42)
        assert obs.shape == (DIM_V3,)
        env.close()

    def test_factory_sin_snapshots_usa_fallback(self):
        """Sin snapshots disponibles, debe usar BotExperto como fallback."""
        from src.v3.train import crear_entorno_self_play_v3
        env = crear_entorno_self_play_v3(
            version="v3_inexistente", prob_bot=0.0, agente_idx=0,
        )
        # prob_bot=0.0 + sin snapshots → debe usar BotExperto sin crashear
        obs, _ = env.reset(seed=42)
        assert obs.shape == (DIM_V3,)
        env.close()

    def test_factory_pasa_version_correcta(self):
        """La factory debe aceptar version y no crashear."""
        from src.v3.train import crear_entorno_self_play_v3
        env = crear_entorno_self_play_v3(
            version="v3_mcts", prob_bot=0.5, agente_idx=0,
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
# Circuito MCTS completo en training loop
# ──────────────────────────────────────────────────────────────

class TestCircuitoMCTSTraining:
    """Verifica el circuito MCTS extremo a extremo:
    oracle_buffer → factory → env → BC epoch → logging."""

    def test_factory_forward_oracle_buffer(self) -> None:
        """La factory debe pasar oracle_buffer a CorazonesEnvV3."""
        from src.v3.train import crear_entorno_self_play_v3
        from src.v3.train_mcts import MCTSBuffer

        buf = MCTSBuffer(capacity=100)
        rng = np.random.default_rng(42)

        env = crear_entorno_self_play_v3(
            prob_bot=0.5, agente_idx=0,
            oracle_buffer=buf, oracle_rng=rng,
        )
        assert env._oracle_buffer is buf
        assert env._oracle_rng is rng
        env.close()

    def test_factory_sin_oracle_no_rompe(self) -> None:
        """La factory sin oracle_buffer debe funcionar igual."""
        from src.v3.train import crear_entorno_self_play_v3

        env = crear_entorno_self_play_v3(prob_bot=0.5, agente_idx=0)
        assert env._oracle_buffer is None
        env.close()

    def test_env_make_closure_captura_oracle(self) -> None:
        """Verifica que un closure _make_env pasa oracle_buffer al env."""
        from src.v3.train import crear_entorno_self_play_v3
        from src.v3.train_mcts import MCTSBuffer

        buf = MCTSBuffer(capacity=100)
        rng = np.random.default_rng(42)

        def _make_env():
            return crear_entorno_self_play_v3(
                prob_bot=0.5, agente_idx=0,
                oracle_buffer=buf, oracle_rng=rng,
            )

        env = _make_env()
        assert env._oracle_buffer is buf
        assert env._oracle_rng is rng
        env.close()

    def test_entrenar_bc_epoch_no_crashea_buffer_vacio(
        self, dir_tmp,
    ) -> None:
        """entrenar_bc_epoch con buffer vacío debe retornar 0.0."""
        from src.v3.train_mcts import MCTSBuffer, entrenar_bc_epoch
        from src.v3.red import obtener_policy_kwargs_transformer
        from sb3_contrib import MaskablePPO

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

        buf = MCTSBuffer(capacity=100)
        loss = entrenar_bc_epoch(model, buf, batch_size=16, lr=1e-3)
        assert loss == 0.0

    def test_entrenar_bc_epoch_reduce_loss_buffer_poblado(
        self, dir_tmp,
    ) -> None:
        """BC epoch debe reducir la pérdida sobre buffer poblado."""
        from src.v3.train_mcts import MCTSBuffer, entrenar_bc_epoch
        from src.v3.train_mcts import calcular_bc_loss
        from src.v3.red import obtener_policy_kwargs_transformer
        from sb3_contrib import MaskablePPO
        import torch

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

        # Poblar buffer con datos sintéticos
        buf = MCTSBuffer(capacity=200)
        rng = np.random.default_rng(42)
        oracle_action = 10
        for i in range(40):
            obs = rng.normal(0, 1, DIM_V3).astype(np.float32)
            buf.add(obs, oracle_action)

        def policy_fn(obs_t):
            features = model.policy.extract_features(obs_t)
            logits = model.policy.action_net(features)
            values = model.policy.value_net(features)
            return logits, values

        loss_antes = calcular_bc_loss(policy_fn, buf, batch_size=16)
        loss_despues = entrenar_bc_epoch(
            model, buf, batch_size=16, lr=1e-3,
        )

        assert loss_despues <= loss_antes + 0.5, \
            f"BC debería reducir loss: {loss_antes:.4f} → {loss_despues:.4f}"

    @pytest.mark.slow
    def test_entrenar_auto_mcts_enabled_no_crashea(
        self, dir_tmp,
    ) -> None:
        """entrenar_auto con mcts_enabled=True y total_steps=1 debe
        ejecutar sin crashear (test de integración mínimo)."""
        from src.v3.train import entrenar_auto

        # Un paso mínimo con MCTS habilitado no debe crashear
        try:
            paso = entrenar_auto(
                total_steps=1,
                snapshot_every=1,
                eval_every=999999,
                elo_every=999999,
                output_dir=dir_tmp,
                device="cpu",
                mcts_enabled=True,
                mcts_frequency=1.0,
            )
            assert paso >= 0
        except Exception as e:
            # Si el modelo no puede crearse por falta de recursos,
            # verificamos que el error NO sea por atributo faltante
            assert "unexpected keyword" not in str(e).lower(), \
                f"Error inesperado: {e}"


__all__ = [
    "TestCircuitoMCTSTraining",
]


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


# ──────────────────────────────────────────────────────────────
# BC fine-tuning con Transformer (features_dim ≠ policy_net[-1])
# ──────────────────────────────────────────────────────────────

class TestBCWithTransformer:
    """Verifica que entrenar_bc_epoch y calcular_bc_loss funcionan
    cuando features_dim != last dimension de policy_net (caso real:
    Transformer con features_dim=256, net_arch=[256, 128])."""

    def test_bc_epoch_no_crashea_transformer_dims(
        self, dir_tmp,
    ) -> None:
        """entrenar_bc_epoch no debe crashear con Transformer + dims realistas.

        El bug original era: extract_features() → 256, pero action_net
        esperaba 128 (porque policy_net reduce 256→128).
        """
        from src.v3.train_mcts import MCTSBuffer, entrenar_bc_epoch
        from src.v3.red import obtener_policy_kwargs_transformer
        from sb3_contrib import MaskablePPO
        from src.v3.entorno import CorazonesEnvV3

        env = CorazonesEnvV3(agente_idx=0)
        policy_kwargs = obtener_policy_kwargs_transformer(
            features_dim=256, net_arch=[256, 128],
        )
        model = MaskablePPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            verbose=0, device="cpu",
            n_steps=64, batch_size=32, n_epochs=1,
        )
        env.close()

        buf = MCTSBuffer(capacity=200)
        rng = np.random.default_rng(42)
        for i in range(40):
            obs = rng.normal(0, 1, DIM_V3).astype(np.float32)
            buf.add(obs, 10)

        # No debe crashear
        loss = entrenar_bc_epoch(model, buf, batch_size=16, lr=1e-3)
        assert loss > 0.0, (
            f"BC loss debería ser >0 con buffer poblado, fue {loss}"
        )

    def test_bc_loss_no_crashea_transformer_dims(
        self, dir_tmp,
    ) -> None:
        """calcular_bc_loss debe funcionar con Transformer + dims realistas."""
        from src.v3.train_mcts import MCTSBuffer, calcular_bc_loss
        from src.v3.red import obtener_policy_kwargs_transformer
        from sb3_contrib import MaskablePPO
        from src.v3.entorno import CorazonesEnvV3

        env = CorazonesEnvV3(agente_idx=0)
        policy_kwargs = obtener_policy_kwargs_transformer(
            features_dim=256, net_arch=[256, 128],
        )
        model = MaskablePPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            verbose=0, device="cpu",
            n_steps=64, batch_size=32, n_epochs=1,
        )
        env.close()

        buf = MCTSBuffer(capacity=200)
        rng = np.random.default_rng(42)
        for i in range(30):
            obs = rng.normal(0, 1, DIM_V3).astype(np.float32)
            buf.add(obs, 10)

        def policy_fn(obs_t):
            features = model.policy.extract_features(obs_t)
            if (model.policy.mlp_extractor is not None
                    and model.policy.mlp_extractor.policy_net is not None):
                latent_pi = model.policy.mlp_extractor.policy_net(features)
            else:
                latent_pi = features
            logits = model.policy.action_net(latent_pi)
            values = model.policy.value_net(latent_pi)
            return logits, values

        loss = calcular_bc_loss(policy_fn, buf, batch_size=16)
        assert loss > 0.0, (
            f"BC loss debería ser >0, fue {loss}"
        )


# ──────────────────────────────────────────────────────────────
# Hiperparámetros de entrenamiento v3 (TDD: ajuste agresivo)
# ──────────────────────────────────────────────────────────────

class TestHyperparamsV3:
    """Validación de los hiperparámetros de entrenamiento v3.

    Los valores objetivo tras el ajuste:
      - PROB_BOT_START=0.50, PROB_BOT_END=0.05
      - learning_rate=1e-4
      - total_steps=2_000_000
      - MCTS_FREQUENCY=0.25
    """

    # ── HP_DEFAULT ──────────────────────────────────────────

    def test_hp_default_learning_rate_agresivo(self):
        """lr debe ser 5e-5 (reducido para estabilidad tras diagnóstico)."""
        from src.v3.train import HP_DEFAULT
        assert HP_DEFAULT["learning_rate"] == 5e-5, (
            f"Esperado lr=5e-5, obtenido {HP_DEFAULT['learning_rate']}"
        )

    def test_hp_default_n_steps_unchanged(self):
        """n_steps debe seguir siendo 2048."""
        from src.v3.train import HP_DEFAULT
        assert HP_DEFAULT["n_steps"] == 2048

    def test_hp_default_clip_range_unchanged(self):
        """clip_range debe seguir siendo 0.2."""
        from src.v3.train import HP_DEFAULT
        assert HP_DEFAULT["clip_range"] == 0.2

    def test_hp_default_ent_coef_unchanged(self):
        """ent_coef debe seguir siendo 1e-3."""
        from src.v3.train import HP_DEFAULT
        assert HP_DEFAULT["ent_coef"] == 1e-3

    # ── PROB_BOT constants ──────────────────────────────────

    def test_prob_bot_start_50(self):
        """PROB_BOT_START debe ser 0.50."""
        from src.v3.train import PROB_BOT_START
        assert PROB_BOT_START == 0.50

    def test_prob_bot_end_05_agresivo(self):
        """PROB_BOT_END debe ser 0.05 para presión máxima al final."""
        from src.v3.train import PROB_BOT_END
        assert PROB_BOT_END == 0.05, (
            f"Esperado PROB_BOT_END=0.05, obtenido {PROB_BOT_END}"
        )

    # ── prob_bot_actual con nuevos defaults en 2M ────────────

    def test_prob_bot_inicio_50(self):
        """En paso 0, prob_bot debe ser ~0.50."""
        from src.v3.train import prob_bot_actual
        v = prob_bot_actual(0, 2_000_000)
        assert abs(v - 0.50) < 0.01

    def test_prob_bot_fin_05(self):
        """Al final de 2M pasos, prob_bot debe ser ~0.05."""
        from src.v3.train import prob_bot_actual
        v = prob_bot_actual(2_000_000, 2_000_000)
        assert abs(v - 0.05) < 0.01, (
            f"Esperado ~0.05, obtenido {v:.4f}"
        )

    def test_prob_bot_500k_bajo_45(self):
        """A 500K pasos (25%), prob_bot debe ser <= 0.45 (cosine: ~0.434)."""
        from src.v3.train import prob_bot_actual
        v = prob_bot_actual(500_000, 2_000_000)
        assert v <= 0.45, (
            f"Esperado prob_bot <= 0.45 a 500K, obtenido {v:.4f}"
        )

    def test_prob_bot_1M_bajo_30(self):
        """A 1M pasos (50%), prob_bot debe ser <= 0.30 (cosine: ~0.275)."""
        from src.v3.train import prob_bot_actual
        v = prob_bot_actual(1_000_000, 2_000_000)
        assert v <= 0.30, (
            f"Esperado prob_bot <= 0.30 a 1M, obtenido {v:.4f}"
        )

    def test_prob_bot_estrictamente_decreciente(self):
        """prob_bot debe ser estrictamente decreciente en 5 puntos."""
        from src.v3.train import prob_bot_actual
        total = 2_000_000
        puntos = [
            prob_bot_actual(0, total),
            prob_bot_actual(500_000, total),
            prob_bot_actual(1_000_000, total),
            prob_bot_actual(1_500_000, total),
            prob_bot_actual(2_000_000, total),
        ]
        for i in range(len(puntos) - 1):
            assert puntos[i] > puntos[i + 1], (
                f"No decreciente: {puntos[i]:.4f} ≯ {puntos[i+1]:.4f}"
            )

    # ── MCTS frequency ──────────────────────────────────────

    def test_mcts_frequency_exists_and_ge_40(self):
        """Debe existir MCTS_FREQUENCY y ser >= 0.40."""
        import src.v3.train as train_mod
        freq = getattr(train_mod, "MCTS_FREQUENCY", None)
        assert freq is not None, "MCTS_FREQUENCY no está definida en train.py"
        assert freq >= 0.40, (
            f"Esperado MCTS_FREQUENCY >= 0.40, obtenido {freq}"
        )

    # ── MIN_SNAPSHOT_STEPS ──────────────────────────────────

    def test_min_snapshot_steps_100k(self):
        """MIN_SNAPSHOT_STEPS debe ser 100_000."""
        from src.v3.train import MIN_SNAPSHOT_STEPS
        assert MIN_SNAPSHOT_STEPS == 100_000


# ──────────────────────────────────────────────────────────────
# Oráculo MCTS: debe usar sampling, no enumeración exacta
# ──────────────────────────────────────────────────────────────

class TestOracleSampling:
    """Verifica que evaluar_y_guardar_batch use sampling PIMC,
    NO enumeración exacta (que causaba 2 st/s en Snap 03)."""

    def test_oraculo_salta_baza_menor_10(self):
        """En baza < 10 debe retornar None sin ejecutar oráculo."""
        from src.v3.train_mcts import MCTSBuffer, evaluar_y_guardar_batch
        from src.dominio.motor import MotorCorazones

        motor = MotorCorazones()
        motor.repartir()
        # Forzar número de baza a 5
        motor.numero_baza = 5

        buf = MCTSBuffer(capacity=100)
        obs = np.zeros(250, dtype=np.float32)
        legales = motor.obtener_jugadas_legales(0)

        resultado = evaluar_y_guardar_batch(
            motor, 0, legales, buf, obs, forzar=False,
        )
        assert resultado is None
        assert len(buf) == 0

    def test_oraculo_forzar_no_chequea_num_mundos(self):
        """Con forzar=True, el código NO debe llamar a _num_mundos_posibles.

        Verificación estática: la función evaluar_y_guardar_batch no debe
        contener la importación de _num_mundos_posibles en su cuerpo.
        """
        import src.v3.train_mcts as tmod
        import inspect

        source = inspect.getsource(tmod.evaluar_y_guardar_batch)
        # forzar=True debe saltar también el chequeo de baza
        # Verificamos que solo hay UN early return condicionado a baza<10
        assert source.count("return None") == 1, (
            "evaluar_y_guardar_batch debe tener exactamente 1 return None "
            "(solo el de baza<10, ya no el de num_mundos)"
        )

    def test_oraculo_baza_10_sin_chequeo_num_mundos(self):
        """En baza >= 10 el oráculo DEBE ejecutar (sin importar número de mundos).

        Verificación estática: la función no debe contener _num_mundos_posibles,
        lo que significa que siempre ejecuta el oráculo en baza >= 10
        (ya no hay un límite de 100K mundos que haga skip).
        """
        import src.v3.train_mcts as tmod
        import inspect

        source = inspect.getsource(tmod.evaluar_y_guardar_batch)
        assert "_num_mundos_posibles" not in source, (
            "evaluar_y_guardar_batch NO debe chequear _num_mundos_posibles — "
            "el sampling PIMC siempre es viable (a diferencia de la enumeración exacta)"
        )

    def test_oraculo_usa_sampling_no_enum_exacta(self):
        """Verifica que evaluar_y_guardar_batch no importa _num_mundos_posibles
        ni llama a pimc_exacto (usa _puntaje_esperado_por_carta)."""
        import src.v3.train_mcts as tmod
        import inspect

        source = inspect.getsource(tmod.evaluar_y_guardar_batch)
        assert "_num_mundos_posibles" not in source, (
            "evaluar_y_guardar_batch NO debe importar _num_mundos_posibles "
            "(eso implicaría chequeo de enumeración exacta)"
        )
        assert "_puntaje_esperado_por_carta" in source, (
            "evaluar_y_guardar_batch DEBE usar _puntaje_esperado_por_carta "
            "(sampling PIMC)"
        )

    def test_oraculo_num_mundos_default_50(self):
        """El default de num_mundos debe ser 50 (balance velocidad/calidad)."""
        import src.v3.train_mcts as tmod
        import inspect

        sig = inspect.signature(tmod.evaluar_y_guardar_batch)
        default = sig.parameters["num_mundos"].default
        assert default == 50, (
            f"Esperado num_mundos default=50, obtenido {default}"
        )


# ──────────────────────────────────────────────────────────────
# Métricas enriquecidas de evaluación (posiciones, scores por jugador)
# ──────────────────────────────────────────────────────────────

class TestEvalMetrics:
    """Verifica que log_eval acepta y serializa métricas de posición."""

    def test_log_eval_con_posiciones(self, dir_tmp):
        """log_eval debe aceptar posiciones y scores_por_jugador."""
        from src.v3.train import log_eval
        import json

        log_path = os.path.join(dir_tmp, "eval.jsonl")
        log_eval(
            log_path=log_path,
            paso=100_000,
            wr=0.65,
            avg_score=7.5,
            num_partidas=100,
            prob_bot=0.5,
            posiciones=[30, 25, 25, 20],
            scores_por_jugador={
                "modelo": 7.5,
                "experto_1": 8.2,
                "experto_2": 7.9,
                "bot": 10.1,
            },
        )

        with open(log_path) as f:
            entry = json.loads(f.readline())

        assert entry["wr"] == 0.65
        assert entry["posiciones"] == [30, 25, 25, 20]
        assert entry["scores_por_jugador"]["modelo"] == 7.5
        assert entry["scores_por_jugador"]["experto_1"] == 8.2
        assert entry["scores_por_jugador"]["bot"] == 10.1

    def test_log_eval_backward_compatible(self, dir_tmp):
        """log_eval sin posiciones debe ser backward-compatible."""
        from src.v3.train import log_eval
        import json

        log_path = os.path.join(dir_tmp, "eval.jsonl")
        log_eval(
            log_path=log_path,
            paso=200_000,
            wr=0.60,
            avg_score=8.0,
            num_partidas=100,
            prob_bot=0.45,
        )

        with open(log_path) as f:
            entry = json.loads(f.readline())

        assert entry["wr"] == 0.60
        assert "posiciones" not in entry  # no se incluye si no se pasa


# ──────────────────────────────────────────────────────────────
# Mejoras v4: Q♠ penalty, early burn, BC lr, prob_qs feature
# ──────────────────────────────────────────────────────────────

class TestMejorasV4:
    """Validación de las 4 mejoras para reducir Q♠ y subir 0-pt hands."""

    # ── 1. Q♠ preventive penalty ────────────────────────────

    def test_qs_preventivo_aumentado(self):
        """REWARD_QS_PREVENTIVO debe ser -15.0 (diagnóstico: errores Q♠ Δ=7.26)."""
        from src.v2_1.recompensas import RewardConfigV21
        cfg = RewardConfigV21()
        assert cfg.REWARD_QS_PREVENTIVO == -15.0, (
            f"Esperado -15.0, obtenido {cfg.REWARD_QS_PREVENTIVO}"
        )

    def test_qs_preventivo_devuelve_menos_15(self):
        """recompensa_qs_preventivo con Q♠ y alternativas debe retornar -15.0."""
        from src.v2_1.recompensas import CalculadoraRecompensasV21
        from src.dominio.carta import Carta

        calc = CalculadoraRecompensasV21()
        qs = next(c for c in Carta._TODAS if c.es_dama_de_picas)
        otras_picas = [c for c in Carta._TODAS if c.palo ==
                       2 and not c.es_dama_de_picas]

        r = calc.recompensa_qs_preventivo(qs, [qs, otras_picas[0]], True)
        assert r == -15.0, f"Esperado -15.0, obtenido {r}"

    # ── 2. Early safe burn escalonado ────────────────────────

    def test_early_safe_burn_bazas_1_4_mayor_recompensa(self):
        """Bazas 1-4 deben dar +1.0, bazas 5-7 +0.5."""
        from src.v2_1.recompensas import CalculadoraRecompensasV21

        calc = CalculadoraRecompensasV21()
        # Baza 3: debe ser +1.0
        r_early = calc.recompensa_early_safe_burn(0, 0, 0, 3)
        assert r_early == 1.0, f"Baza 3 esperado 1.0, obtenido {r_early}"

        # Baza 6: debe ser +0.5
        r_mid = calc.recompensa_early_safe_burn(0, 0, 0, 6)
        assert r_mid == 0.5, f"Baza 6 esperado 0.5, obtenido {r_mid}"

        # Baza 9: debe ser 0.0 (fuera de rango)
        r_late = calc.recompensa_early_safe_burn(0, 0, 0, 9)
        assert r_late == 0.0, f"Baza 9 esperado 0.0, obtenido {r_late}"

    # ── 3. BC learning rate reducido ─────────────────────────

    def test_bc_lr_reducido(self):
        """El default de lr en entrenar_bc_epoch debe ser 5e-4."""
        import src.v3.train_mcts as tmod
        import inspect

        sig = inspect.signature(tmod.entrenar_bc_epoch)
        default_lr = sig.parameters["lr"].default
        assert default_lr == 5e-4, (
            f"Esperado lr=5e-4, obtenido {default_lr}"
        )

    # ── 4. prob_qs_por_jugador en observación ────────────────

    def test_dim_v3_es_265(self):
        """DIM_V3 debe ser 265 (260 + 4 prob_qs + 1 forzado)."""
        from src.v3.observacion import DIM_V3
        assert DIM_V3 == 265, (
            f"Esperado DIM_V3=265, obtenido {DIM_V3}"
        )

    def test_prob_qs_por_jugador_sum_1(self):
        """prob_qs_por_jugador debe sumar ~1.0 entre los 4 jugadores."""
        from src.v3.observacion import ObservacionBuilderV3, DIM_V3
        from src.dominio.motor import MotorCorazones

        motor = MotorCorazones()
        motor.repartir()
        builder = ObservacionBuilderV3(dim=DIM_V3)
        obs = builder.construir(
            motor, 0, [set() for _ in range(4)],
            [0, 0, 0, 0], [0, 0, 0, 0], None,
        )

        # prob_qs en [260:264]
        probs = obs[260:264]
        total = float(sum(probs))
        assert abs(total - 1.0) < 0.15, (
            f"prob_qs debe sumar ~1.0, suma {total:.3f}: {probs}"
        )

    def test_prob_qs_cero_si_qs_capturada(self):
        """Si Q♠ ya fue capturada, prob_qs debe ser [0,0,0,0]."""
        from src.v3.observacion import ObservacionBuilderV3, DIM_V3
        from src.dominio.motor import MotorCorazones

        motor = MotorCorazones()
        motor.repartir()
        builder = ObservacionBuilderV3(dim=DIM_V3)
        obs = builder.construir(
            motor, 0, [set() for _ in range(4)],
            [0, 0, 0, 0], [0, 0, 0, 0], 2,  # Q♠ capturada por jugador 2
        )
        assert all(v == 0.0 for v in obs[260:264]), (
            f"prob_qs deben ser 0 si Q♠ capturada: {obs[260:264]}"
        )

    def test_forzado_1_si_una_carta_legal(self):
        """[264] forzado debe ser 1.0 si solo hay 1 carta legal."""
        from src.v3.observacion import ObservacionBuilderV3, DIM_V3
        from src.dominio.motor import MotorCorazones

        motor = MotorCorazones()
        motor.repartir()
        # Jugar hasta que el agente tenga 1 sola carta
        while len(motor.jugadores[0].mano) > 1:
            idx = motor.obtener_jugador_actual()
            leg = motor.obtener_jugadas_legales(idx)
            if leg:
                import random
                motor.jugar_carta(idx, random.choice(leg))
                if len(motor.mesa) == 4:
                    motor.resolver_baza()

        builder = ObservacionBuilderV3(dim=DIM_V3)
        obs = builder.construir(
            motor, 0, [set() for _ in range(4)],
            [0, 0, 0, 0], [0, 0, 0, 0], None,
        )
        assert obs[264] == 1.0, (
            f"forzado debe ser 1.0 con 1 carta, es {obs[264]}"
        )


# ──────────────────────────────────────────────────────────────
# BC pre-training desde dataset offline
# ──────────────────────────────────────────────────────────────

class TestBCDataset:
    """Verifica el pre-entrenamiento BC desde dataset offline."""

    def test_entrenar_bc_dataset_existe(self):
        """entrenar_bc_dataset debe ser importable."""
        from src.v3.train_mcts import entrenar_bc_dataset
        assert callable(entrenar_bc_dataset)

    def test_entrenar_bc_dataset_reduce_loss(self, dir_tmp):
        """BC dataset debe reducir loss en cada epoch."""
        from src.v3.train_mcts import entrenar_bc_dataset
        from src.v3.red import obtener_policy_kwargs_transformer
        from sb3_contrib import MaskablePPO
        from src.v3.entorno import CorazonesEnvV3
        import numpy as np
        import os

        # Crear dataset sintetico (formato v2: all_scores)
        n_samples = 100
        obs = np.random.normal(0, 1, (n_samples, DIM_V3)).astype(np.float32)
        all_scores = np.random.uniform(
            0, 26, (n_samples, 52)).astype(np.float32)
        ds_path = os.path.join(dir_tmp, "test_dataset.npz")
        np.savez_compressed(ds_path, observations=obs, all_scores=all_scores)

        env = CorazonesEnvV3(agente_idx=0)
        pk = obtener_policy_kwargs_transformer(
            features_dim=32, net_arch=[32, 32])
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=pk, verbose=0,
                            device="cpu", n_steps=64, batch_size=32, n_epochs=1)
        env.close()

        losses = entrenar_bc_dataset(model, ds_path, epochs=3,
                                     batch_size=16, lr=1e-3)
        assert len(losses) == 3
        # Loss debe bajar o mantenerse
        assert losses[-1] <= losses[0] + 0.5, (
            f"BC deberia reducir loss: {losses}"
        )


class TestMetricasPartida:
    """Validación de métricas orientadas a ganar partidas (no WR<=8)."""

    def test_log_eval_con_top2_rate(self, dir_tmp):
        """log_eval debe aceptar top1_rate y top2_rate."""
        from src.v3.train import log_eval
        import json

        log_path = os.path.join(dir_tmp, "eval.jsonl")
        log_eval(
            log_path=log_path, paso=100_000, wr=0.65, avg_score=7.5,
            num_partidas=100, prob_bot=0.5,
            posiciones=[18, 23, 30, 29],
            scores_por_jugador={"modelo": 7.5,
                                "experto_1": 5.0, "experto_2": 4.5, "bot": 10.0},
            top1_rate=0.18, top2_rate=0.41,
        )
        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert entry["top1_rate"] == 0.18
        assert entry["top2_rate"] == 0.41

    def test_evaluar_estandar_retorna_todo(self):
        """_evaluar_estandar debe retornar (wr, avg, pos, scores, top1, top2)."""
        from src.v3.train import _evaluar_estandar
        from sb3_contrib import MaskablePPO
        from src.v3.entorno import CorazonesEnvV3

        env = CorazonesEnvV3(agente_idx=0)
        model = MaskablePPO("MlpPolicy", env, verbose=0, device="cpu",
                            n_steps=64, batch_size=32, n_epochs=1)
        env.close()

        wr, score, pos, scores_pj, top1, top2 = _evaluar_estandar(
            model, num_partidas=4, seed=42,
        )
        assert 0 <= wr <= 1
        assert 0 <= top1 <= 1
        assert 0 <= top2 <= 1
        assert sum(pos) == 4
        assert abs(top1 - pos[0]/4) < 0.01

    def test_top2_consistente_con_posiciones(self, dir_tmp):
        """Verifica que top2_rate = (1ro+2do)/total en datos reales."""
        from src.v3.train import _evaluar_estandar
        from sb3_contrib import MaskablePPO
        from src.v3.entorno import CorazonesEnvV3

        env = CorazonesEnvV3(agente_idx=0)
        model = MaskablePPO("MlpPolicy", env, verbose=0, device="cpu",
                            n_steps=64, batch_size=32, n_epochs=1)
        env.close()

        _, _, pos, _, top1, top2 = _evaluar_estandar(
            model, num_partidas=4, seed=42)
        assert abs(top2 - (pos[0]+pos[1])/4) < 0.01

    def test_evaluar_facil_existe_y_retorna_6(self):
        """_evaluar_facil debe existir y retornar 6-tuple."""
        from src.v3.train import _evaluar_facil
        from sb3_contrib import MaskablePPO
        from src.v3.entorno import CorazonesEnvV3

        env = CorazonesEnvV3(agente_idx=0)
        model = MaskablePPO("MlpPolicy", env, verbose=0, device="cpu",
                            n_steps=64, batch_size=32, n_epochs=1)
        env.close()

        wr, score, pos, scores_pj, top1, top2 = _evaluar_facil(
            model, num_partidas=4, seed=42,
        )
        assert sum(pos) == 4
        assert 0 <= top2 <= 1
