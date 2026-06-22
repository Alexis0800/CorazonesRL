"""
Tests unitarios para el Transformer Feature Extractor de v3.

Verifica:
  - TransformerFeatureExtractor produce shapes correctos
  - LSTM policy kwargs son compatibles con MaskablePPO/RecurrentPPO
  - La arquitectura maneja batch y secuencia correctamente
"""

from __future__ import annotations

import numpy as np
import pytest

import torch
import gymnasium as gym
from gymnasium import spaces

from src.v3.red import (
    TransformerFeatureExtractor,
    obtener_policy_kwargs_transformer,
    obtener_policy_kwargs_lstm,
    obtener_policy_kwargs_mlp,
)


class TestTransformerFeatureExtractor:
    """Suite de tests para el extractor Transformer de v3."""

    @pytest.fixture
    def obs_space_260(self) -> spaces.Box:
        """Observation space con 260 dims (v3)."""
        return spaces.Box(low=0.0, high=26.0, shape=(260,), dtype=np.float32)

    @pytest.fixture
    def obs_space_220(self) -> spaces.Box:
        """Observation space con 220 dims (v2 compat)."""
        return spaces.Box(low=0.0, high=1.0, shape=(220,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Tests de dimensionalidad
    # ------------------------------------------------------------------

    def test_output_shape(self, obs_space_260) -> None:
        """El output debe tener features_dim (256)."""
        extractor = TransformerFeatureExtractor(
            obs_space_260, features_dim=256)
        x = torch.randn(4, 260)  # batch de 4
        out = extractor(x)
        assert out.shape == (
            4, 256), f"Esperado (4, 256), obtenido {out.shape}"

    def test_output_shape_custom_dim(self, obs_space_260) -> None:
        """Debe respetar features_dim personalizado."""
        extractor = TransformerFeatureExtractor(
            obs_space_260, features_dim=128)
        x = torch.randn(2, 260)
        out = extractor(x)
        assert out.shape == (2, 128)

    def test_batch_independence(self, obs_space_260) -> None:
        """Diferentes batches deben producir diferentes outputs."""
        extractor = TransformerFeatureExtractor(
            obs_space_260, features_dim=256)
        x1 = torch.randn(1, 260)
        x2 = torch.randn(1, 260)
        out1 = extractor(x1)
        out2 = extractor(x2)
        assert not torch.allclose(out1, out2)

    def test_deterministic_eval_mode(self, obs_space_260) -> None:
        """En modo eval, mismo input → mismo output."""
        extractor = TransformerFeatureExtractor(
            obs_space_260, features_dim=256)
        extractor.eval()
        x = torch.randn(1, 260)
        out1 = extractor(x)
        out2 = extractor(x)
        assert torch.allclose(out1, out2)

    def test_compatible_with_220_dim(self, obs_space_220) -> None:
        """Debe aceptar 220 dims (compatibilidad hacia atrás)."""
        extractor = TransformerFeatureExtractor(
            obs_space_220, features_dim=256)
        x = torch.randn(4, 220)
        out = extractor(x)
        assert out.shape == (4, 256)

    def test_gradient_flow(self, obs_space_260) -> None:
        """Los gradientes deben fluir a través del extractor."""
        extractor = TransformerFeatureExtractor(
            obs_space_260, features_dim=256)
        x = torch.randn(4, 260, requires_grad=True)
        out = extractor(x)
        loss = out.sum()
        loss.backward()
        # Verificar que hay gradientes en los parámetros
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in extractor.parameters()
        )
        assert has_grad, "No hay gradientes fluyendo"


class TestPolicyKwargs:
    """Suite de tests para las factorías de policy_kwargs."""

    def test_transformer_kwargs_structure(self) -> None:
        """obtener_policy_kwargs_transformer debe retornar dict con keys esperadas."""
        kwargs = obtener_policy_kwargs_transformer(features_dim=256)
        assert "features_extractor_class" in kwargs
        assert kwargs["features_extractor_class"] == TransformerFeatureExtractor
        assert "features_extractor_kwargs" in kwargs
        assert kwargs["features_extractor_kwargs"]["features_dim"] == 256
        assert "net_arch" in kwargs

    def test_lstm_kwargs_structure(self) -> None:
        """obtener_policy_kwargs_lstm debe configurar LSTM correctamente."""
        kwargs = obtener_policy_kwargs_lstm(
            features_dim=256, lstm_hidden_size=256, n_lstm_layers=2
        )
        assert "features_extractor_class" in kwargs
        assert "features_extractor_kwargs" in kwargs
        assert kwargs["features_extractor_kwargs"]["features_dim"] == 256
        # Las capas LSTM se configuran en net_arch
        assert "net_arch" in kwargs
        # Debe tener pi y vf
        net = kwargs["net_arch"]
        assert "pi" in net or isinstance(net, list)

    def test_mlp_kwargs_structure(self) -> None:
        """obtener_policy_kwargs_mlp debe ser compatible con MaskablePPO."""
        kwargs = obtener_policy_kwargs_mlp(features_dim=256)
        assert "features_extractor_class" in kwargs
        assert "features_extractor_kwargs" in kwargs
        assert "net_arch" in kwargs

    def test_transformer_defaults(self) -> None:
        """Valores por defecto razonables."""
        kwargs = obtener_policy_kwargs_transformer()
        assert kwargs["features_extractor_kwargs"]["features_dim"] == 256

    def test_lstm_defaults(self) -> None:
        """Valores por defecto razonables."""
        kwargs = obtener_policy_kwargs_lstm()
        assert kwargs["features_extractor_kwargs"]["features_dim"] == 256

    def test_mlp_defaults(self) -> None:
        """Valores por defecto razonables."""
        kwargs = obtener_policy_kwargs_mlp()
        assert kwargs["features_extractor_kwargs"]["features_dim"] == 256
