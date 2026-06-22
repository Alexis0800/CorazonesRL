"""
Red neuronal para v3 — Transformer Feature Extractor + LSTM/MLP factories.

Proporciona tres arquitecturas alternativas para el feature extractor:

  1. **TransformerFeatureExtractor**: Divide la observación en grupos (tokens)
     y aplica self-attention entre ellos. Captura relaciones entre bloques
     de features sin requerir secuencia temporal explícita.

  2. **LSTM (RecurrentPPO)**: Usa sb3-contrib RecurrentPPO con política LSTM.
     La memoria persiste entre steps dentro del episodio, resolviendo
     directamente el problema de tracking entre bazas.

  3. **MLP (v2 compat)**: MLP clásico [512,512,256] para comparación A/B.

Basado en el diagnóstico PIMC (45% error rate, causas: planning 33%,
risk_assessment 46%, card_counting 17%), la arquitectura Transformer+LSTM
es la recomendada para v3.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Extractor de características basado en Transformer para v3.

    Divide el vector de observación en N tokens (grupos de features)
    y aplica self-attention entre ellos para capturar relaciones
    inter-bloque (ej. relación entre mi mano y el cementerio).

    Arquitectura:
        obs(250) → Split en 5 tokens de 50-d
                 → Linear(50→128) cada token
                 → + PositionalEncoding
                 → TransformerEncoder(2 layers, 4 heads, d=128)
                 → Mean Pooling
                 → Linear(128→features_dim) + ReLU

    Ventajas sobre MLP:
    - Self-attention captura relaciones entre bloques de features
    - Positional encoding da estructura al espacio de observación
    - Mejor generalización: atención selectiva a features relevantes
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_tokens: int = 5,
    ) -> None:
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])

        self.n_tokens = n_tokens
        self.d_model = d_model
        self.token_size = input_dim // n_tokens
        self._input_dim = input_dim

        # Proyección de cada token
        self.token_proj = nn.ModuleList([
            nn.Linear(self.token_size, d_model)
            for _ in range(n_tokens)
        ])

        # Padding projection para features sobrantes (si input_dim % n_tokens != 0)
        remainder = input_dim % n_tokens
        if remainder > 0:
            self.remainder_proj = nn.Linear(remainder, d_model)
        else:
            self.remainder_proj = None

        # Positional encoding aprendible
        n_effective_tokens = n_tokens + (1 if remainder > 0 else 0)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_effective_tokens, d_model) * 0.02
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Capa de salida
        self.output = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Procesa un batch de observaciones a través del Transformer.

        Args:
            observations: Tensor (batch, input_dim).

        Returns:
            Tensor (batch, features_dim).
        """
        batch_size = observations.shape[0]

        # Dividir en tokens
        tokens = []
        for i in range(self.n_tokens):
            start = i * self.token_size
            end = start + self.token_size
            token = self.token_proj[i](observations[:, start:end])
            tokens.append(token)

        # Token extra para features sobrantes
        if self.remainder_proj is not None:
            remainder_start = self.n_tokens * self.token_size
            rem = observations[:, remainder_start:]
            tokens.append(self.remainder_proj(rem))

        # Stack: (batch, n_tokens, d_model)
        x = torch.stack(tokens, dim=1)

        # Añadir positional encoding
        x = x + self.pos_encoding[:, :x.shape[1], :]

        # Transformer
        x = self.transformer(x)

        # Pooling (mean sobre tokens)
        x = x.mean(dim=1)

        return self.output(x)


# ------------------------------------------------------------------
# Policy kwargs factories
# ------------------------------------------------------------------


def obtener_policy_kwargs_transformer(
    features_dim: int = 256,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    net_arch: Optional[Sequence[int]] = None,
) -> Dict:
    """Policy kwargs para MaskablePPO con TransformerFeatureExtractor.

    Args:
        features_dim: Dimensión de salida del extractor.
        d_model: Dimensión del embedding del Transformer.
        n_heads: Cabezas de atención.
        n_layers: Capas del encoder.
        net_arch: Arquitectura de las redes pi/vf.

    Returns:
        Dict para `MaskablePPO(policy_kwargs=...)`.
    """
    if net_arch is None:
        net_arch = [256, 128]

    return {
        "features_extractor_class": TransformerFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
        },
        "net_arch": {
            "pi": list(net_arch),
            "vf": list(net_arch),
        },
    }


def obtener_policy_kwargs_lstm(
    features_dim: int = 256,
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 2,
    net_arch: Optional[Sequence[int]] = None,
) -> Dict:
    """Policy kwargs para RecurrentPPO con LSTM.

    La memoria LSTM persiste entre steps dentro del episodio,
    permitiendo al agente trackear fallos, cartas jugadas, y Q♠
    a través de múltiples bazas.

    Args:
        features_dim: Dimensión de salida del extractor base.
        lstm_hidden_size: Tamaño del estado oculto LSTM.
        n_lstm_layers: Número de capas LSTM.
        net_arch: Arquitectura de las redes pi/vf.

    Returns:
        Dict para `RecurrentPPO(policy_kwargs=...)`.
    """
    if net_arch is None:
        net_arch = [256, 128]

    return {
        "features_extractor_class": TransformerFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
        },
        "net_arch": {
            "pi": list(net_arch),
            "vf": list(net_arch),
        },
        # LSTM config — usado por RecurrentActorCriticPolicy
        "lstm_hidden_size": lstm_hidden_size,
        "n_lstm_layers": n_lstm_layers,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }


def obtener_policy_kwargs_mlp(
    features_dim: int = 256,
    net_arch: Optional[Sequence[int]] = None,
) -> Dict:
    """Policy kwargs para MaskablePPO con MLP clásico (v2 compat).

    Útil como baseline para comparación A/B con Transformer y LSTM.

    Args:
        features_dim: Dimensión de salida del extractor.
        net_arch: Arquitectura de las redes pi/vf.

    Returns:
        Dict para `MaskablePPO(policy_kwargs=...)`.
    """
    from src.red import CorazonesFeatureExtractor

    if net_arch is None:
        net_arch = [512, 512, 256]

    return {
        "features_extractor_class": CorazonesFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": {
            "pi": list(net_arch),
            "vf": list(net_arch),
        },
    }


__all__ = [
    "TransformerFeatureExtractor",
    "obtener_policy_kwargs_transformer",
    "obtener_policy_kwargs_lstm",
    "obtener_policy_kwargs_mlp",
]
