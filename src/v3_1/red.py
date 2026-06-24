"""
Red neuronal para v3.1 — MLP clásico (más estable que Transformer para 228 dims).

El diagnóstico de los 6 runs de v3 mostró que el Transformer con 265 dims
no convergía mejor que el MLP de v2 (explained_variance 0.25-0.50,
KL 0.12-0.18, clip_fraction 0.25-0.35). Con 228 dims depuradas,
un MLP de 3 capas es más estable y rápido.

Arquitectura:
    obs(228) → Linear(228→512) + ReLU
             → Linear(512→256) + ReLU
             → Linear(256→128) + ReLU
             → features_dim=128
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MLPFeatureExtractorV31(BaseFeaturesExtractor):
    """MLP de 3 capas para v3.1 — 228 dims de entrada.

    Más estable y rápido que el Transformer para este espacio
    de features. Usado con MaskablePPO (MlpPolicy).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        net_arch: Optional[Sequence[int]] = None,
    ) -> None:
        if net_arch is None:
            net_arch = [512, 256, 128]
        super().__init__(observation_space, features_dim)

        input_dim = int(observation_space.shape[0])
        layers = []
        prev_dim = input_dim

        for hidden_dim in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Última capa: al features_dim final
        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


def obtener_policy_kwargs_v31(
    features_dim: int = 128,
    net_arch: Optional[Sequence[int]] = None,
) -> Dict:
    """Retorna policy_kwargs para MaskablePPO con MLP de v3.1.

    Args:
        features_dim: Dimensión de salida del feature extractor.
        net_arch: Arquitectura de la MLP (default: [512, 256, 128]).

    Returns:
        Dict con policy_kwargs para SB3.
    """
    if net_arch is None:
        net_arch = [512, 256, 128]

    return {
        "features_extractor_class": MLPFeatureExtractorV31,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "net_arch": list(net_arch),
        },
        "net_arch": [],  # Sin capas adicionales en policy/value
    }
