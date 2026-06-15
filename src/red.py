"""
Red neuronal personalizada para el agente RL de Corazones.

Arquitectura MLP [256, 256, 128] como extractor de características
compatible con MaskablePPO de sb3-contrib.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CorazonesFeatureExtractor(BaseFeaturesExtractor):
    """Extractor de características MLP para el entorno de Corazones.

    Arquitectura: input_dim → 256 → 256 → 128 (ReLU).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def obtener_policy_kwargs(
    net_arch: Optional[Sequence[int]] = None,
    features_dim: int = 128,
) -> Dict:
    """Retorna los policy_kwargs para MaskablePPO."""
    if net_arch is None:
        net_arch = [256, 256, 128]
    return {
        "features_extractor_class": CorazonesFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": {
            "pi": list(net_arch),
            "vf": list(net_arch),
        },
    }


__all__ = ["CorazonesFeatureExtractor", "obtener_policy_kwargs"]
