"""
Red neuronal personalizada para el agente RL de Corazones (Módulo 3).

Define la arquitectura MLP [256, 256, 128] requerida por el Master Plan
como extractor de características compatible con MaskablePPO de sb3-contrib.

Topología:
    Input:  194 neuronas (Vector de Observación)
    Capa 1: 256 neuronas (ReLU)
    Capa 2: 256 neuronas (ReLU)
    Capa 3: 128 neuronas (ReLU)
    Output: features_dim (para Actor-Critic de PPO)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Type

import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CorazonesFeatureExtractor(BaseFeaturesExtractor):
    """Extractor de características MLP para el entorno de Corazones.

    Arquitectura: 194 → 256 → 256 → 128 (sin capa extra final).
    128 neuronas de salida alimentan directamente al Actor-Critic de PPO.

    Diseñada para procesar el vector de observación one-hot de 194
    dimensiones y extraer características relevantes para el Actor
    (selección de carta) y el Crítico (estimación de valor).

    Args:
        observation_space: Espacio de observación Box(194,) del entorno.
        features_dim: Dimensión del vector de características de salida (128).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim)

        input_dim = int(observation_space.shape[0])  # 194

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),  # 256 → 128 directamente
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Propaga un batch de observaciones a través de la red.

        Args:
            observations: Tensor de shape (batch_size, 194).

        Returns:
            Tensor de características de shape (batch_size, 128).
        """
        return self.net(observations)


# ------------------------------------------------------------------
# Configuración de política para MaskablePPO
# ------------------------------------------------------------------

def obtener_policy_kwargs(
    net_arch: Optional[Sequence[int]] = None,
    features_dim: int = 128,
) -> Dict:
    """Retorna los policy_kwargs para MaskablePPO con la arquitectura
    especificada en el Master Plan.

    Args:
        net_arch: Arquitectura de capas ocultas. Por defecto [256, 256, 128].
        features_dim: Dimensión de salida del extractor (128).

    Returns:
        Diccionario con argumentos para el constructor de MaskablePPO.
    """
    if net_arch is None:
        net_arch = [256, 256, 128]

    return {
        "net_arch": list(net_arch),
        "features_extractor_class": CorazonesFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
    }
