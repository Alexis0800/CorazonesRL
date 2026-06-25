"""
Modelo PyTorch con action masking para Ray RLlib (old API stack).

Arquitectura: MLP encoder → policy head + value head.
La máscara de acciones se aplica añadiendo -∞ a las acciones ilegales
antes del softmax, lo que las hace efectivamente imposibles.

Uso (en PPOConfig):
    from ray.rllib.models import ModelCatalog
    from src.rllib.model import HeartsActionMaskModel

    ModelCatalog.register_custom_model("hearts_model", HeartsActionMaskModel)
    config.training(model={"custom_model": "hearts_model", ...})

Observation space esperado:
    Dict({"obs": Box(obs_dim,), "action_mask": Box(52,)})
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class HeartsActionMaskModel(TorchModelV2, nn.Module):
    """MLP con action masking nativo para Hearts.

    La action_mask se obtiene de input_dict["obs"]["action_mask"].
    Las acciones ilegales reciben logit = -∞ antes de la política.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Dimensión del vector de obs (sin incluir action_mask)
        obs_dim: int = obs_space["obs"].shape[0]
        hiddens: List[int] = model_config.get("fcnet_hiddens", [512, 512, 256])

        # MLP encoder
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hiddens:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self._encoder = nn.Sequential(*layers)

        # Cabezas de política y valor
        self._policy_head = nn.Linear(prev, num_outputs)
        self._value_head = nn.Linear(prev, 1)
        self._cur_value: torch.Tensor | None = None

    def forward(
        self,
        input_dict: dict,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"]["obs"].float()
        action_mask = input_dict["obs"]["action_mask"].float()

        features = self._encoder(obs)
        self._cur_value = self._value_head(features).squeeze(1)

        logits = self._policy_head(features)

        # Añadir -inf a acciones ilegales (donde mask == 0)
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e9)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "forward() no ha sido llamado"
        return self._cur_value
