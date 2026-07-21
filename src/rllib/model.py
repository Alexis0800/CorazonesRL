"""
Modelo PyTorch con action masking para Ray RLlib (old API stack).

Arquitectura: MLP encoder → policy head + value head.
La máscara de acciones se aplica añadiendo -1e9 a las acciones ilegales
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

import torch
import torch.nn as nn
from typing import List, Tuple

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class HeartsActionMaskModel(TorchModelV2, nn.Module):
    """MLP con action masking nativo para Hearts.

    La action_mask se obtiene de input_dict["obs"]["action_mask"].
    Las acciones ilegales reciben logit = -1e9 antes de la política.
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


class HeartsLSTMModel(RecurrentNetwork, nn.Module):
    """LSTM con action masking para Hearts (RLlib old API stack).

    Arquitectura: MLP encoder (obs_dim→256→256) → LSTM (256→lstm_hidden) → heads.
    El estado oculto persiste entre steps de la misma mano (13 tricks), lo que
    permite trackear patrones de oponentes dinámicamente dentro del episodio.

    Uso (en PPOConfig):
        ModelCatalog.register_custom_model("hearts_lstm_model", HeartsLSTMModel)
        config.training(model={"custom_model": "hearts_lstm_model",
                               "lstm_cell_size": 256, "max_seq_len": 13})
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        obs_dim: int = obs_space["obs"].shape[0]
        lstm_hidden: int = model_config.get("lstm_cell_size", 256)
        self._lstm_hidden = lstm_hidden
        self._cur_value: torch.Tensor | None = None

        # MLP encoder antes del LSTM
        self._encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self._lstm_layer = nn.LSTM(256, lstm_hidden, batch_first=True)
        self._policy_head = nn.Linear(lstm_hidden, num_outputs)
        self._value_head = nn.Linear(lstm_hidden, 1)

    @override(ModelV2)
    def get_initial_state(self) -> List[torch.Tensor]:
        h = self._policy_head.weight.new(self._lstm_hidden).zero_()
        c = self._policy_head.weight.new(self._lstm_hidden).zero_()
        return [h, c]

    @override(RecurrentNetwork)
    def forward(
        self,
        input_dict: dict,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"]["obs"].float()            # [B*T, obs_dim]
        action_mask = input_dict["obs"]["action_mask"].float()  # [B*T, 52]

        # B = batch (num secuencias), T = pasos por secuencia (max_seq_len)
        B = state[0].shape[0]
        T = obs.shape[0] // B

        encoded = self._encoder(obs.view(B, T, -1))  # [B, T, 256]

        h = state[0].unsqueeze(0).to(obs.device)   # [1, B, lstm_hidden]
        c = state[1].unsqueeze(0).to(obs.device)
        lstm_out, (h_new, c_new) = self._lstm_layer(encoded, (h, c))  # [B, T, lstm_hidden]

        flat = lstm_out.reshape(B * T, self._lstm_hidden)
        self._cur_value = self._value_head(flat).squeeze(1)

        logits = self._policy_head(flat)
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e9)
        return logits + inf_mask, [h_new.squeeze(0), c_new.squeeze(0)]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        raise NotImplementedError("HeartsLSTMModel overrides forward() directly")

    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "forward() no ha sido llamado"
        return self._cur_value
