"""
Pool de oponentes para self-play en Hearts.

Gestiona la selección de oponentes según la fase de entrenamiento:
  Fase 0  (0–5%):  3 bots heurísticos simples
  Fase 1  (5–15%): 2 bots + 1 BotExperto
  Fase 2 (15–40%): 1 BotExperto + snapshots históricos
  Fase 3 (40–70%): mayoría snapshots + 1 BotExperto
  Fase 4 (70–100%): snapshots puros

Los snapshots son callables cargados desde weights de políticas RLlib.
"""
from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Optional

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
from src.agentes.bot_experto import BotExperto
from src.agentes.bot_castigador import BotCastigador
from src.entorno.dimensiones import DIM_ENTORNO
from src.entorno.observacion import ObservacionBuilder

PolicyFn = Callable[[MotorCorazones, int, List[Carta]], Carta]

_BOTS_SIMPLES: List[PolicyFn] = [bot_conservador, bot_agresivo, bot_evasivo]


def _select_opponent(progress: float, snapshots: list) -> PolicyFn:
    """Selecciona un oponente según la fase de entrenamiento. Función de módulo (picklable)."""
    if progress < 0.05 or not snapshots:
        return random.choice(_BOTS_SIMPLES)
    elif progress < 0.15:
        return BotExperto() if random.random() < 0.5 else random.choice(_BOTS_SIMPLES)
    elif progress < 0.40:
        return random.choice(snapshots) if snapshots and random.random() < 0.5 else BotExperto()
    elif progress < 0.70:
        return random.choice(snapshots) if snapshots and random.random() < 0.7 else BotExperto()
    else:
        return random.choice(snapshots) if snapshots else BotExperto()


class SnapshotPolicy:
    """Wrapper serializable que adapta pesos de política a (motor, idx, legales) -> Carta.

    Almacena los pesos como dict de numpy arrays (picklable) en lugar del objeto
    Policy de Ray (que contiene _thread.RLock y no es serializable entre workers).
    El modelo PyTorch se reconstruye de forma lazy en cada worker.
    """

    def __init__(self, policy, obs_dim: int = DIM_ENTORNO):
        # Extraer pesos como numpy arrays (siempre serializables)
        self._weights: dict = policy.get_weights()
        self._obs_dim = obs_dim
        self._obs_builder = ObservacionBuilder(dim=obs_dim)
        self._model = None  # reconstruido lazy en cada proceso

    @classmethod
    def from_weights(cls, weights: dict, obs_dim: int = DIM_ENTORNO) -> "SnapshotPolicy":
        """Crea un SnapshotPolicy directamente desde un dict de pesos numpy."""
        obj = object.__new__(cls)
        obj._weights = weights
        obj._obs_dim = obs_dim
        obj._obs_builder = ObservacionBuilder(dim=obs_dim)
        obj._model = None
        return obj

    # ---- pickle support: solo guardar pesos + dim, nunca el modelo torch ----
    def __getstate__(self):
        return {"weights": self._weights, "obs_dim": self._obs_dim}

    def __setstate__(self, state):
        self._weights = state["weights"]
        self._obs_dim = state["obs_dim"]
        self._obs_builder = ObservacionBuilder(dim=self._obs_dim)
        self._model = None

    def _get_model(self):
        """Construye el modelo PyTorch desde los pesos la primera vez (lazy)."""
        if self._model is not None:
            return self._model

        import torch
        from src.rllib.model import HeartsActionMaskModel
        from gymnasium import spaces

        obs_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(52,), dtype=np.float32),
        })
        action_space = spaces.Discrete(52)
        model_config = {"fcnet_hiddens": [512, 512, 256], "fcnet_activation": "relu", "vf_share_layers": False}

        model = HeartsActionMaskModel(obs_space, action_space, 52, model_config, "snapshot")

        # Los pesos desde policy.get_weights() o policy_state.pkl usan nombres PyTorch exactos
        torch_state = {k: torch.tensor(v) for k, v in self._weights.items()}
        model.load_state_dict(torch_state, strict=True)
        model.eval()
        self._model = model
        return model

    def __call__(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
    ) -> Carta:
        import torch

        obs_vec = self._obs_builder.construir_desde_motor(motor, jugador_idx=idx)
        mask = np.zeros(52, dtype=np.float32)
        for c in legales:
            mask[c.id] = 1.0

        model = self._get_model()
        with torch.no_grad():
            obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            input_dict = {"obs": {"obs": obs_t, "action_mask": mask_t}}
            logits, _ = model.forward(input_dict, [], None)
            action = int(logits.argmax(dim=1).item())

        carta = Carta._TODAS[action]
        if carta not in legales:
            carta = random.choice(legales)
        return carta


class OpponentPool:
    """Pool de oponentes para self-play con bots y snapshots históricos.

    Uso:
        pool = OpponentPool(agente_idx=0, snapshot_dir="models/v_rllib/snapshots")
        env_config = {"opponent_factory": pool.make_factory(progress=0.3)}
    """

    def __init__(
        self,
        agente_idx: int = 0,
        snapshot_dir: Optional[str] = None,
        max_snapshots: int = 50,
        obs_dim: int = DIM_ENTORNO,
    ):
        self._agente_idx = agente_idx
        self._snapshot_dir = snapshot_dir
        self._max_snapshots = max_snapshots
        self._obs_dim = obs_dim

        self._snapshots: List[SnapshotPolicy] = []
        self._bot_experto_pool: List[PolicyFn] = []

    def add_snapshot(self, policy) -> None:
        """Añade una nueva política snapshot al pool (FIFO si excede el máximo)."""
        snap = SnapshotPolicy(policy, obs_dim=self._obs_dim)
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def make_factory(self, progress: float) -> Callable[[], Dict[int, PolicyFn]]:
        """Devuelve un opponent_factory serializable para la fase actual.

        El closure NO captura self — solo datos picklables — para que pueda
        enviarse a workers remotos de Ray sin errores de serialización.

        Args:
            progress: fracción de entrenamiento completada (0.0 – 1.0).
        """
        opp_indices = [i for i in range(4) if i != self._agente_idx]
        snapshots = list(self._snapshots)  # copia picklable en este momento

        def _factory() -> Dict[int, PolicyFn]:
            fns: Dict[int, PolicyFn] = {}
            for idx in opp_indices:
                fns[idx] = _select_opponent(progress, snapshots)
            return fns

        return _factory

