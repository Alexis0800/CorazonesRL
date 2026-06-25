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


class SnapshotPolicy:
    """Wrapper que adapta un policy RLlib (pesos) a la interfaz (motor, idx, legales) -> Carta."""

    def __init__(self, policy, obs_dim: int = DIM_ENTORNO):
        self._policy = policy
        self._obs_builder = ObservacionBuilder(dim=obs_dim)

    def __call__(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
    ) -> Carta:
        obs_vec = self._obs_builder.construir_desde_motor(motor=motor, agente_idx=idx)
        mask = np.zeros(52, dtype=np.float32)
        for c in legales:
            mask[c.id] = 1.0

        obs_dict = {
            "obs": obs_vec.astype(np.float32),
            "action_mask": mask,
        }
        action = self._policy.compute_single_action(obs_dict, explore=False)[0]
        carta = Carta._TODAS[int(action)]

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
        """Devuelve un opponent_factory para la fase actual de entrenamiento.

        Args:
            progress: fracción de entrenamiento completada (0.0 – 1.0).
        """
        opp_indices = [i for i in range(4) if i != self._agente_idx]
        snapshots = list(self._snapshots)

        def _factory() -> Dict[int, PolicyFn]:
            fns: Dict[int, PolicyFn] = {}
            for idx in opp_indices:
                fns[idx] = self._select_opponent(progress, snapshots)
            return fns

        return _factory

    def _select_opponent(
        self, progress: float, snapshots: List[SnapshotPolicy]
    ) -> PolicyFn:
        if progress < 0.05 or not snapshots:
            return random.choice(_BOTS_SIMPLES)
        elif progress < 0.15:
            if random.random() < 0.5:
                return BotExperto()
            return random.choice(_BOTS_SIMPLES)
        elif progress < 0.40:
            if snapshots and random.random() < 0.5:
                return random.choice(snapshots)
            return BotExperto()
        elif progress < 0.70:
            if snapshots and random.random() < 0.7:
                return random.choice(snapshots)
            return BotExperto()
        else:
            if snapshots:
                return random.choice(snapshots)
            return BotExperto()
