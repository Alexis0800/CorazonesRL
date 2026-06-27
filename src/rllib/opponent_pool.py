"""
Pool de oponentes para self-play en Hearts — curriculum de 5 fases.

  Fase 0  (0–5%):   3 bots simples — bootstrap, aprender reglas básicas.
  Fase 1  (5–15%):  2 bots simples + 1 BotExperto — introducir oponente duro.
  Fase 2 (15–40%):  1 BotExperto + 2 snapshots — mezcla experto + self-play.
  Fase 3 (40–70%):  1 BotExperto (ANCLA) + 2 snapshots del pool completo.
  Fase 4 (70–100%): 1 BotExperto (ANCLA) + 2 snapshots recientes — presión máxima.

BotExperto se mantiene como ANCLA fija en todas las fases ≥1. Lección de
v10_lstm: el self-play PURO (fases 3-4 sin ancla) provoca regresión vs
oponentes de distribución distinta (el agente olvida el juego robusto y se
sobre-especializa en su propio estilo). Un ancla de 1 experto en cada mesa
preserva la robustez sin renunciar al refinamiento por self-play.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
from src.entorno.dimensiones import DIM_ENTORNO
from src.entorno.observacion import ObservacionBuilder

PolicyFn = Callable[[MotorCorazones, int, List[Carta]], Carta]

_BOTS_SIMPLES: List[PolicyFn] = [bot_conservador, bot_agresivo, bot_evasivo]


class SnapshotPolicy:
    """Wrapper serializable que adapta pesos de política a (motor, idx, legales) -> Carta.

    Almacena los pesos como dict de numpy arrays (picklable) en lugar del objeto
    Policy de Ray (que contiene _thread.RLock y no es serializable entre workers).
    El modelo PyTorch se reconstruye de forma lazy en cada worker.
    """

    def __init__(self, policy, obs_dim: int = DIM_ENTORNO):
        self._weights: dict = policy.get_weights()
        self._obs_dim = obs_dim
        self._obs_builder = ObservacionBuilder(dim=obs_dim)
        self._model = None
        # Estado LSTM por jugador (idx -> [h, c]) y marcador previo por jugador,
        # para arrastrar la memoria a lo largo de la partida y resetearla al
        # inicio de cada partida. Dict por idx para el caso de que el mismo
        # snapshot ocupe varios asientos a la vez.
        self._lstm_state: dict = {}
        self._prev_scores_sum: dict = {}

    @classmethod
    def from_weights(cls, weights: dict, obs_dim: int = DIM_ENTORNO) -> "SnapshotPolicy":
        """Crea un SnapshotPolicy directamente desde un dict de pesos numpy."""
        obj = object.__new__(cls)
        obj._weights = weights
        obj._obs_dim = obs_dim
        obj._obs_builder = ObservacionBuilder(dim=obs_dim)
        obj._model = None
        obj._lstm_state = {}
        obj._prev_scores_sum = {}
        return obj

    def __getstate__(self):
        return {"weights": self._weights, "obs_dim": self._obs_dim}

    def __setstate__(self, state):
        self._weights = state["weights"]
        self._obs_dim = state["obs_dim"]
        self._obs_builder = ObservacionBuilder(dim=self._obs_dim)
        self._model = None
        self._lstm_state = {}
        self._prev_scores_sum = {}

    def _get_model(self):
        """Construye el modelo PyTorch desde los pesos la primera vez (lazy).

        Detecta automáticamente si los pesos son de HeartsLSTMModel o HeartsActionMaskModel
        inspeccionando las claves del state_dict.
        """
        if self._model is not None:
            return self._model

        import torch
        from src.rllib.model import HeartsActionMaskModel, HeartsLSTMModel
        from gymnasium import spaces

        obs_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(52,), dtype=np.float32),
        })
        action_space = spaces.Discrete(52)

        is_lstm = any("_lstm_layer" in k for k in self._weights.keys())

        if is_lstm:
            lstm_hidden = self._weights["_lstm_layer.weight_hh_l0"].shape[1]
            model_config = {
                "lstm_cell_size": lstm_hidden,
                "max_seq_len": 13,
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            }
            model = HeartsLSTMModel(obs_space, action_space, 52, model_config, "snapshot")
        else:
            model_config = {"fcnet_hiddens": [512, 512, 256], "fcnet_activation": "relu", "vf_share_layers": False}
            model = HeartsActionMaskModel(obs_space, action_space, 52, model_config, "snapshot")

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

            initial = model.get_initial_state()
            es_recurrente = bool(initial)
            if es_recurrente:
                # Arrastrar el estado LSTM a lo largo de la PARTIDA (igual que en
                # entrenamiento, donde el episodio es una partida completa). Se
                # resetea al inicio de una partida nueva, detectado porque el
                # marcador acumulado baja (de ~100 a 0) o es la primera llamada.
                scores_sum = sum(j.puntuacion_historica for j in motor.jugadores)
                prev = self._prev_scores_sum.get(idx)
                if idx not in self._lstm_state or prev is None or scores_sum < prev:
                    self._lstm_state[idx] = [s.unsqueeze(0) for s in initial]
                self._prev_scores_sum[idx] = scores_sum
                state = self._lstm_state[idx]
            else:
                state = []

            logits, new_state = model.forward(input_dict, state, None)
            if es_recurrente:
                self._lstm_state[idx] = new_state  # arrastrar al siguiente step
            action = int(logits.argmax(dim=1).item())

        carta = Carta._TODAS[action]
        if carta not in legales:
            carta = random.choice(legales)
        return carta


class OpponentPool:
    """Pool de oponentes para self-play con bots y snapshots históricos.

    Usa 3 fases:
      - Fase 0 (0–5%):   3 bots simples para aprender las reglas básicas.
      - Fase 1 (5–20%):  2 bots + 1 snapshot. Transición gradual que evita el
                          shock de distribución y permite recuperar entropía.
      - Fase 2 (20–100%): SIEMPRE 1 bot + 2 snapshots. Nunca self-play puro.

    La factory retornada acepta `agente_idx` como parámetro para soportar
    rotación multi-posición (el agente puede entrenar desde cualquier asiento).

    Uso:
        pool = OpponentPool(snapshot_dir="models/v_rllib/snapshots")
        factory = pool.make_factory(progress=0.3)
        # En el env:
        opponents = factory(agente_idx=2)  # dict {0: fn, 1: fn, 3: fn}
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

    def add_snapshot(self, policy) -> None:
        """Añade una nueva política snapshot al pool (FIFO si excede el máximo)."""
        snap = SnapshotPolicy(policy, obs_dim=self._obs_dim)
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def make_factory(self, progress: float) -> Callable:
        """Devuelve un opponent_factory serializable para la fase actual.

        El closure captura una copia de los snapshots disponibles en este momento.
        Llamar periódicamente para incorporar nuevos snapshots al pool de rivales.

        Args:
            progress: fracción de entrenamiento completada (0.0 – 1.0).

        Returns:
            Callable(agente_idx: int) -> dict[int, PolicyFn]
        """
        snapshots = list(self._snapshots)
        # Snapshots recientes: últimos 10 (fase 4 solo usa estos para máxima presión)
        snapshots_recientes = snapshots[-10:] if len(snapshots) >= 10 else snapshots

        def _factory(agente_idx: int = 0) -> Dict[int, PolicyFn]:
            opp_indices = [i for i in range(4) if i != agente_idx]
            random.shuffle(opp_indices)
            fns: Dict[int, PolicyFn] = {}

            if progress < 0.05 or len(snapshots) == 0:
                # Fase 0: bootstrap con 3 bots simples
                for idx in opp_indices:
                    fns[idx] = random.choice(_BOTS_SIMPLES)

            elif progress < 0.15 or len(snapshots) < 1:
                # Fase 1: 2 bots simples + 1 BotExperto — introduce oponente duro
                fns[opp_indices[0]] = random.choice(_BOTS_SIMPLES)
                fns[opp_indices[1]] = random.choice(_BOTS_SIMPLES)
                fns[opp_indices[2]] = BotExperto()

            elif progress < 0.40 or len(snapshots) < 2:
                # Fase 2: 1 BotExperto + 2 snapshots — mezcla experto + self-play
                fns[opp_indices[0]] = BotExperto()
                fns[opp_indices[1]] = random.choice(snapshots)
                fns[opp_indices[2]] = random.choice(snapshots)

            elif progress < 0.70:
                # Fase 3: 1 BotExperto (ANCLA) + 2 snapshots del pool completo.
                # El ancla fija evita la regresión del self-play puro vs oponentes
                # de distribución distinta (ver Rediseño_v10: regresión observada
                # en v10_lstm al quitar el ancla).
                fns[opp_indices[0]] = BotExperto()
                fns[opp_indices[1]] = random.choice(snapshots)
                fns[opp_indices[2]] = random.choice(snapshots)

            else:
                # Fase 4: 1 BotExperto (ANCLA) + 2 snapshots recientes — presión
                # máxima contra versiones propias pero conservando robustez.
                pool = snapshots_recientes if len(snapshots_recientes) >= 1 else snapshots
                fns[opp_indices[0]] = BotExperto()
                fns[opp_indices[1]] = random.choice(pool)
                fns[opp_indices[2]] = random.choice(pool)

            return fns

        return _factory
