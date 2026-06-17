"""
Política RL para el juego de Corazones (Strategy Pattern para RL).

Adaptador que envuelve un modelo MaskablePPO de SB3 como política de juego,
compatible con la interfaz (motor, jugador_idx, legales) → Carta.

Soporta normalización VecNormalize y padding automático 190→194.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, List, Optional

import numpy as np

from src.dominio.carta import Carta
from src.entorno.observacion import ObservacionBuilder


class PoliticaSB3:
    """Adaptador que envuelve un snapshot MaskablePPO como política de oponente.

    El modelo se carga con MaskablePPO.load() que restaura sus stats de
    VecNormalize. Al llamar predict(), las observaciones se normalizan
    automáticamente usando esas stats.

    Auto-detects obs_dim from the model's observation_space para
    compatibilidad entre generaciones (194 ↔ 220).
    """

    def __init__(
        self,
        model: Any,
        agente_idx: int,
        vecnorm_path: Optional[str] = None,
    ):
        self.model = model
        self.agente_idx = agente_idx
        self._obs_rms = None

        # Auto-detectar dimensión de observación del modelo cargado
        try:
            obs_dim = model.observation_space.shape[0]
        except Exception:
            obs_dim = 194
        self._obs_builder = ObservacionBuilder(dim=obs_dim)

        if vecnorm_path and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    data = pickle.load(f)
                self._obs_rms = data.get("obs_rms", None)
            except Exception:
                self._obs_rms = None

    def __call__(self, motor, jugador_idx: int, legales: List[Carta]) -> Carta:
        obs = self._obs_builder.construir_desde_motor(motor, jugador_idx)

        if self._obs_rms is not None:
            mean = np.array(self._obs_rms.mean)
            var = np.array(self._obs_rms.var)
            obs = np.clip(
                (obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0
            ).astype(np.float32)

        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True

        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True
        )
        return Carta._TODAS[int(action)]
