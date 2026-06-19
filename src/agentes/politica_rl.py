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

from src.entorno.dimensiones import DIM_ENTORNO

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
            obs_dim = DIM_ENTORNO
        self._obs_builder = ObservacionBuilder(dim=obs_dim)

        if vecnorm_path and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    data = pickle.load(f)
                self._obs_rms = data.get("obs_rms", None)
            except Exception:
                self._obs_rms = None

    def _pad_or_truncate(self, obs: np.ndarray) -> np.ndarray:
        """Ajusta observación al dim esperado por el modelo (padding o truncado).

        Útil para compatibilidad entre generaciones (194↔220).
        """
        target = self._obs_builder.dim
        if len(obs) == target:
            return obs
        if len(obs) < target:
            padded = np.zeros(target, dtype=np.float32)
            padded[:len(obs)] = obs
            return padded
        return obs[:target].astype(np.float32)

    def __call__(
        self,
        motor,
        jugador_idx: int,
        legales: List[Carta],
        obs: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Carta:
        """Elige una carta usando la política RL.

        Args:
            motor: Instancia de MotorCorazones.
            jugador_idx: Índice del jugador (0-3).
            legales: Lista de cartas legales.
            obs: Observación pre-construida (v12: oponentes reciben full obs).
                 Si es None, se construye desde el motor (compatibilidad).
            deterministic: Si True, argmax. Si False, samplea de la distribución
                (v12: oponentes usan False para diversidad en self-play).

        Returns:
            Carta elegida.
        """
        if obs is None:
            obs = self._obs_builder.construir_desde_motor(motor, jugador_idx)
        else:
            # Asegurar dtype y shape
            obs = np.asarray(obs, dtype=np.float32)
            if len(obs) != self._obs_builder.dim:
                # Padding o truncado para compatibilidad entre generaciones
                obs = self._pad_or_truncate(obs)

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
            obs, action_masks=mask, deterministic=deterministic
        )
        return Carta._TODAS[int(action)]
