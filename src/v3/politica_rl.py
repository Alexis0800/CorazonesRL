"""
Política RL para v3 (250-dim, TransformerFeatureExtractor).

Adaptador que envuelve un MaskablePPO v3 como política de juego compatible
con la interfaz (motor, jugador_idx, legales) → Carta.

Usa ObservacionBuilderV3 para generar observaciones completas de 250 dimensiones,
incluyendo las 30 features enriquecidas que el modelo espera.

A diferencia de PoliticaSB3 (src/agentes/politica_rl.py), este adaptador:
  - Usa ObservacionBuilderV3 (no ObservacionBuilder base)
  - Requiere un VecNormalize wrapper externo para normalización
  - Espera que el modelo ya tenga el env wrappeado con VecNormalize
"""

from __future__ import annotations

import os
import pickle
from typing import Any, List, Optional

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.v3.observacion import ObservacionBuilderV3, DIM_V3


class PoliticaSB3V3:
    """Adaptador de política RL para modelos v3 (250-dim, Transformer).

    Carga el modelo MaskablePPO y lo envuelve como callable compatible
    con la interfaz de estrategia (motor, idx, legales) → Carta.

    Args:
        model: Instancia MaskablePPO ya cargada con VecNormalize wrapper.
        agente_idx: Índice del jugador que controla (0-3).
        vecnorm_path: Ruta al archivo _vecnorm.pkl (para normalizar obs).
            Si es None, se asume que el modelo tiene VecNormalize wrappeado.
    """

    def __init__(
        self,
        model: Any,
        agente_idx: int,
        vecnorm_path: Optional[str] = None,
    ):
        self.model = model
        self.agente_idx = agente_idx
        self._obs_builder = ObservacionBuilderV3(dim=DIM_V3)
        self._obs_rms = None

        if vecnorm_path and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    data = pickle.load(f)
                self._obs_rms = data.get("obs_rms", None)
            except Exception:
                self._obs_rms = None

    def normalizar_obs(self, obs: np.ndarray) -> np.ndarray:
        """Aplica normalización VecNormalize a la observación."""
        if self._obs_rms is None:
            return obs
        mean = np.array(self._obs_rms.mean)
        var = np.array(self._obs_rms.var)
        return np.clip(
            (obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0
        ).astype(np.float32)

    def __call__(
        self,
        motor: MotorCorazones,
        jugador_idx: int,
        legales: List[Carta],
        deterministic: bool = True,
    ) -> Carta:
        """Elige una carta usando la política RL v3.

        Construye la observación completa 250-dim usando ObservacionBuilderV3,
        la normaliza (si hay stats VecNormalize), aplica la máscara de acciones
        y predice la mejor acción.

        Args:
            motor: Instancia de MotorCorazones.
            jugador_idx: Índice del jugador (0-3).
            legales: Lista de cartas legales.
            deterministic: Si True, argmax. Si False, samplea.

        Returns:
            Carta elegida.
        """
        # Construir observación completa 250-dim
        from src.entorno.single_agent import CorazonesEnv
        # Usamos CorazonesEnvV3 pero solo para construir la obs — necesitamos
        # trackear el estado intramano como lo hace CorazonesEnvV3.
        obs = self._construir_obs_completa(motor, jugador_idx)

        # Normalizar
        if self._obs_rms is not None:
            obs = self.normalizar_obs(obs)

        # Máscara de acciones
        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True

        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=deterministic
        )
        return Carta._TODAS[int(action)]

    def _construir_obs_completa(
        self, motor: MotorCorazones, jugador_idx: int
    ) -> np.ndarray:
        """Construye observación 250-dim completa desde el estado del motor.

        Replica la lógica de CorazonesEnvV3._construir_observacion()
        usando ObservacionBuilderV3.
        """
        a = jugador_idx

        # Inferir vacíos desde el estado del motor
        vacios = self._inferir_vacios(motor, a)

        # Puntuación histórica
        puntuacion_historica = [
            motor.jugadores[i].puntuacion_historica for i in range(4)
        ]

        # Puntos en la mano actual
        puntos_mano_actual = self._calcular_puntos_mano(motor)

        # Dama de picas: inferir quién la tiene
        dama_picas_en = self._inferir_q_picas(motor, a, vacios)

        # Flags estratégicos
        pozo_viable = self._detectar_pozo_viable(motor, a, puntos_mano_actual)
        debo_arriesgar = puntuacion_historica[a] >= 85
        puedo_alimentar = any(
            puntuacion_historica[i] >= 80 and i != a
            for i in range(4)
        )

        return self._obs_builder.construir(
            motor=motor,
            agente_idx=a,
            vacios=vacios,
            puntuacion_historica=puntuacion_historica,
            puntos_mano_actual=puntos_mano_actual,
            dama_picas_en=dama_picas_en,
            pozo_viable=pozo_viable,
            debo_arriesgar=debo_arriesgar,
            puedo_alimentar=puedo_alimentar,
        )

    # ------------------------------------------------------------------
    # Helpers de estado (replican lógica de CorazonesEnvV3)
    # ------------------------------------------------------------------

    @staticmethod
    def _inferir_vacios(motor: MotorCorazones, agente_idx: int) -> List[set]:
        """Infiera vacíos conocidos desde el estado del motor."""
        vacios: List[set] = [set(), set(), set(), set()]

        # El agente conoce sus propios vacíos
        palos_en_mano = {c.palo for c in motor.jugadores[agente_idx].mano}
        for p in range(4):
            if p not in palos_en_mano:
                vacios[agente_idx].add(p)

        # Inferir de bazas anteriores
        for j in range(4):
            if j == agente_idx:
                continue
            palos_jugados = {c.palo for c in motor.jugadores[j].bazas_ganadas}
            for p in range(4):
                if p not in palos_jugados:
                    # Podría ser void, pero no podemos confirmar sin más info
                    pass

        return vacios

    @staticmethod
    def _calcular_puntos_mano(motor: MotorCorazones) -> List[int]:
        """Calcula puntos acumulados en la mano actual."""
        puntos = [0, 0, 0, 0]
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                puntos[j] += c.puntos
        # + puntos en la mesa actual que se asignarán al ganador (no se sabe aún)
        return puntos

    @staticmethod
    def _inferir_q_picas(
        motor: MotorCorazones, agente_idx: int, vacios: List[set]
    ) -> Optional[int]:
        """Infiera quién tiene Q♠."""
        # Si ya fue jugada y está en las bazas de alguien
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                if c.es_dama_de_picas:
                    return j
        # Si está en la mesa
        for _, c in motor.mesa:
            if c.es_dama_de_picas:
                return None  # Está en juego, aún no capturada
        # Si está en mi mano
        for c in motor.jugadores[agente_idx].mano:
            if c.es_dama_de_picas:
                return agente_idx
        # No sabemos quién la tiene
        return None

    @staticmethod
    def _detectar_pozo_viable(
        motor: MotorCorazones, agente_idx: int, puntos_mano: List[int]
    ) -> bool:
        """Detecta si shooting the moon es viable para el agente."""
        mi_puntos = puntos_mano[agente_idx]
        # Necesito al menos 10 corazones + Q♠ para que sea viable
        return mi_puntos >= 10
