"""
Entorno Gymnasium single-hand para Corazones v3.

Una mano = un episodio. Sin puntajes históricos, sin contexto multi-mano.

Mejoras sobre v2_1:
  - Observación enriquecida a 250 dims (ObservacionBuilderV3)
  - Recompensas v2_1 (8 señales: 4 base + 4 tácticas)
  - Hooks para MCTS-guided training (oráculo PIMC)
  - Compatible con TransformerFeatureExtractor y LSTM
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_V3
from src.v3.observacion import ObservacionBuilderV3
from src.v2_1.recompensas import CalculadoraRecompensasV21, RewardConfigV21


class CorazonesEnvV3(gym.Env):
    """Entorno Gymnasium para UNA mano de Corazones (v3).

    Cada episodio es exactamente una mano (13 bazas, max 52 pasos del agente).
    No hay puntuación histórica (todos empiezan en 0).

    Novedades v3:
    - Observation space: Box(250,) float32 (30 features extra sobre v2)
    - Rewards: 8 señales (4 base + 4 tácticas v2_1)
    - Hooks para MCTS oráculo (evaluar_y_guardar_batch)
    - Action space: Discrete(52) con Action Masking
    """

    def __init__(
        self,
        agente_idx: int = 0,
        politicas_oponentes: Optional[Dict[int, object]] = None,
        oracle_buffer: Optional[object] = None,
        oracle_rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()

        if not (0 <= agente_idx <= 3):
            raise ValueError(
                f"agente_idx debe estar entre 0 y 3, recibido {agente_idx}")

        self.agente_idx: int = agente_idx

        # Calculadora de recompensas v2_1 (8 señales)
        self._calc: CalculadoraRecompensasV21 = CalculadoraRecompensasV21()

        # Builder de observación v3 (250 dims)
        self._obs_builder: ObservacionBuilderV3 = ObservacionBuilderV3()

        # Políticas de oponentes
        self._politicas_oponentes: Dict[int, object] = (
            politicas_oponentes or {}
        )

        # MCTS oracle
        self._oracle_buffer = oracle_buffer
        self._oracle_rng = oracle_rng

        # Espacios Gymnasium
        self.observation_space = spaces.Box(
            low=0.0, high=26.0, shape=(DIM_V3,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(52)

        # Motor de juego
        self.motor: MotorCorazones = MotorCorazones()

        # Estado por mano
        self._vacios: List[set] = [set(), set(), set(), set()]
        self._puntos_mano_actual: List[int] = [0, 0, 0, 0]
        self._dama_picas_en: Optional[int] = None
        self._ultima_carta_agente: Optional[Carta] = None
        self._recompensa_pendiente: float = 0.0
        self._mano_terminada: bool = False

        # RNG
        self._rng: random.Random = random.Random()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno: nueva mano desde cero."""
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)
            random.seed(seed)

        self._iniciar_mano()
        obs = self._obs_builder.construir(
            self.motor, self.agente_idx,
            self._vacios, [0, 0, 0, 0],
            self._puntos_mano_actual, self._dama_picas_en,
        )
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ejecuta una acción del agente y avanza la mano.

        Args:
            action: Índice de la carta (0-51).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if self._mano_terminada:
            raise RuntimeError(
                "step() llamado en un episodio ya terminado. "
                "Llama a reset() primero."
            )

        carta = Carta._TODAS[action]
        recompensa = self._ejecutar_jugada_agente(carta)

        if self._mano_terminada:
            obs = np.zeros(DIM_V3, dtype=np.float32)
            return obs, recompensa, True, False, self._info_final()

        recompensa += self._jugar_oponentes()

        if self._mano_terminada:
            obs = np.zeros(DIM_V3, dtype=np.float32)
            return obs, recompensa, True, False, self._info_final()

        obs = self._obs_builder.construir(
            self.motor, self.agente_idx,
            self._vacios, [0, 0, 0, 0],
            self._puntos_mano_actual, self._dama_picas_en,
        )
        return obs, recompensa, False, False, self._info_parcial()

    def action_masks(self) -> np.ndarray:
        """Máscara de acciones legales para el agente."""
        if self._mano_terminada:
            return np.zeros(52, dtype=bool)

        legales = self.motor.obtener_jugadas_legales(self.agente_idx)
        mascara = np.zeros(52, dtype=bool)
        for c in legales:
            mascara[c.id] = True
        return mascara

    # ------------------------------------------------------------------
    # Estado interno
    # ------------------------------------------------------------------

    def _iniciar_mano(self) -> None:
        """Prepara una mano nueva."""
        self.motor = MotorCorazones()
        self.motor.repartir()
        self._vacios = [set(), set(), set(), set()]
        self._puntos_mano_actual = [0, 0, 0, 0]
        self._dama_picas_en = None
        self._ultima_carta_agente = None
        self._recompensa_pendiente = 0.0
        self._mano_terminada = False

        # Auto-jugar hasta que sea el turno del agente
        if self.motor.obtener_jugador_actual() != self.agente_idx:
            self._jugar_oponentes()

    def _ejecutar_jugada_agente(self, carta: Carta) -> float:
        """Ejecuta la jugada del agente y retorna recompensa acumulada."""
        self._ultima_carta_agente = carta

        idx = self.motor.obtener_jugador_actual()
        self._ejecutar_jugada(idx, carta)

        recompensa = 0.0
        if len(self.motor.mesa) == 4:
            recompensa += self._resolver_baza()
            if all(len(j.mano) == 0 for j in self.motor.jugadores):
                recompensa += self._finalizar_mano()

        return recompensa

    def _jugar_oponentes(self) -> float:
        """Juega las cartas de los oponentes hasta el turno del agente."""
        recompensa = 0.0

        while not self._mano_terminada:
            idx = self.motor.obtener_jugador_actual()
            if idx == self.agente_idx:
                break

            legales = self.motor.obtener_jugadas_legales(idx)
            if not legales:
                break

            carta = self._seleccionar_carta_oponente(idx, legales)
            self._ejecutar_jugada(idx, carta)

            if len(self.motor.mesa) == 4:
                recompensa += self._resolver_baza()
                if all(len(j.mano) == 0 for j in self.motor.jugadores):
                    recompensa += self._finalizar_mano()
                    break

        return recompensa

    def _ejecutar_jugada(self, jugador_idx: int, carta: Carta) -> None:
        """Ejecuta una jugada y actualiza tracking de vacíos."""
        if self.motor.mesa and self.motor.palo_de_salida is not None:
            if carta.palo != self.motor.palo_de_salida:
                self._vacios[jugador_idx].add(self.motor.palo_de_salida)
        self.motor.jugar_carta(jugador_idx, carta)

    def _seleccionar_carta_oponente(
        self, idx: int, legales: List[Carta]
    ) -> Carta:
        """Selecciona carta para un oponente usando su política."""
        if idx in self._politicas_oponentes:
            politica = self._politicas_oponentes[idx]
            obs = self._obs_builder.construir_desde_motor(
                self.motor, idx,
            )
            try:
                return politica(self.motor, idx, legales, obs=obs)
            except TypeError:
                return politica(self.motor, idx, legales)
        return self._rng.choice(legales)

    # ------------------------------------------------------------------
    # Resolución de bazas y fin de mano
    # ------------------------------------------------------------------

    def _resolver_baza(self) -> float:
        """Resuelve la baza actual y retorna recompensa."""
        cartas_baza = [c for _, c in self.motor.mesa]
        palo_salida = self.motor.palo_de_salida
        puntos_en_baza = sum(c.puntos for c in cartas_baza)
        ganador = self.motor.resolver_baza()

        # Actualizar puntos de la mano
        for i, jug in enumerate(self.motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

        # Tracking Dama de Picas
        for c in cartas_baza:
            if c.es_dama_de_picas:
                self._dama_picas_en = ganador
                break

        # ── Recompensa base per-baza ──
        recompensa = self._calc.recompensa_baza(
            cartas_baza, self.agente_idx, ganador,
        )

        # ── Q♠ dump on rival ──
        if self._ultima_carta_agente is not None:
            recompensa += self._calc.recompensa_qs_dump(
                self.agente_idx,
                self._ultima_carta_agente,
                palo_salida,
                ganador,
            )
        self._ultima_carta_agente = None

        # ── Moon block ──
        corazones_rivales = [
            sum(1 for c in self.motor.jugadores[i].bazas_ganadas
                if c.es_corazon)
            for i in range(4) if i != self.agente_idx
        ]
        recompensa += self._calc.recompensa_moon_block(
            self.agente_idx, ganador, puntos_en_baza, corazones_rivales,
        )

        # ── Early safe burn ──
        recompensa += self._calc.recompensa_early_safe_burn(
            self.agente_idx, ganador, puntos_en_baza,
            self.motor.numero_baza - 1,
        )

        # ── Liability hold penalty (tras baza 7+) ──
        if self.motor.numero_baza >= 8:
            q_activa = self._dama_picas_en is None
            if q_activa:
                if not any(c.es_dama_de_picas for _, c in self.motor.mesa):
                    mano_agente = self.motor.jugadores[self.agente_idx].mano
                    recompensa += self._calc.recompensa_liability_hold(
                        list(mano_agente), q_activa,
                        self.motor.numero_baza,
                    )

        return recompensa

    def _finalizar_mano(self) -> float:
        """Finaliza la mano y retorna recompensa final."""
        self._mano_terminada = True

        # Detectar shooting moon
        for i, jug in enumerate(self.motor.jugadores):
            if jug.contar_puntos_bazas() == 26:
                if i == self.agente_idx:
                    return (
                        self._calc.recompensa_shooting_moon(
                            self.agente_idx, i,
                        )
                        + self._calc.recompensa_fin_mano(
                            self._puntos_mano_actual[self.agente_idx],
                        )
                        + self._calc.recompensa_posicion(
                            self.agente_idx,
                            list(self._puntos_mano_actual),
                        )
                    )
                break

        # Recompensa de fin de mano: 26 - mis_puntos
        mis_puntos = self._puntos_mano_actual[self.agente_idx]
        return (
            self._calc.recompensa_fin_mano(mis_puntos)
            + self._calc.recompensa_posicion(
                self.agente_idx, list(self._puntos_mano_actual),
            )
        )

    # ------------------------------------------------------------------
    # Info dicts
    # ------------------------------------------------------------------

    def _info_parcial(self) -> Dict[str, Any]:
        return {
            "baza": self.motor.numero_baza,
            "puntos_mano": list(self._puntos_mano_actual),
            "mano_terminada": False,
        }

    def _info_final(self) -> Dict[str, Any]:
        return {
            "baza": self.motor.numero_baza,
            "puntos_mano": list(self._puntos_mano_actual),
            "puntos_agente": self._puntos_mano_actual[self.agente_idx],
            "mano_terminada": True,
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def get_observation(self) -> np.ndarray:
        """Retorna la observación actual sin avanzar el estado."""
        if self._mano_terminada:
            return np.zeros(DIM_V3, dtype=np.float32)
        return self._obs_builder.construir(
            self.motor, self.agente_idx,
            self._vacios, [0, 0, 0, 0],
            self._puntos_mano_actual, self._dama_picas_en,
        )

    @property
    def mano_terminada(self) -> bool:
        return self._mano_terminada


def crear_entorno_v3(
    agente_idx: int = 0,
    politicas_oponentes: Optional[Dict[int, object]] = None,
    oracle_buffer: Optional[object] = None,
    oracle_rng: Optional[np.random.Generator] = None,
) -> CorazonesEnvV3:
    """Factory function para crear un entorno v3."""
    return CorazonesEnvV3(
        agente_idx=agente_idx,
        politicas_oponentes=politicas_oponentes,
        oracle_buffer=oracle_buffer,
        oracle_rng=oracle_rng,
    )


__all__ = ["CorazonesEnvV3", "crear_entorno_v3"]
