"""
Entorno Gymnasium single-hand para Corazones (v2_1).

Una mano = un episodio. Sin puntajes históricos, sin contexto multi-mano.
Recompensa mejorada con 4 señales tácticas adicionales a v2_ronda.

Importa componentes compartidos de src.dominio (lógica pura),
src.agentes (bots), src.entorno.observacion y src.entorno.dimensiones.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.observacion import ObservacionBuilder
from src.entorno.dimensiones import DIM_ENTORNO, DIMS_VALIDAS
from src.v2_1.recompensas import (
    CalculadoraRecompensasV21,
    RewardConfigV21,
)


class CorazonesEnvV21(gym.Env):
    """Entorno Gymnasium para UNA mano de Corazones con rewards tácticos v2.1.

    Cada episodio es exactamente una mano (13 bazas, max 52 pasos del agente).
    No hay puntuación histórica (todos empiezan en 0).
    No hay contexto de partida multi-mano.

    Nuevo vs v2_ronda: 4 señales tácticas adicionales (Q♠ dump, moon block,
    early safe burn, liability hold).

    Observation space: Box(220,) float32 (mismas dimensiones que v1/v2).
    Action space: Discrete(52) con Action Masking.
    """

    def __init__(
        self,
        agente_idx: int = 0,
        politicas_oponentes: Optional[Dict[int, object]] = None,
        obs_dim: int = DIM_ENTORNO,
    ) -> None:
        super().__init__()

        if not (0 <= agente_idx <= 3):
            raise ValueError(
                f"agente_idx debe estar entre 0 y 3, recibido {agente_idx}")
        if obs_dim not in DIMS_VALIDAS:
            raise ValueError(
                f"obs_dim debe ser una de {DIMS_VALIDAS}, recibido {obs_dim}")

        self.agente_idx: int = agente_idx
        self._obs_dim: int = obs_dim

        # Calculadora de recompensas v2.1 (mejorada)
        self._calc: CalculadoraRecompensasV21 = CalculadoraRecompensasV21()

        # Builder de observación (compartido con v1)
        self._obs_builder: ObservacionBuilder = ObservacionBuilder(dim=obs_dim)

        # Políticas de oponentes
        self._politicas_oponentes: Dict[int,
                                        object] = politicas_oponentes or {}

        # Espacios Gymnasium
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(52)

        # Motor de juego (lógica pura, compartido)
        self.motor: MotorCorazones = MotorCorazones()

        # Estado por mano
        self._vacios: List[set] = [set(), set(), set(), set()]
        self._dama_picas_en: Optional[int] = None
        self._puntos_mano_actual: List[int] = [0, 0, 0, 0]
        self._pleno_jugador: Optional[int] = None
        self._ultima_carta_agente: Optional[Carta] = None

        # Recompensa acumulada pendiente
        self._recompensa_pendiente: float = 0.0

        # Flag de mano terminada
        self._mano_terminada: bool = False

        # RNG para reproducibilidad
        self._rng: random.Random = random.Random()

    # ------------------------------------------------------------------
    # Ciclo de vida Gymnasium
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno: nueva mano desde cero.

        Returns:
            Tupla (observacion_inicial, info).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Iniciar mano fresca
        self._iniciar_mano()

        obs = self._construir_observacion()
        info: Dict[str, Any] = {
            "agente_idx": self.agente_idx,
            "puntos_mano": list(self._puntos_mano_actual),
            "numero_baza": self.motor.numero_baza,
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ejecuta una acción (carta) del agente.

        Args:
            action: ID de la carta a jugar (0-51).

        Returns:
            Tupla (obs, reward, terminated, truncated, info).
        """
        if self._mano_terminada:
            return (
                self._construir_observacion(),
                0.0,
                True,
                False,
                {"agente_idx": self.agente_idx, "puntos_mano": [0, 0, 0, 0],
                 "numero_baza": 14},
            )

        # Convertir acción a carta
        carta = Carta._TODAS[action]

        # Guardar para usar en _resolver_baza (Q♠ dump tracking)
        self._ultima_carta_agente = carta

        # Ejecutar la jugada del agente
        self._ejecutar_jugada(self.agente_idx, carta)

        # Auto-jugar hasta que sea el turno del agente de nuevo
        self._autoplay_hasta_turno_agente()

        # Construir observación y recolectar recompensa
        obs = self._construir_observacion()
        reward = self._recompensa_pendiente
        self._recompensa_pendiente = 0.0

        # Determinar si la mano terminó
        terminated = self._mano_terminada
        truncated = False

        info: Dict[str, Any] = {
            "agente_idx": self.agente_idx,
            "puntos_mano": list(self._puntos_mano_actual),
            "numero_baza": self.motor.numero_baza,
        }

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Retorna la máscara de acciones legales para el agente.

        Returns:
            Array booleano de shape (52,) donde True indica acción legal.
        """
        if self._mano_terminada:
            return np.zeros(52, dtype=np.bool_)

        actual = self.motor.obtener_jugador_actual()
        if actual != self.agente_idx:
            return np.zeros(52, dtype=np.bool_)

        legales = self.motor.obtener_jugadas_legales(self.agente_idx)
        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True
        return mask

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------

    def _iniciar_mano(self) -> None:
        """Inicia una mano nueva: reparte y resetea estado."""
        self.motor = MotorCorazones()
        self.motor.repartir()
        self._vacios = [set(), set(), set(), set()]
        self._dama_picas_en = None
        self._puntos_mano_actual = [0, 0, 0, 0]
        self._pleno_jugador = None
        self._ultima_carta_agente = None
        self._recompensa_pendiente = 0.0
        self._mano_terminada = False

        # Si no es turno del agente, auto-jugar hasta que lo sea
        if self.motor.obtener_jugador_actual() != self.agente_idx:
            self._autoplay_hasta_turno_agente()

    def _ejecutar_jugada(self, jugador_idx: int, carta: Carta) -> None:
        """Ejecuta una jugada y actualiza tracking de vacíos."""
        if self.motor.mesa and self.motor.palo_de_salida is not None:
            if carta.palo != self.motor.palo_de_salida:
                self._vacios[jugador_idx].add(self.motor.palo_de_salida)
        self.motor.jugar_carta(jugador_idx, carta)

    def _autoplay_hasta_turno_agente(self) -> None:
        """Auto-juega rivales hasta que sea el turno del agente
        o termine la mano."""
        while True:
            # Resolver baza si está completa
            if len(self.motor.mesa) == 4:
                self._resolver_baza()
                if self.motor.numero_baza > 13:
                    # Mano terminada (13 bazas resueltas)
                    self._finalizar_mano()
                    return

            # Si es turno del agente, detener
            if self.motor.obtener_jugador_actual() == self.agente_idx:
                return

            # Jugar para el rival
            actual = self.motor.obtener_jugador_actual()
            legales = self.motor.obtener_jugadas_legales(actual)
            if not legales:
                return

            if actual in self._politicas_oponentes:
                politica = self._politicas_oponentes[actual]
                try:
                    obs_completa = self._construir_observacion_desde(actual)
                    carta = politica(
                        self.motor, actual, legales,
                        obs=obs_completa, deterministic=False,
                    )
                except TypeError:
                    try:
                        carta = politica(
                            self.motor, actual, legales,
                            obs=obs_completa,
                        )
                    except TypeError:
                        carta = politica(self.motor, actual, legales)
            else:
                carta = self._rng.choice(legales)

            self._ejecutar_jugada(actual, carta)

    # ------------------------------------------------------------------
    # Resolución de bazas (con nuevas señales tácticas)
    # ------------------------------------------------------------------

    def _resolver_baza(self) -> None:
        """Resuelve la baza actual y asigna recompensas per-baza."""
        cartas_en_mesa = [c for _, c in self.motor.mesa]
        puntos_en_baza = sum(c.puntos for c in cartas_en_mesa)
        palo_salida = self.motor.palo_de_salida

        # Determinar ganador
        ganador_idx = self.motor.resolver_baza()

        # Actualizar puntos de la mano
        for i, jug in enumerate(self.motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

        # Tracking Dama de Picas
        for c in cartas_en_mesa:
            if c.es_dama_de_picas:
                self._dama_picas_en = ganador_idx
                break

        # ── Recompensa base per-baza (v2_ronda) ──
        r_baza = self._calc.recompensa_baza(
            cartas_en_mesa, self.agente_idx, ganador_idx,
        )
        self._recompensa_pendiente += r_baza

        # ── NEW: Q♠ dump on rival ──
        if self._ultima_carta_agente is not None:
            r_qs = self._calc.recompensa_qs_dump(
                self.agente_idx,
                self._ultima_carta_agente,
                palo_salida,
                ganador_idx,
            )
            self._recompensa_pendiente += r_qs
        self._ultima_carta_agente = None

        # ── NEW: Moon block ──
        corazones_rivales = [
            sum(1 for c in self.motor.jugadores[i]
                .bazas_ganadas if c.es_corazon)
            for i in range(4) if i != self.agente_idx
        ]
        r_block = self._calc.recompensa_moon_block(
            self.agente_idx, ganador_idx, puntos_en_baza, corazones_rivales,
        )
        self._recompensa_pendiente += r_block

        # ── NEW: Early safe burn ──
        r_burn = self._calc.recompensa_early_safe_burn(
            self.agente_idx, ganador_idx, puntos_en_baza,
            self.motor.numero_baza - 1,  # ya se resolvió, retrocedemos 1
        )
        self._recompensa_pendiente += r_burn

        # ── NEW: Liability hold penalty (aplicar tras baza 7+) ──
        if self.motor.numero_baza >= 8:  # post-baza 7+
            q_activa = self._dama_picas_en is None
            if q_activa:
                # Verificar si Q♠ no está en la mesa actual
                if not any(c.es_dama_de_picas for _, c in self.motor.mesa):
                    mano_agente = self.motor.jugadores[self.agente_idx].mano
                    r_liab = self._calc.recompensa_liability_hold(
                        list(mano_agente), q_activa,
                        self.motor.numero_baza,
                    )
                    self._recompensa_pendiente += r_liab

    def _finalizar_mano(self) -> None:
        """Aplica recompensas de fin de mano."""
        # Detectar shooting moon
        for i, jug in enumerate(self.motor.jugadores):
            if jug.contar_puntos_bazas() == 26:
                self._pleno_jugador = i
                if i == self.agente_idx:
                    self._recompensa_pendiente += \
                        self._calc.recompensa_shooting_moon(
                            self.agente_idx, i,
                        )
                break

        # Aplicar puntuación (solo para consistencia, no se usa fuera)
        self.motor.aplicar_puntuacion()

        # Recompensa de fin de mano: 26 - mis_puntos
        mis_puntos = self._puntos_mano_actual[self.agente_idx]
        r_fin = self._calc.recompensa_fin_mano(mis_puntos)
        self._recompensa_pendiente += r_fin

        # Bonus de posición
        r_pos = self._calc.recompensa_posicion(
            self.agente_idx, list(self._puntos_mano_actual),
        )
        self._recompensa_pendiente += r_pos

        # Marcar mano como terminada
        self._mano_terminada = True

    # ------------------------------------------------------------------
    # Construcción de observación
    # ------------------------------------------------------------------

    def _construir_observacion(self) -> np.ndarray:
        """Construye el vector de observación para el agente."""
        return self._construir_observacion_desde(self.agente_idx)

    def _construir_observacion_desde(self, jugador_idx: int) -> np.ndarray:
        """Construye observación desde la perspectiva de un jugador.

        En v2_1, la puntuación histórica siempre es [0,0,0,0]
        porque no hay contexto multi-mano.
        """
        return self._obs_builder.construir(
            motor=self.motor,
            agente_idx=jugador_idx,
            vacios=self._vacios,
            puntuacion_historica=[0, 0, 0, 0],  # sin historial
            puntos_mano_actual=self._puntos_mano_actual,
            dama_picas_en=self._dama_picas_en,
            pozo_viable=self._pozo_viable(jugador_idx),
            debo_arriesgar=False,
            puedo_alimentar=False,
        )

    def _pozo_viable(self, jugador_idx: int) -> bool:
        """Determina si shooting the moon es viable para un jugador.

        Condiciones: tiene Q♠ + al menos 8 corazones en mano,
        y no se han jugado cartas altas que lo impidan.
        """
        mano = self.motor.jugadores[jugador_idx].mano
        tiene_q = any(c.es_dama_de_picas for c in mano)
        corazones_en_mano = sum(1 for c in mano if c.es_corazon)
        return tiene_q and corazones_en_mano >= 8
