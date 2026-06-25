"""
Entorno Gymnasium para v3.1 — 228 dimensiones.

Adaptado de v3. Usa ObservacionBuilderV31 para el vector de 228 dims.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.v3_1.dimensiones import DIM_V3_1
from src.v3_1.observacion import ObservacionBuilderV31
from src.v2_1.recompensas import CalculadoraRecompensasV21


class CorazonesEnvV31(gym.Env):
    """Entorno Gymnasium para Hearts — single-hand episodes, 228 dims.

    El agente juega UNA mano completa (13 bazas) contra oponentes
    con políticas fijas (bots heurísticos, snapshots, BotExperto).

    Observation space: Box(228,) float32
    Action space: Discrete(52) con action masking
    """

    def __init__(
        self,
        agente_idx: int = 0,
        politicas_oponentes: Optional[Dict[int, object]] = None,
        oracle_buffer: Optional[object] = None,
        oracle_rng: Optional[np.random.Generator] = None,
        epsilon: float = 0.0,
    ):
        super().__init__()

        self.agente_idx = agente_idx
        self.politicas_oponentes = politicas_oponentes or {}
        self._oracle_buffer = oracle_buffer
        self._oracle_rng = oracle_rng
        # exploration rate (0=off, 0.3=max exploration)
        self._epsilon = epsilon

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=26.0, shape=(DIM_V3_1,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(52)

        self._obs_builder = ObservacionBuilderV31()
        self._calc = CalculadoraRecompensasV21()

        # Estado interno
        self.motor: MotorCorazones = None  # type: ignore
        self._vacios: List[set] = []
        self._puntos_mano_actual: List[int] = []
        self._dama_picas_en: Optional[int] = None
        self._picas_agente_antes: List[Carta] = []
        self._ultima_carta_agente: Optional[Carta] = None
        self._recompensa_pendiente: float = 0.0
        self._mano_terminada: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reinicia el entorno para una nueva mano."""
        super().reset(seed=seed)
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

        RECOMPENSA TERMINAL-ONLY: solo se emite reward al final de la mano.
        Reward = 26 - puntos_agente (0=ganó, 26=perdió).
        Si el agente hace pleno (shooting the moon): reward = 52 (26 - (-26)).
        Las señales intermedias (Q♠, corazones, etc.) se descartan.

        Si el episodio ya terminó, hace auto-reset (comportamiento esperado
        por SB3/VecNormalize cuando se llama step() tras done=True).
        """
        # Auto-reset si se llama step() en episodio terminado
        if self._mano_terminada:
            obs, _ = self.reset()
            return obs, 0.0, False, False, {}

        carta = Carta._TODAS[action]

        # ── Epsilon-greedy exploration ──
        if self._epsilon > 0 and random.random() < self._epsilon:
            legales = self.motor.obtener_jugadas_legales(self.agente_idx)
            if legales:
                carta = random.choice(legales)
                action = carta.id

        # ── Hook MCTS Oracle ──
        if self._oracle_buffer is not None and self._oracle_rng is not None:
            self._recolectar_dato_oraculo(action)

        # Ejecutar jugada del agente (ignorar recompensa intermedia)
        self._ejecutar_jugada_agente(carta)

        if self._mano_terminada:
            obs = np.zeros(DIM_V3_1, dtype=np.float32)
            reward = self._calcular_recompensa_terminal()
            return obs, reward, True, False, self._info_final()

        # Jugar oponentes (ignorar recompensa intermedia)
        self._jugar_oponentes()

        if self._mano_terminada:
            obs = np.zeros(DIM_V3_1, dtype=np.float32)
            reward = self._calcular_recompensa_terminal()
            return obs, reward, True, False, self._info_final()

        obs = self._obs_builder.construir(
            self.motor, self.agente_idx,
            self._vacios, [0, 0, 0, 0],
            self._puntos_mano_actual, self._dama_picas_en,
        )
        return obs, 0.0, False, False, self._info_parcial()

    def action_masks(self) -> np.ndarray:
        """Máscara de acciones legales. Retorna all-zeros si mano terminada."""
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
        self._picas_agente_antes = []
        self._recompensa_pendiente = 0.0
        self._mano_terminada = False

        if self.motor.obtener_jugador_actual() != self.agente_idx:
            self._jugar_oponentes()

    def _ejecutar_jugada_agente(self, carta: Carta) -> float:
        """Juega la carta del agente y retorna recompensa parcial.

        Si la carta no es legal, juega la primera legal disponible.
        Si la baza se completa (4 cartas), la resuelve inmediatamente.
        """
        a = self.agente_idx
        legales = self.motor.obtener_jugadas_legales(a)

        if not legales:
            return 0.0

        if not any(c == carta for c in legales):
            carta = legales[0]

        # Guardar estado pre-jugada
        self._picas_agente_antes = [
            c for c in self.motor.jugadores[a].mano if c.palo == 2
        ]
        self._ultima_carta_agente = carta

        # Recompensa Q♠ preventiva
        qs_activa = self._dama_picas_en is None
        cartas_picas = [c for c in self.motor.jugadores[a].mano if c.palo == 2]
        recompensa = self._calc.recompensa_qs_preventivo(
            carta, cartas_picas, qs_activa,
        )

        self.motor.jugar_carta(a, carta)

        # ── Si la baza se completa (agente jugó 4°), resolver ──
        if len(self.motor.mesa) == 4:
            cartas_baza = [c for _, c in self.motor.mesa]
            puntos_baza = sum(c.puntos for c in cartas_baza)
            palo_salida = self.motor.palo_de_salida
            ganador = self.motor.resolver_baza()

            for _, c in self.motor.mesa:
                if c.es_dama_de_picas:
                    self._dama_picas_en = ganador

            recompensa += self._procesar_baza(
                ganador, puntos_baza, cartas_baza, palo_salida,
            )

            if all(len(j.mano) == 0 for j in self.motor.jugadores):
                recompensa += self._finalizar_mano()

        return recompensa

    def _jugar_oponentes(self) -> float:
        """Juega los turnos de los oponentes hasta que sea turno del agente."""
        recompensa = 0.0

        while (not self._mano_terminada
               and self.motor.obtener_jugador_actual() != self.agente_idx):
            actual = self.motor.obtener_jugador_actual()

            # ── Seleccionar carta del oponente ──
            legales = self.motor.obtener_jugadas_legales(actual)
            if not legales:
                break  # safety: no hay cartas legales → mano rota

            carta = self._seleccionar_carta_oponente(actual, legales)
            if carta is None:
                carta = legales[0]  # fallback

            # ── Tracking de vacíos ──
            if self.motor.mesa and self.motor.palo_de_salida is not None:
                if carta.palo != self.motor.palo_de_salida:
                    self._vacios[actual].add(self.motor.palo_de_salida)

            # ── Jugar ──
            self.motor.jugar_carta(actual, carta)

            # ── Resolver baza si está completa ──
            if len(self.motor.mesa) == 4:
                cartas_baza = [c for _, c in self.motor.mesa]
                puntos_baza = sum(c.puntos for c in cartas_baza)
                palo_salida = self.motor.palo_de_salida
                ganador = self.motor.resolver_baza()

                # Actualizar Q♠ tracker
                for _, c in self.motor.mesa:
                    if c.es_dama_de_picas:
                        self._dama_picas_en = ganador

                recompensa += self._procesar_baza(
                    ganador, puntos_baza, cartas_baza, palo_salida,
                )

                # ── ¿Fin de mano? ──
                if all(len(j.mano) == 0 for j in self.motor.jugadores):
                    recompensa += self._finalizar_mano()
                    break

        return recompensa

    def _seleccionar_carta_oponente(
        self, idx: int, legales: list,
    ):
        """Selecciona carta para un oponente usando su política."""
        politica = self.politicas_oponentes.get(idx)
        if politica is None:
            return legales[0] if legales else None
        try:
            return politica(self.motor, idx, legales)
        except Exception:
            return legales[0] if legales else None

    def _procesar_baza(
        self, ganador: int, puntos_baza: int,
        cartas_baza: list, palo_salida: Optional[int],
    ) -> float:
        """Procesa el resultado de una baza y retorna recompensa."""
        a = self.agente_idx
        recompensa = 0.0

        # Actualizar puntos
        self._puntos_mano_actual[ganador] += puntos_baza

        # Recompensa base
        recompensa += self._calc.recompensa_baza(cartas_baza, a, ganador)

        # Q♠ dump
        if self._ultima_carta_agente:
            recompensa += self._calc.recompensa_qs_dump(
                a, self._ultima_carta_agente,
                palo_salida, ganador,
            )

        # Moon block
        corazones_rivales = [
            sum(1 for c in self.motor.jugadores[j]
                .bazas_ganadas if c.es_corazon)
            for j in range(4) if j != a
        ]
        recompensa += self._calc.recompensa_moon_block(
            a, ganador, puntos_baza, corazones_rivales,
        )

        # Early safe burn
        recompensa += self._calc.recompensa_early_safe_burn(
            a, ganador, puntos_baza, self.motor.numero_baza - 1,
        )

        # Liability hold
        if self.motor.numero_baza >= 8 and self._dama_picas_en is None:
            if not any(c.es_dama_de_picas for _, c in self.motor.mesa):
                mano_agente = self.motor.jugadores[a].mano
                recompensa += self._calc.recompensa_liability_hold(
                    list(mano_agente), True, self.motor.numero_baza,
                )

        # ¿Fin de mano?
        if self.motor.numero_baza > 13:
            recompensa += self._finalizar_mano()

        return recompensa

    def _calcular_recompensa_terminal(self) -> float:
        """Recompensa terminal: 26 - puntos del agente.

        Shooting the moon: si el agente capturó los 26 puntos,
        recibe 52 (26 - (-26)) = el máximo posible.
        """
        a = self.agente_idx
        puntos = self._puntos_mano_actual[a]
        return 26.0 - float(puntos)

    def _finalizar_mano(self) -> float:
        """Finaliza la mano y retorna recompensa final."""
        self._mano_terminada = True
        a = self.agente_idx

        # Detectar shooting moon
        for i, jug in enumerate(self.motor.jugadores):
            if jug.contar_puntos_bazas() == 26:
                if i == a:
                    return (
                        self._calc.recompensa_shooting_moon(a, i)
                        + self._calc.recompensa_fin_mano(
                            self._puntos_mano_actual[a],
                        )
                        + self._calc.recompensa_posicion(
                            a, list(self._puntos_mano_actual),
                        )
                    )

        return (
            self._calc.recompensa_fin_mano(self._puntos_mano_actual[a])
            + self._calc.recompensa_posicion(
                a, list(self._puntos_mano_actual),
            )
        )

    def _recolectar_dato_oraculo(self, accion_agente: int) -> None:
        """Invoca el oráculo PIMC en bazas altas."""
        from src.v3.train_mcts import evaluar_y_guardar_batch

        if self.motor.numero_baza < 10:
            return

        legales = self.motor.obtener_jugadas_legales(self.agente_idx)
        if len(legales) <= 1:
            return

        obs = self._obs_builder.construir(
            self.motor, self.agente_idx,
            self._vacios, [0, 0, 0, 0],
            self._puntos_mano_actual, self._dama_picas_en,
        )

        evaluar_y_guardar_batch(
            motor=self.motor,
            agente_idx=self.agente_idx,
            legales=legales,
            buffer=self._oracle_buffer,
            obs=obs,
            rng=self._oracle_rng,
        )

    def _info_final(self) -> Dict[str, Any]:
        return {
            "score": self._puntos_mano_actual[self.agente_idx],
            "scores": list(self._puntos_mano_actual),
            "bazas": self.motor.numero_baza - 1,
        }

    def _info_parcial(self) -> Dict[str, Any]:
        return {"baza": self.motor.numero_baza}
