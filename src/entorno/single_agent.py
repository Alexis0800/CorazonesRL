"""
Entorno Gymnasium para el juego de Corazones (Módulo 2).

Envuelve el MotorCorazones (Módulo 1) en una interfaz estándar
Farama Gymnasium. Traduce el estado del juego a un vector de
observación de 187 dimensiones (np.float32), aplica Action Masking
conectado a obtener_jugadas_legales(), e implementa un sistema de
recompensas de suma cero (corto y largo plazo).

No depende de PyTorch ni de PettingZoo (reservados para el Módulo 3).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.dominio.carta import Carta, _PALOS, _PUNTOS, _ES_CORAZON, _ES_DOS_TREBOL
from src.dominio.motor import MotorCorazones


class CorazonesEnv(gym.Env):
    """Entorno Gymnasium para el juego de Corazones con Action Masking.

    El agente controla a un solo jugador (por defecto el índice 0).
    Los otros 3 jugadores son controlados por bots aleatorios que
    eligen cartas legales al azar.

    Observation space: Box(187,) de np.float32, normalizado [0, 1].
    Action space: Discrete(52) con enmascaramiento vía action_masks().
    """

    # ------------------------------------------------------------------
    # Constantes de recompensa
    # ------------------------------------------------------------------
    # Recompensas por evento (casting de cartas)
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -10.0
    REWARD_SHOOTING_MOON: float = 50.0
    # Recompensa final: ganar la partida es lo más importante (5x refuerzo)
    REWARD_PRIMERO: float = 500.0
    REWARD_SEGUNDO: float = 200.0
    REWARD_TERCERO: float = -200.0
    REWARD_CUARTO: float = -500.0

    # Recompensas densas (reward shaping): señales sutiles, no dominantes
    REWARD_NO_GANAR_BAZA_CON_PUNTOS: float = 0.3
    REWARD_DESCARTAR_CORAZON_SEGURO: float = 0.2
    REWARD_DESCARTAR_DAMA_SEGURO: float = 2.0     # v12: reducido 8.0→2.0 (evitar reward hacking)
    REWARD_GANAR_BAZA_SIN_PUNTOS: float = -0.1
    REWARD_PERDER_MANO: float = -2.0
    REWARD_GANAR_MANO: float = 2.0

    # --- NUEVAS RECOMPENSAS v5: correcciones estratégicas ---
    REWARD_Q_SPADES_SIN_POZO: float = -5.0         # v12: reducido -8.0→-5.0
    REWARD_GANAR_BAZA_CON_CORAZON: float = -2.0    # v12: reducido -3.0→-2.0
    REWARD_POR_PUNTO_EN_MANO: float = -0.1         # v12: reducido -0.2→-0.1

    # --- RECOMPENSAS v9-v10: correcciones tácticas (v12: magnitudes reducidas) ---
    PENALTY_LIDERAR_PICA_CON_Q_ACTIVA: float = -2.0     # v12: -1.0→-2.0
    REWARD_QUEMAR_MAXIMA_PALO_SEGURO: float = 1.0       # v12: 0.5→1.0
    PENALTY_LIDERAR_Q_EQUIVOCADO: float = -6.0          # v12: -3.0→-6.0 (penalty > reward de dump)
    REWARD_DUMP_Q_SIGUIENDO_PICAS: float = 1.0         # v12: 2.0→1.0
    PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE: float = -2.0  # v12: -3.0→-2.0
    REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA: float = 1.0  # v12: 2.0→1.0
    REWARD_QUEMAR_MAXIMA_FORZADA: float = 0.5           # v12: 0.3→0.5
    REWARD_QUEMAR_ALTA_SIGUIENDO_PALO: float = 1.0     # v12: 0.5→1.0
    PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD: float = -1.0  # v12: -5.0→-1.0
    REWARD_LIDERAR_Q_DUMP_SEGURO: float = 2.0          # v12: 8.0→2.0
    REWARD_DESCARTAR_CORAZON_BAJO_ROTO: float = 0.5    # v12: 2.5→0.5

    # Phase-gating: recompensas tácticas solo en bazas ≥ este umbral
    BAZA_TARDIA: int = 7
    PUNTUACION_MAXIMA: float = 100.0  # Umbral de fin de partida

    def __init__(
        self,
        agente_idx: int = 0,
        politicas_oponentes: Optional[Dict[int, object]] = None,
        obs_dim: int = 194,
    ) -> None:
        super().__init__()

        if not (0 <= agente_idx <= 3):
            raise ValueError(
                f"agente_idx debe estar entre 0 y 3, recibido {agente_idx}")
        if obs_dim not in (194, 220):
            raise ValueError(f"obs_dim debe ser 194 o 220, recibido {obs_dim}")

        self.agente_idx: int = agente_idx
        self._obs_dim: int = obs_dim

        # Políticas de oponentes: dict jugador_idx → callable(motor, idx, legales) → Carta
        # Si no se especifica, se usa selección aleatoria
        self._politicas_oponentes: Dict[int,
                                        object] = politicas_oponentes or {}

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(52)

        # Motor del Módulo 1 (sin dependencias de IA)
        self.motor: MotorCorazones = MotorCorazones()

        # Estado persistente entre manos
        self._puntuacion_historica: List[int] = [0, 0, 0, 0]
        self._vacios: List[set] = [set(), set(), set(), set()]
        self._dama_picas_en: Optional[int] = None
        self._puntos_mano_actual: List[int] = [0, 0, 0, 0]
        self._pleno_jugador: Optional[int] = None

        # Recompensa acumulada durante auto-play
        self._recompensa_pendiente: float = 0.0

        # Semilla aleatoria para reproducibilidad
        self._rng: random.Random = random.Random()

    # ------------------------------------------------------------------
    # Métodos del ciclo de vida Gymnasium
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno: nueva partida desde cero.

        Args:
            seed: Semilla aleatoria para reproducibilidad.
            options: Opciones adicionales (no utilizadas).

        Returns:
            Tupla (observación_inicial, info).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Reiniciar estado persistente
        self._puntuacion_historica = [0, 0, 0, 0]
        self._vacios = [set(), set(), set(), set()]
        self._dama_picas_en = None
        self._puntos_mano_actual = [0, 0, 0, 0]
        self._pleno_jugador = None
        self._recompensa_pendiente = 0.0

        # Iniciar primera mano
        self._iniciar_nueva_mano()

        return self._construir_observacion(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ejecuta una acción del agente y avanza el juego hasta su siguiente turno.

        Si la acción es ilegal, se reemplaza automáticamente por una acción
        legal aleatoria (fallback para compatibilidad con check_env).

        Args:
            action: Índice de la carta a jugar (0-51).

        Returns:
            Tupla (observación, recompensa, terminated, truncated, info).
        """
        mask = self.action_masks()

        # Si no hay acciones legales, el juego ya terminó o no es turno del agente
        if not np.any(mask):
            obs = self._construir_observacion()
            return obs, 0.0, self._juego_terminado(), False, {
                "puntuacion_historica": list(self._puntuacion_historica),
                "puntos_mano": list(self._puntos_mano_actual),
                "agente_idx": self.agente_idx,
            }

        # Fallback: si la acción es ilegal, elegir una legal aleatoria
        if not mask[action]:
            legales = [i for i, m in enumerate(mask) if m]
            if legales:
                action = self._rng.choice(legales)
            else:
                # Sin legales: devolver estado actual
                obs = self._construir_observacion()
                return obs, 0.0, self._juego_terminado(), False, {}

        # Convertir acción (int) a Carta
        carta = Carta._TODAS[action]

        # Capturar estado PRE-jugada para correcciones tácticas
        # 0=first, 1=second, 2=third, 3=last
        posicion_en_baza = len(self.motor.mesa)

        # Ejecutar la jugada del agente
        self._ejecutar_jugada(self.agente_idx, carta)

        # --- PENALTY: liderar pica no-máxima con Q♠ activa ---
        # Solo si el agente lideró (posición 0), Q♠ sigue en juego,
        # la carta es pica (no Q♠), y no era la máxima pica del agente.
        if (posicion_en_baza == 0
                and self._dama_picas_en is None
                and carta.palo == 2
                and not carta.es_dama_de_picas):
            picas_en_mano = [
                c for c in self.motor.jugadores[self.agente_idx].mano
                if c.palo == 2
            ]
            if picas_en_mano and carta.valor < max(c.valor for c in picas_en_mano):
                self._recompensa_pendiente += self.PENALTY_LIDERAR_PICA_CON_Q_ACTIVA

        # --- REWARD: liderar máxima de palo seguro (♣/♦) ---
        if (posicion_en_baza == 0
                and carta.palo in (0, 1)
                and self._es_maxima_en_mano(carta, self.agente_idx)):
            self._recompensa_pendiente += self.REWARD_QUEMAR_MAXIMA_PALO_SEGURO

        # --- PENALTY: liderar Q♠ en mal momento ---
        if posicion_en_baza == 0 and carta.es_dama_de_picas:
            baza_temprana = self.motor.numero_baza < self.BAZA_TARDIA
            soy_maxima_picas = self._es_maxima_en_mano(carta, self.agente_idx)
            if baza_temprana or soy_maxima_picas:
                self._recompensa_pendiente += self.PENALTY_LIDERAR_Q_EQUIVOCADO

        # --- REWARD: liderar Q♠ como dump seguro (baza ≥BAZA_TARDIA, K♠/A♠ en circ.) ---
        if posicion_en_baza == 0 and carta.es_dama_de_picas:
            baza_tardia = self.motor.numero_baza >= self.BAZA_TARDIA
            soy_maxima_picas = self._es_maxima_en_mano(carta, self.agente_idx)
            if baza_tardia and not soy_maxima_picas:
                # Verificar si K♠/A♠ están en circulación (no en cementerio ni en mano)
                cementerio_ids = set()
                for j in range(4):
                    for c in self.motor.jugadores[j].bazas_ganadas:
                        cementerio_ids.add(c.id)
                k_spades = next(
                    (c for c in Carta._TODAS if c.palo == 2 and c.valor == 13), None)
                a_spades = next(
                    (c for c in Carta._TODAS if c.palo == 2 and c.valor == 14), None)
                k_en_circulacion = k_spades and k_spades.id not in cementerio_ids
                a_en_circulacion = a_spades and a_spades.id not in cementerio_ids
                if k_en_circulacion or a_en_circulacion:
                    self._recompensa_pendiente += self.REWARD_LIDERAR_Q_DUMP_SEGURO

        # --- REWARD: quemar máxima forzada (siguiendo palo) ---
        if (posicion_en_baza > 0
                and self.motor.palo_de_salida is not None
                and carta.palo == self.motor.palo_de_salida
                and self._forzado_a_ganar(self.agente_idx)):
            cartas_palo = [
                c for c in self.motor.jugadores[self.agente_idx].mano
                if c.palo == self.motor.palo_de_salida
            ]
            if not cartas_palo or carta.valor >= max(c.valor for c in cartas_palo):
                self._recompensa_pendiente += self.REWARD_QUEMAR_MAXIMA_FORZADA

        # Auto-jugar hasta que sea el turno del agente de nuevo
        self._autoplay_hasta_turno_agente()

        # Construir observación y recolectar recompensa
        obs = self._construir_observacion()
        reward = self._recompensa_pendiente
        self._recompensa_pendiente = 0.0

        # Verificar estado terminal
        terminated = self._juego_terminado()
        truncated = False

        # Si el juego terminó, añadir recompensa de fin de partida
        if terminated:
            reward += self._calcular_recompensa_final()

        info: Dict[str, Any] = {
            "puntuacion_historica": list(self._puntuacion_historica),
            "puntos_mano": list(self._puntos_mano_actual),
            "agente_idx": self.agente_idx,
        }

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Retorna la máscara de acciones legales para el agente.

        Returns:
            Array booleano de shape (52,) donde True indica acción legal.
        """
        if not self.motor._mano_activa:
            # Juego terminado o sin mano activa
            return np.zeros(52, dtype=np.bool_)

        actual = self.motor.obtener_jugador_actual()
        if actual != self.agente_idx:
            # No es el turno del agente
            return np.zeros(52, dtype=np.bool_)

        legales = self.motor.obtener_jugadas_legales(self.agente_idx)
        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True
        return mask

    # ------------------------------------------------------------------
    # Lógica interna: ciclo de mano
    # ------------------------------------------------------------------

    def _iniciar_nueva_mano(self) -> None:
        """Inicia una nueva mano: reparte cartas y reinicia estado por mano."""
        self.motor.repartir()
        self._vacios = [set(), set(), set(), set()]
        self._dama_picas_en = None
        self._puntos_mano_actual = [0, 0, 0, 0]
        self._pleno_jugador = None

        # Limpiar bazas ganadas de la mano anterior
        for j in self.motor.jugadores:
            j.bazas_ganadas = []

        # Si no es el turno del agente al inicio, auto-jugar hasta que lo sea
        if self.motor.obtener_jugador_actual() != self.agente_idx:
            self._autoplay_hasta_turno_agente()

    def _ejecutar_jugada(self, jugador_idx: int, carta: Carta) -> None:
        """Ejecuta una jugada y actualiza el tracking de vacíos y dama de picas.

        Args:
            jugador_idx: Índice del jugador que juega.
            carta: Carta a jugar.
        """
        # Detectar void: si hay palo de salida y el jugador no sigue el palo
        if self.motor.mesa and self.motor.palo_de_salida is not None:
            if carta.palo != self.motor.palo_de_salida:
                self._vacios[jugador_idx].add(self.motor.palo_de_salida)

        self.motor.jugar_carta(jugador_idx, carta)

    def _autoplay_hasta_turno_agente(self) -> None:
        """Auto-juega para los rivales hasta que sea el turno del agente
        o hasta que termine la mano/partida."""
        while True:
            # Resolver baza si está completa
            if len(self.motor.mesa) == 4:
                self._resolver_baza_actual()

                # Si la mano terminó, iniciar nueva o detener
                if self.motor.numero_baza > 13:
                    self._finalizar_mano()
                    if self._juego_terminado():
                        return
                    self._iniciar_nueva_mano()
                    # Continuar el ciclo: verificar si es turno del agente
                    continue

            # Si es turno del agente, detener auto-play
            if self.motor.obtener_jugador_actual() == self.agente_idx:
                return

            # Auto-jugar para el jugador actual (rival)
            actual = self.motor.obtener_jugador_actual()
            legales = self.motor.obtener_jugadas_legales(actual)
            if not legales:
                # No debería ocurrir; por seguridad
                return
            # Usar política configurada o random por defecto
            if actual in self._politicas_oponentes:
                # v12: pasar observación completa a PoliticaSB3 para eliminar
                # el distribution mismatch entre entrenamiento (220d) e inferencia
                politica = self._politicas_oponentes[actual]
                try:
                    # Intentar pasar obs completa (soportado por PoliticaSB3)
                    obs_completa = self._construir_observacion_desde(actual)
                    carta = politica(self.motor, actual, legales, obs=obs_completa)
                except TypeError:
                    # Fallback: política legacy sin soporte de obs
                    carta = politica(self.motor, actual, legales)
            else:
                carta = self._rng.choice(legales)
            self._ejecutar_jugada(actual, carta)

    # ------------------------------------------------------------------
    # Helpers tácticos (v9)
    # ------------------------------------------------------------------

    def _es_maxima_en_mano(self, carta: Carta, jugador_idx: int) -> bool:
        """True si no hay carta más alta del mismo palo en la mano del jugador."""
        for c in self.motor.jugadores[jugador_idx].mano:
            if c.palo == carta.palo and c.valor > carta.valor:
                return False
        return True

    def _forzado_a_ganar(self, jugador_idx: int) -> bool:
        """True si el jugador está forzado a ganar la baza actual:
        TODAS sus cartas del palo de salida superan la máxima actual en mesa."""
        palo = self.motor.palo_de_salida
        mesa = self.motor.mesa
        if palo is None or not mesa:
            return False
        max_en_mesa = max(
            (c.valor for _, c in mesa if c.palo == palo), default=0)
        for c in self.motor.jugadores[jugador_idx].mano:
            if c.palo == palo and c.valor <= max_en_mesa:
                return False
        return True

    # ------------------------------------------------------------------
    # Resolución de bazas
    # ------------------------------------------------------------------

    def _resolver_baza_actual(self) -> None:
        """Resuelve la baza, asigna recompensas densas al agente y actualiza
        el tracking de la Dama de Picas."""
        # Capturar las cartas de la mesa ANTES de resolver
        cartas_en_mesa = [c for _, c in self.motor.mesa]
        idx_agente_en_mesa = None
        for i, (j, _) in enumerate(self.motor.mesa):
            if j == self.agente_idx:
                idx_agente_en_mesa = i
                break

        # Capturar palo de salida ANTES de resolver (resolver_baza lo limpia)
        palo_salida = self.motor.palo_de_salida
        # Capturar número de baza ANTES de resolver (se incrementa en resolver_baza)
        numero_baza = self.motor.numero_baza

        ganador = self.motor.resolver_baza()

        # Actualizar puntos de la mano actual
        for i, jug in enumerate(self.motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

        # Tracking de la Dama de Picas
        for c in cartas_en_mesa:
            if c.es_dama_de_picas:
                self._dama_picas_en = ganador
                break

        # Puntos totales en esta baza
        puntos_baza = sum(c.puntos for c in cartas_en_mesa)

        # Calcular recompensa para el agente por esta baza
        if ganador == self.agente_idx:
            # Penalización por puntos ganados (corazones + dama)
            gano_corazon = False
            gano_q_spades = False
            for c in cartas_en_mesa:
                if c.es_corazon:
                    self._recompensa_pendiente += self.REWARD_CORAZON
                    gano_corazon = True
                if c.es_dama_de_picas:
                    self._recompensa_pendiente += self.REWARD_DAMA_PICAS
                    gano_q_spades = True

            # --- NUEVO v5: penalización extra si ganó Q♠ sin pozo viable ---
            if gano_q_spades and not self._pozo_viable():
                self._recompensa_pendiente += self.REWARD_Q_SPADES_SIN_POZO

            # --- NUEVO v5: penalización extra si ganó corazones sin pozo ---
            if gano_corazon and not self._pozo_viable():
                self._recompensa_pendiente += self.REWARD_GANAR_BAZA_CON_CORAZON

            # Si ganó baza sin puntos, pequeña penalización
            if puntos_baza == 0:
                self._recompensa_pendiente += self.REWARD_GANAR_BAZA_SIN_PUNTOS
        else:
            # El agente NO ganó la baza → ¡BUENO si la baza tenía puntos!
            if puntos_baza > 0:
                self._recompensa_pendiente += self.REWARD_NO_GANAR_BAZA_CON_PUNTOS * \
                    min(puntos_baza, 3)

            # ¿El agente jugó un corazón o la Dama en esta baza y NO la ganó?
            if idx_agente_en_mesa is not None:
                carta_agente = cartas_en_mesa[idx_agente_en_mesa]
                if carta_agente.es_corazon:
                    self._recompensa_pendiente += self.REWARD_DESCARTAR_CORAZON_SEGURO
                if carta_agente.es_dama_de_picas:
                    self._recompensa_pendiente += self.REWARD_DESCARTAR_DAMA_SEGURO

        # --- v9-v10: recompensas tácticas por baza (phase-gated: solo bazas ≥ BAZA_TARDIA) ---
        if idx_agente_en_mesa is not None and numero_baza >= self.BAZA_TARDIA:
            carta_agente = cartas_en_mesa[idx_agente_en_mesa]

            # REWARD: quemar alta (A/K) siguiendo palo en baza limpia
            if (puntos_baza == 0
                    and palo_salida is not None
                    and carta_agente.palo == palo_salida
                    and carta_agente.valor >= 13):
                self._recompensa_pendiente += self.REWARD_QUEMAR_ALTA_SIGUIENDO_PALO

            # REWARD: soltar Q♠ siguiendo picas (dump seguro)
            if (carta_agente.es_dama_de_picas
                    and ganador != self.agente_idx
                    and palo_salida == 2):
                self._recompensa_pendiente += self.REWARD_DUMP_Q_SIGUIENDO_PICAS

            # REWARD: descartar K♠/A♠ en baza limpia con Q♠ activa
            if (puntos_baza == 0
                    and self._dama_picas_en is None
                    and carta_agente.palo == 2
                    and carta_agente.valor >= 13
                    and palo_salida is not None
                    and carta_agente.palo != palo_salida):
                self._recompensa_pendiente += self.REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA

            # PENALTY: ganar baza con puntos teniendo cartas perdedoras
            if (ganador == self.agente_idx
                    and puntos_baza > 0
                    and palo_salida is not None
                    and carta_agente.palo == palo_salida
                    and not self._pozo_viable()):
                self._recompensa_pendiente += self.PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE

            # --- v10 (Fase B): recompensas tácticas adicionales (phase-gated, baza ≥ BAZA_TARDIA) ---

            # PENALTY: ganar baza tardía sin puntos con máxima de palo seguro
            if (ganador == self.agente_idx
                    and puntos_baza == 0
                    and palo_salida in (0, 1)  # ♣/♦ = palos seguros
                    and carta_agente.palo == palo_salida
                    and self._es_maxima_en_mano(carta_agente, self.agente_idx)
                    and carta_agente.valor >= 13):  # A o K
                self._recompensa_pendiente += self.PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD

            # REWARD: descartar corazón en baza limpia con corazones rotos
            if (self.motor.corazones_rotos
                    and palo_salida is not None
                    and carta_agente.palo != palo_salida  # es descarte (void)
                    and carta_agente.es_corazon
                    and puntos_baza == 0):
                self._recompensa_pendiente += self.REWARD_DESCARTAR_CORAZON_BAJO_ROTO

    def _finalizar_mano(self) -> None:
        """Finaliza la mano actual: aplica puntuación, verifica pleno,
        y actualiza puntuaciones históricas."""
        # Detectar pleno ANTES de aplicar puntuación (los puntos crudos)
        puntos_crudos = [j.contar_puntos_bazas() for j in self.motor.jugadores]
        for i, pts in enumerate(puntos_crudos):
            if pts == 26:
                self._pleno_jugador = i
                if i == self.agente_idx:
                    self._recompensa_pendiente += self.REWARD_SHOOTING_MOON
                break

        # Aplicar puntuación (el motor maneja la conversión de pleno)
        puntuaciones_mano = self.motor.aplicar_puntuacion()

        # Actualizar puntuación histórica desde el motor
        for i in range(4):
            self._puntuacion_historica[i] = self.motor.jugadores[i].puntuacion_historica

        # Recompensa por mano: premiar si el agente sumó menos puntos que el promedio
        if self._pleno_jugador is None:
            puntos_agente = puntuaciones_mano[self.agente_idx]
            puntos_otros = [puntuaciones_mano[i]
                            for i in range(4) if i != self.agente_idx]
            if puntos_agente < min(puntos_otros):
                self._recompensa_pendiente += self.REWARD_GANAR_MANO
            elif puntos_agente > max(puntos_otros):
                self._recompensa_pendiente += self.REWARD_PERDER_MANO

            # --- NUEVO v5: penalización suave por punto acumulado ---
            self._recompensa_pendiente += (
                puntos_agente * self.REWARD_POR_PUNTO_EN_MANO
            )
        elif self._pleno_jugador == self.agente_idx:
            self._recompensa_pendiente += self.REWARD_GANAR_MANO * 2

    def _juego_terminado(self) -> bool:
        """Determina si la partida ha terminado (algún jugador >= 100 puntos)."""
        return any(p >= self.PUNTUACION_MAXIMA for p in self._puntuacion_historica)

    def _calcular_recompensa_final(self) -> float:
        """Calcula la recompensa de fin de partida basada en la posición final.

        Returns:
            Recompensa según el puesto: +500 (1º), +200 (2º), -200 (3º), -500 (4º).
        """
        # Ordenar jugadores por puntuación (menor es mejor)
        puntuaciones = list(self._puntuacion_historica)
        ranking = sorted(range(4), key=lambda i: puntuaciones[i])

        posicion = ranking.index(self.agente_idx)

        if posicion == 0:
            return self.REWARD_PRIMERO
        elif posicion == 1:
            return self.REWARD_SEGUNDO
        elif posicion == 2:
            return self.REWARD_TERCERO
        else:
            return self.REWARD_CUARTO

    # ------------------------------------------------------------------
    # Construcción del vector de observación (190 dimensiones, v5)
    # ------------------------------------------------------------------

    def _construir_observacion(self) -> np.ndarray:
        """Construye el vector de observación desde la perspectiva del agente."""
        return self._construir_observacion_desde(self.agente_idx)

    def _construir_observacion_desde(self, agente_idx: int) -> np.ndarray:
        """Construye el vector de observación desde la perspectiva de cualquier jugador.

        v12: Permite que los oponentes (PoliticaSB3) reciban la observación completa
        de 220 dimensiones, eliminando el distribution mismatch entre entrenamiento
        e inferencia de snapshots históricos.

        Args:
            agente_idx: Índice del jugador desde cuya perspectiva se construye.

        Returns:
            Array np.float32 de shape (obs_dim,).
        """
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        a = agente_idx

        # --- [0:52] Mano del jugador ---
        for c in self.motor.jugadores[a].mano:
            obs[c.id] = 1.0

        # --- [52:104] Mesa actual ---
        for _, c in self.motor.mesa:
            obs[52 + c.id] = 1.0

        # --- [104:156] Cementerio (solo cartas de bazas ya resueltas) ---
        for i in range(4):
            for c in self.motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        # --- [156:172] Vacíos conocidos (4 jugadores × 4 palos) ---
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            for palo in self._vacios[jug_idx]:
                obs[156 + rel * 4 + palo] = 1.0

        # --- [172:176] Puntajes históricos (normalizados /100) ---
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[172 +
                rel] = min(self._puntuacion_historica[jug_idx] / 100.0, 1.0)

        # --- [176:180] Puntos de la mano actual (normalizados /26) ---
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[176 + rel] = min(self._puntos_mano_actual[jug_idx] / 26.0, 1.0)

        # --- [180] Corazones rotos ---
        obs[180] = 1.0 if self.motor.corazones_rotos else 0.0

        # --- [181] Posición en la baza actual ---
        posiciones = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
        obs[181] = posiciones.get(len(self.motor.mesa), 0.0)

        # --- [182:187] Rastreador de la Dama de Picas ---
        if self._dama_picas_en is None:
            obs[182] = 1.0  # Oculta
        else:
            rel = (self._dama_picas_en - a) % 4
            obs[183 + rel] = 1.0

        # --- [187:190] Features estratégicas v5 ---
        obs[187] = 1.0 if self._pozo_viable(a) else 0.0
        obs[188] = 1.0 if self._debo_arriesgar(a) else 0.0
        obs[189] = 1.0 if self._puedo_alimentar(a) else 0.0

        # --- [190:194] Features all_void v6 ---
        for palo in range(4):
            todos_vacios = all(
                palo in self._vacios[j]
                for j in range(4) if j != a
            )
            obs[190 + palo] = 1.0 if todos_vacios else 0.0

        # --- Bloque v9 [194:220] (solo si obs_dim == 220) ---
        if self._obs_dim >= 220:
            self._construir_bloque_v9(obs, a)

        return obs

    def _construir_bloque_v9(self, obs: np.ndarray, a: int) -> None:
        """Añade los 26 features v9 al vector obs en los índices [194:220]."""
        # [194] baza_numero / 13.0
        obs[194] = min(self.motor.numero_baza / 13.0, 1.0)

        # [195] jugadores_cerca_de_100 / 3.0
        cerca = sum(1 for p in self._puntuacion_historica if p >= 85)
        obs[195] = cerca / 3.0

        # [196] Q♠ ya fue capturada (dama_picas_en conocido → fue jugada)
        obs[196] = 1.0 if self._dama_picas_en is not None else 0.0

        # [197] soy líder en puntaje (tengo el puntaje más bajo)
        mi_pts = self._puntuacion_historica[a]
        obs[197] = 1.0 if all(
            mi_pts <= self._puntuacion_historica[j] for j in range(4)
        ) else 0.0

        # [198] mano terminal posible (algún jugador ≥74 → esta mano puede acabar)
        obs[198] = 1.0 if any(
            p >= 74 for p in self._puntuacion_historica) else 0.0

        # [199:203] cartas restantes por palo (no en cementerio) / 13.0
        cementerio_por_palo = [0, 0, 0, 0]
        for j in range(4):
            for c in self.motor.jugadores[j].bazas_ganadas:
                cementerio_por_palo[c.palo] += 1
        for palo in range(4):
            obs[199 + palo] = max(0.0, (13 - cementerio_por_palo[palo]) / 13.0)

        # [203:207] cartas altas (J/Q/K/A) restantes por palo / 4.0
        altas_cementerio = [0, 0, 0, 0]
        for j in range(4):
            for c in self.motor.jugadores[j].bazas_ganadas:
                if c.valor >= 11:
                    altas_cementerio[c.palo] += 1
        for palo in range(4):
            obs[203 + palo] = max(0.0, (4 - altas_cementerio[palo]) / 4.0)

        # [207:211] probabilidad Q♠ por jugador relativo
        q_prob = self._calcular_prob_q_picas(a)
        for r in range(4):
            obs[207 + r] = q_prob[r]

        # [211:215] corazones capturados ESTA MANO por jugador relativo / 13.0
        for j in range(4):
            rel = (j - a) % 4
            corazones = sum(
                1 for c in self.motor.jugadores[j].bazas_ganadas
                if c.es_corazon
            )
            obs[211 + rel] = min(corazones / 13.0, 1.0)

        # [215:219] alerta pozo: jugador capturó ≥6 corazones esta mano
        for j in range(4):
            rel = (j - a) % 4
            corazones = sum(
                1 for c in self.motor.jugadores[j].bazas_ganadas
                if c.es_corazon
            )
            obs[215 + rel] = 1.0 if corazones >= 6 else 0.0

        # [219] palo_salida: -1.0 si no hay palo de salida, else palo/3.0
        obs[219] = (
            -1.0 if self.motor.palo_de_salida is None
            else self.motor.palo_de_salida / 3.0
        )

    def _calcular_prob_q_picas(self, a: int) -> list:
        """Distribuye probabilidad de Q♠ entre jugadores por eliminación de voids."""
        _PICA = 2

        # Q♠ ya capturada: nadie la tiene en la mano
        if self._dama_picas_en is not None:
            return [0.0, 0.0, 0.0, 0.0]

        # Q♠ está en mi mano
        mi_mano = self.motor.jugadores[a].mano
        if any(c.es_dama_de_picas for c in mi_mano):
            return [1.0, 0.0, 0.0, 0.0]  # posición relativa 0 = yo

        # Q♠ está en la mesa (alguien la acaba de jugar)
        for jug_idx, carta in self.motor.mesa:
            if carta.es_dama_de_picas:
                rel = (jug_idx - a) % 4
                result = [0.0, 0.0, 0.0, 0.0]
                result[rel] = 1.0
                return result

        # Q♠ podría estar en cualquier rival que no sea void en picas
        candidatos = [
            r for r in range(1, 4)
            if _PICA not in self._vacios[(a + r) % 4]
        ]
        if not candidatos:
            return [0.0, 0.0, 0.0, 0.0]
        prob = 1.0 / len(candidatos)
        result = [0.0, 0.0, 0.0, 0.0]
        for r in candidatos:
            result[r] = prob
        return result

    # ------------------------------------------------------------------
    # Features estratégicas v5
    # ------------------------------------------------------------------

    def _pozo_viable(self, agente_idx: Optional[int] = None) -> bool:
        """Determina si es viable intentar shooting the moon.

        Condiciones:
          - Corazones NO rotos
          - ≥6 corazones en mano
          - ≥3 corazones altos (J, Q, K, A)
          - Puntaje histórico < 80 (margen para fallar)
        """
        if self.motor.corazones_rotos:
            return False

        a = agente_idx if agente_idx is not None else self.agente_idx
        mano = self.motor.jugadores[a].mano

        corazones_en_mano = sum(1 for c in mano if c.es_corazon)
        if corazones_en_mano < 6:
            return False

        corazones_altos = sum(
            1 for c in mano
            if c.es_corazon and c.valor >= 11  # J=11, Q=12, K=13, A=14
        )
        if corazones_altos < 3:
            return False

        if self._puntuacion_historica[a] >= 80:
            return False

        return True

    def _debo_arriesgar(self, agente_idx: Optional[int] = None) -> bool:
        """Determina si el jugador está tan atrás que debe arriesgarse.

        Condiciones:
          - Puntaje del jugador > 75
          - Existe al menos un rival con puntaje < 30
        """
        a = agente_idx if agente_idx is not None else self.agente_idx
        if self._puntuacion_historica[a] <= 75:
            return False

        for i in range(4):
            if i != a and self._puntuacion_historica[i] < 30:
                return True
        return False

    def _puedo_alimentar(self, agente_idx: Optional[int] = None) -> bool:
        """Determina si conviene darle puntos a un rival para que pierda.

        Condiciones:
          - Existe un rival con puntaje > 85 (cerca de perder)
          - El jugador tiene puntaje < 70 (margen seguro)
        """
        a = agente_idx if agente_idx is not None else self.agente_idx
        if self._puntuacion_historica[a] >= 70:
            return False

        for i in range(4):
            if i != a and self._puntuacion_historica[i] > 85:
                return True
        return False
