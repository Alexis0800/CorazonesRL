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

from src.carta import Carta, _PALOS, _PUNTOS, _ES_CORAZON, _ES_DOS_TREBOL
from src.motor import MotorCorazones


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
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -13.0
    REWARD_SHOOTING_MOON: float = 50.0
    REWARD_PRIMERO: float = 1000.0
    REWARD_SEGUNDO: float = 300.0
    REWARD_TERCERO: float = -300.0
    REWARD_CUARTO: float = -1000.0

    PUNTUACION_MAXIMA: float = 100.0  # Umbral de fin de partida

    def __init__(
        self,
        agente_idx: int = 0,
        politicas_oponentes: Optional[Dict[int, object]] = None,
    ) -> None:
        super().__init__()

        if not (0 <= agente_idx <= 3):
            raise ValueError(
                f"agente_idx debe estar entre 0 y 3, recibido {agente_idx}")

        self.agente_idx: int = agente_idx

        # Políticas de oponentes: dict jugador_idx → callable(motor, idx, legales) → Carta
        # Si no se especifica, se usa selección aleatoria
        self._politicas_oponentes: Dict[int, object] = politicas_oponentes or {}

        # Espacios Gymnasium
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(187,), dtype=np.float32
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

        # Ejecutar la jugada del agente
        self._ejecutar_jugada(self.agente_idx, carta)

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
                carta = self._politicas_oponentes[actual](
                    self.motor, actual, legales
                )
            else:
                carta = self._rng.choice(legales)
            self._ejecutar_jugada(actual, carta)

    def _resolver_baza_actual(self) -> None:
        """Resuelve la baza, asigna recompensas al agente y actualiza
        el tracking de la Dama de Picas."""
        # Capturar las cartas de la mesa ANTES de resolver
        cartas_en_mesa = [c for _, c in self.motor.mesa]

        ganador = self.motor.resolver_baza()

        # Actualizar puntos de la mano actual
        for i, jug in enumerate(self.motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

        # Tracking de la Dama de Picas
        for c in cartas_en_mesa:
            if c.es_dama_de_picas:
                self._dama_picas_en = ganador
                break

        # Calcular recompensa para el agente por esta baza
        if ganador == self.agente_idx:
            for c in cartas_en_mesa:
                if c.es_corazon:
                    self._recompensa_pendiente += self.REWARD_CORAZON
                if c.es_dama_de_picas:
                    self._recompensa_pendiente += self.REWARD_DAMA_PICAS

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
        self.motor.aplicar_puntuacion()

        # Actualizar puntuación histórica desde el motor
        for i in range(4):
            self._puntuacion_historica[i] = self.motor.jugadores[i].puntuacion_historica

    def _juego_terminado(self) -> bool:
        """Determina si la partida ha terminado (algún jugador >= 100 puntos)."""
        return any(p >= self.PUNTUACION_MAXIMA for p in self._puntuacion_historica)

    def _calcular_recompensa_final(self) -> float:
        """Calcula la recompensa de fin de partida basada en la posición final.

        Returns:
            Recompensa según el puesto: +1000 (1º), +300 (2º), -300 (3º), -1000 (4º).
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
    # Construcción del vector de observación (187 dimensiones)
    # ------------------------------------------------------------------

    def _construir_observacion(self) -> np.ndarray:
        """Construye el vector de observación de 187 dimensiones.

        Bloques:
            [0:52]    Mano del agente (one-hot)
            [52:104]  Mesa actual / baza en curso (one-hot)
            [104:156] Cementerio / cartas jugadas en bazas anteriores (one-hot)
            [156:172] Vacíos conocidos (4 jugadores × 4 palos)
            [172:176] Puntajes históricos globales (normalizados /100)
            [176:180] Puntos acumulados en la mano actual (normalizados /26)
            [180]     Corazones rotos (0.0 o 1.0)
            [181]     Posición en la baza actual (0.0, 0.33, 0.66, 1.0)
            [182:187] Rastreador de la Dama de Picas (one-hot, 5 estados)

        Returns:
            Array np.float32 de shape (187,).
        """
        obs = np.zeros(187, dtype=np.float32)
        a = self.agente_idx

        # --- [0:52] Mano del agente ---
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
        # Cinco estados mutuamente excluyentes
        if self._dama_picas_en is None:
            obs[182] = 1.0  # Oculta
        else:
            rel = (self._dama_picas_en - a) % 4
            obs[183 + rel] = 1.0

        return obs
