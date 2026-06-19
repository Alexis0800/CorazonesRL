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
from src.entorno.recompensas import CalculadoraRecompensas, RewardConfig
from src.entorno.observacion import ObservacionBuilder
from src.entorno.dimensiones import DIM_ENTORNO, DIMS_VALIDAS


class CorazonesEnv(gym.Env):
    """Entorno Gymnasium para el juego de Corazones con Action Masking.

    El agente controla a un solo jugador (por defecto el índice 0).
    Los otros 3 jugadores son controlados por bots aleatorios que
    eligen cartas legales al azar.

    Observation space: Box(187,) de np.float32, normalizado [0, 1].
    Action space: Discrete(52) con enmascaramiento vía action_masks().
    """

    # ------------------------------------------------------------------
    # Recompensas — delegadas a RewardConfig (SSOT en recompensas.py)
    # ------------------------------------------------------------------
    # Acceso de clase para retrocompatibilidad con tests.
    # El SSOT real es RewardConfig; estas referencias se actualizan
    # automáticamente al cambiar los valores en recompensas.py.

    _REWARD_CFG: RewardConfig = RewardConfig()

    REWARD_CORAZON: float = _REWARD_CFG.REWARD_CORAZON
    REWARD_DAMA_PICAS: float = _REWARD_CFG.REWARD_DAMA_PICAS
    REWARD_SHOOTING_MOON: float = _REWARD_CFG.REWARD_SHOOTING_MOON
    REWARD_CORAZON_POZO: float = _REWARD_CFG.REWARD_CORAZON_POZO
    REWARD_PRIMERO: float = _REWARD_CFG.REWARD_PRIMERO
    REWARD_SEGUNDO: float = _REWARD_CFG.REWARD_SEGUNDO
    REWARD_TERCERO: float = _REWARD_CFG.REWARD_TERCERO
    REWARD_CUARTO: float = _REWARD_CFG.REWARD_CUARTO
    REWARD_NO_GANAR_BAZA_CON_PUNTOS: float = _REWARD_CFG.REWARD_NO_GANAR_BAZA_CON_PUNTOS
    REWARD_DESCARTAR_CORAZON_SEGURO: float = _REWARD_CFG.REWARD_DESCARTAR_CORAZON_SEGURO
    REWARD_DESCARTAR_DAMA_SEGURO: float = _REWARD_CFG.REWARD_DESCARTAR_DAMA_SEGURO
    REWARD_GANAR_BAZA_SIN_PUNTOS: float = _REWARD_CFG.REWARD_GANAR_BAZA_SIN_PUNTOS
    REWARD_PERDER_MANO: float = _REWARD_CFG.REWARD_PERDER_MANO
    REWARD_GANAR_MANO: float = _REWARD_CFG.REWARD_GANAR_MANO
    REWARD_Q_SPADES_SIN_POZO: float = _REWARD_CFG.REWARD_Q_SPADES_SIN_POZO
    REWARD_GANAR_BAZA_CON_CORAZON: float = _REWARD_CFG.REWARD_GANAR_BAZA_CON_CORAZON
    REWARD_POR_PUNTO_EN_MANO: float = _REWARD_CFG.REWARD_POR_PUNTO_EN_MANO
    PENALTY_LIDERAR_PICA_CON_Q_ACTIVA: float = _REWARD_CFG.PENALTY_LIDERAR_PICA_CON_Q_ACTIVA
    REWARD_QUEMAR_MAXIMA_PALO_SEGURO: float = _REWARD_CFG.REWARD_QUEMAR_MAXIMA_PALO_SEGURO
    PENALTY_LIDERAR_Q_EQUIVOCADO: float = _REWARD_CFG.PENALTY_LIDERAR_Q_EQUIVOCADO
    REWARD_DUMP_Q_SIGUIENDO_PICAS: float = _REWARD_CFG.REWARD_DUMP_Q_SIGUIENDO_PICAS
    PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE: float = _REWARD_CFG.PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE
    REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA: float = _REWARD_CFG.REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA
    REWARD_QUEMAR_MAXIMA_FORZADA: float = _REWARD_CFG.REWARD_QUEMAR_MAXIMA_FORZADA
    REWARD_QUEMAR_ALTA_SIGUIENDO_PALO: float = _REWARD_CFG.REWARD_QUEMAR_ALTA_SIGUIENDO_PALO
    PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD: float = _REWARD_CFG.PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD
    REWARD_LIDERAR_Q_DUMP_SEGURO: float = _REWARD_CFG.REWARD_LIDERAR_Q_DUMP_SEGURO
    REWARD_DESCARTAR_CORAZON_BAJO_ROTO: float = _REWARD_CFG.REWARD_DESCARTAR_CORAZON_BAJO_ROTO
    BAZA_TARDIA: int = _REWARD_CFG.BAZA_TARDIA
    PUNTUACION_MAXIMA: float = _REWARD_CFG.PUNTUACION_MAXIMA

    @property
    def _cfg(self) -> RewardConfig:
        """Acceso de instancia al RewardConfig (mismo SSOT que los class attrs)."""
        return self._calc.cfg

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

        # Calculadora de recompensas (SSOT — recompensas.py)
        self._calc: CalculadoraRecompensas = CalculadoraRecompensas()

        # Builder de observación (SSOT — observacion.py)
        self._obs_builder: ObservacionBuilder = ObservacionBuilder(dim=obs_dim)

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
        if (posicion_en_baza == 0
                and self._dama_picas_en is None
                and carta.palo == 2
                and not carta.es_dama_de_picas):
            picas_en_mano = [
                c for c in self.motor.jugadores[self.agente_idx].mano
                if c.palo == 2
            ]
            if picas_en_mano and carta.valor < max(c.valor for c in picas_en_mano):
                self._recompensa_pendiente += self._calc.recompensa_liderar_pica(
                    self.agente_idx, carta,
                    self.motor.jugadores[self.agente_idx].mano,
                    posicion_en_baza, self._dama_picas_en is None,
                )

        # --- REWARD: liderar máxima de palo seguro (♣/♦) ---
        if (posicion_en_baza == 0
                and carta.palo in (0, 1)
                and self._es_maxima_en_mano(carta, self.agente_idx)):
            self._recompensa_pendiente += self._calc.recompensa_quemar_maxima_palo_seguro(
                posicion_en_baza, carta, True,
            )

        # --- PENALTY: liderar Q♠ en mal momento ---
        if posicion_en_baza == 0 and carta.es_dama_de_picas:
            soy_maxima_picas = self._es_maxima_en_mano(carta, self.agente_idx)
            self._recompensa_pendiente += self._calc.recompensa_liderar_q_equivocado(
                self.motor.numero_baza, carta, soy_maxima_picas,
            )

        # --- REWARD: liderar Q♠ como dump seguro (baza ≥BAZA_TARDIA, K♠/A♠ en circ.) ---
        if posicion_en_baza == 0 and carta.es_dama_de_picas:
            baza_tardia = self.motor.numero_baza >= self._cfg.BAZA_TARDIA
            soy_maxima_picas = self._es_maxima_en_mano(carta, self.agente_idx)
            if baza_tardia and not soy_maxima_picas:
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
                    self._recompensa_pendiente += self._calc.recompensa_liderar_q_dump(
                        self.motor.numero_baza, carta, soy_maxima_picas,
                        k_en_circulacion or a_en_circulacion,
                    )

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
                self._recompensa_pendiente += self._calc.recompensa_quemar_maxima_forzada(
                    posicion_en_baza, True, carta,
                )

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
                # v12: pasar observación completa y deterministic=False
                # para eliminar distribution mismatch y agregar diversidad
                politica = self._politicas_oponentes[actual]
                try:
                    obs_completa = self._construir_observacion_desde(actual)
                    carta = politica(self.motor, actual, legales,
                                     obs=obs_completa, deterministic=False)
                except TypeError:
                    try:
                        # Fallback: sin deterministic (políticas legacy)
                        carta = politica(self.motor, actual, legales,
                                         obs=obs_completa)
                    except TypeError:
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
        en_modo_pozo = self._pozo_viable()
        if ganador == self.agente_idx:
            r_baza = self._calc.recompensa_baza_ganada(
                cartas_en_mesa, self.agente_idx, ganador,
                pozo_viable=self._pozo_viable(),
                numero_baza=numero_baza,
                en_modo_pozo=en_modo_pozo,
            )
            self._recompensa_pendiente += r_baza
        else:
            r_evitada = self._calc.recompensa_baza_evitada(
                cartas_en_mesa, self.agente_idx, idx_agente_en_mesa,
                numero_baza=numero_baza,
            )
            self._recompensa_pendiente += r_evitada

        # --- v9-v10: recompensas tácticas por baza (phase-gated: solo bazas ≥ BAZA_TARDIA) ---
        if idx_agente_en_mesa is not None and numero_baza >= self._cfg.BAZA_TARDIA:
            carta_agente = cartas_en_mesa[idx_agente_en_mesa]

            # REWARD: quemar alta (A/K) siguiendo palo en baza limpia
            self._recompensa_pendiente += self._calc.recompensa_quemar_alta_siguiendo_palo(
                puntos_baza, carta_agente,
                palo_salida is not None and carta_agente.palo == palo_salida,
            )

            # REWARD: soltar Q♠ siguiendo picas (dump seguro)
            self._recompensa_pendiente += self._calc.recompensa_dump_q_siguiendo_picas(
                carta_agente, ganador != self.agente_idx, palo_salida, numero_baza,
            )

            # REWARD: descartar K♠/A♠ en baza limpia con Q♠ activa
            es_descarte = (
                palo_salida is not None and carta_agente.palo != palo_salida
            )
            self._recompensa_pendiente += self._calc.recompensa_descartar_k_a_picas(
                carta_agente, puntos_baza,
                self._dama_picas_en is None, es_descarte,
            )

            # PENALTY: ganar baza con puntos teniendo cartas perdedoras
            if (ganador == self.agente_idx
                    and puntos_baza > 0
                    and palo_salida is not None
                    and carta_agente.palo == palo_salida
                    and not self._pozo_viable()):
                self._recompensa_pendiente += self._calc.recompensa_ganar_baza_con_puntos_evitable(
                    numero_baza, puntos_baza, self._pozo_viable(),
                )

            # --- v10 (Fase B): penalización ganar baza tardía sin puntos ---
            if (ganador == self.agente_idx
                    and puntos_baza == 0
                    and palo_salida in (0, 1)
                    and carta_agente.palo == palo_salida
                    and self._es_maxima_en_mano(carta_agente, self.agente_idx)
                    and carta_agente.valor >= 13):
                self._recompensa_pendiente += self._calc.recompensa_ganar_baza_tardia(
                    numero_baza, puntos_baza, palo_salida, carta_agente,
                    self._es_maxima_en_mano(carta_agente, self.agente_idx),
                )

            # REWARD: descartar corazón en baza limpia con corazones rotos
            self._recompensa_pendiente += self._calc.recompensa_descartar_corazon_bajo(
                self.motor.corazones_rotos, carta_agente, es_descarte, puntos_baza,
            )

    def _finalizar_mano(self) -> None:
        """Finaliza la mano actual: aplica puntuación, verifica pleno,
        y actualiza puntuaciones históricas."""
        puntos_crudos = [j.contar_puntos_bazas() for j in self.motor.jugadores]
        for i, pts in enumerate(puntos_crudos):
            if pts == 26:
                self._pleno_jugador = i
                if i == self.agente_idx:
                    self._recompensa_pendiente += self._cfg.REWARD_SHOOTING_MOON
                break

        puntuaciones_mano = self.motor.aplicar_puntuacion()

        for i in range(4):
            self._puntuacion_historica[i] = self.motor.jugadores[i].puntuacion_historica

        # Recompensa de fin de mano (delegada a CalculadoraRecompensas)
        r_fin_mano = self._calc.recompensa_fin_mano(
            self.agente_idx, puntuaciones_mano, self._pleno_jugador,
        )
        self._recompensa_pendiente += r_fin_mano

    def _juego_terminado(self) -> bool:
        """Determina si la partida ha terminado (algún jugador >= 100 puntos)."""
        return any(p >= self._cfg.PUNTUACION_MAXIMA for p in self._puntuacion_historica)

    def _calcular_recompensa_final(self) -> float:
        """Calcula la recompensa de fin de partida basada en la posición final.
        Delegada a CalculadoraRecompensas (SSOT)."""
        return self._calc.recompensa_fin_partida(
            self.agente_idx, list(self._puntuacion_historica),
        )

    # ------------------------------------------------------------------
    # Construcción del vector de observación (190 dimensiones, v5)
    # ------------------------------------------------------------------

    def _construir_observacion(self) -> np.ndarray:
        """Construye el vector de observación desde la perspectiva del agente."""
        return self._construir_observacion_desde(self.agente_idx)

    def _construir_observacion_desde(self, agente_idx: int) -> np.ndarray:
        """Construye el vector de observación desde la perspectiva de cualquier jugador.

        Delega en ObservacionBuilder (SSOT en observacion.py). Soporta 194 y 220 dims.

        Args:
            agente_idx: Índice del jugador desde cuya perspectiva se construye.

        Returns:
            Array np.float32 de shape (obs_dim,).
        """
        return self._obs_builder.construir(
            self.motor, agente_idx,
            vacios=self._vacios,
            puntuacion_historica=self._puntuacion_historica,
            puntos_mano_actual=self._puntos_mano_actual,
            dama_picas_en=self._dama_picas_en,
            pozo_viable=self._pozo_viable(agente_idx),
            debo_arriesgar=self._debo_arriesgar(agente_idx),
            puedo_alimentar=self._puedo_alimentar(agente_idx),
        )

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
