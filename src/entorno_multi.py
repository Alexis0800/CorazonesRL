"""
Entorno PettingZoo AEC para el juego de Corazones (Módulo 3).

Implementa la interfaz AEC (Agent Environment Cycle) de PettingZoo
para gestionar 4 jugadores secuenciales. Cada agente recibe una
observación de 190 dimensiones desde su perspectiva y una máscara
de acciones legales.

Envuelve directamente MotorCorazones (Módulo 1) sin depender de
CorazonesEnv (Módulo 2).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from gymnasium import spaces

from src.carta import Carta, _PALOS, _PUNTOS, _ES_CORAZON, _ES_DOS_TREBOL
from src.motor import MotorCorazones


class CorazonesAEC:
    """PettingZoo AEC Environment para Corazones con 4 agentes.

    Atributos:
        agents: Lista de agentes activos en el ciclo actual.
        possible_agents: Lista de todos los agentes posibles.
        max_cycles: Límite máximo de ciclos (None = ilimitado).
    """

    metadata: Dict[str, Any] = {
        "name": "corazones_aec_v0",
        "is_parallelizable": False,
    }

    # ------------------------------------------------------------------
    # Constantes de recompensa (sincronizadas con CorazonesEnv)
    # ------------------------------------------------------------------
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -10.0
    REWARD_SHOOTING_MOON: float = 50.0
    REWARD_PRIMERO: float = 500.0
    REWARD_SEGUNDO: float = 200.0
    REWARD_TERCERO: float = -200.0
    REWARD_CUARTO: float = -500.0

    # Nuevas recompensas v5
    REWARD_Q_SPADES_SIN_POZO: float = -8.0
    REWARD_GANAR_BAZA_CON_CORAZON: float = -3.0
    REWARD_POR_PUNTO_EN_MANO: float = -0.2
    PUNTUACION_MAXIMA: float = 100.0

    def __init__(self) -> None:
        self.possible_agents: List[str] = ["norte", "este", "sur", "oeste"]
        self.agents: List[str] = []
        self.agent_name_mapping: Dict[str, int] = {
            name: i for i, name in enumerate(self.possible_agents)
        }

        # Espacios por agente (v6: 194 dimensiones con all_void)
        self.observation_spaces: Dict[str, spaces.Box] = {
            agent: spaces.Box(low=0.0, high=1.0,
                              shape=(194,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces: Dict[str, spaces.Discrete] = {
            agent: spaces.Discrete(52) for agent in self.possible_agents
        }

        # Motor del juego
        self.motor: MotorCorazones = MotorCorazones()

        # Estado interno
        self._puntuacion_historica: List[int] = [0, 0, 0, 0]
        self._vacios: List[Set[int]] = [set(), set(), set(), set()]
        self._dama_picas_en: Optional[int] = None
        self._puntos_mano_actual: List[int] = [0, 0, 0, 0]
        self._rng: random.Random = random.Random()

        # Control del ciclo AEC
        self._seed: Optional[int] = None
        self._current_agent_idx: int = 0
        self._terminations: Dict[str, bool] = {}
        self._truncations: Dict[str, bool] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self._rewards_since_last: Dict[str, float] = {}
        self._infos: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Métodos de espacio (compatibilidad PettingZoo)
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        """Espacio de observación para un agente."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        """Espacio de acción para un agente."""
        return self.action_spaces[agent]

    # ------------------------------------------------------------------
    # Ciclo de vida AEC
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reinicia el entorno para una nueva partida.

        Args:
            seed: Semilla aleatoria para reproducibilidad.
            options: Opciones adicionales (no utilizadas).
        """
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Reiniciar estado
        self._puntuacion_historica = [0, 0, 0, 0]
        self._vacios = [set(), set(), set(), set()]
        self._dama_picas_en = None
        self._puntos_mano_actual = [0, 0, 0, 0]

        self.agents = list(self.possible_agents)
        self._terminations = {agent: False for agent in self.agents}
        self._truncations = {agent: False for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self._rewards_since_last = {agent: 0.0 for agent in self.agents}
        self._infos = {agent: {} for agent in self.agents}

        # Iniciar primera mano
        self._iniciar_nueva_mano()
        self._current_agent_idx = self.motor.obtener_jugador_actual()

    def step(self, action: Optional[int]) -> None:
        """Ejecuta una acción para el agente actual y avanza al siguiente.

        Args:
            action: Índice de la carta a jugar (0-51), o None si el agente
                    está terminado/truncado.
        """
        current_agent = self.agents[self._current_agent_idx]
        agent_idx = self.agent_name_mapping[current_agent]

        # Si el agente está terminado, no hacer nada
        if self._terminations[current_agent] or self._truncations[current_agent]:
            self._advance_to_next()
            return

        # Si la acción es None (agente ya terminado)
        if action is None:
            self._advance_to_next()
            return

        # Aplicar máscara — si la acción es ilegal, elegir una legal
        mask = self.action_mask(current_agent)
        if not mask[action]:
            legales = [i for i, m in enumerate(mask) if m]
            if legales:
                action = self._rng.choice(legales)
            else:
                self._advance_to_next()
                return

        # Convertir acción a Carta y ejecutar
        carta = Carta._TODAS[action]
        self._ejecutar_jugada(agent_idx, carta)

        # Acumular recompensas de la baza si se resolvió
        if len(self.motor.mesa) == 0:
            # La baza se resolvió dentro de _ejecutar_jugada
            # o se resolverá en el próximo ciclo
            pass

        # Resolver baza si hay 4 cartas
        if len(self.motor.mesa) == 4:
            self._resolver_baza_con_recompensas()

        # Verificar fin de mano
        if self.motor.numero_baza > 13:
            self._finalizar_mano_con_recompensas()
            if self._juego_terminado():
                self._aplicar_recompensas_finales()
                for agent in self.agents:
                    self._terminations[agent] = True
            else:
                self._iniciar_nueva_mano()

        self._advance_to_next()

    def _advance_to_next(self) -> None:
        """Avanza al siguiente agente vivo en el ciclo."""
        if all(self._terminations.values()) or all(self._truncations.values()):
            self._current_agent_idx = 0
            return

        # Calcular quién es el siguiente según el motor
        self._current_agent_idx = self.motor.obtener_jugador_actual()

    def observe(self, agent: str) -> np.ndarray:
        """Retorna la observación desde la perspectiva del agente.

        Args:
            agent: Nombre del agente.

        Returns:
            Array np.float32 de shape (187,).
        """
        return self._construir_observacion(self.agent_name_mapping[agent])

    def action_mask(self, agent: str) -> np.ndarray:
        """Retorna la máscara de acciones legales para un agente.

        Args:
            agent: Nombre del agente.

        Returns:
            Array booleano de shape (52,).
        """
        agent_idx = self.agent_name_mapping[agent]

        if not self.motor._mano_activa:
            return np.zeros(52, dtype=np.bool_)

        if self.motor.obtener_jugador_actual() != agent_idx:
            return np.zeros(52, dtype=np.bool_)

        legales = self.motor.obtener_jugadas_legales(agent_idx)
        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True
        return mask

    def last(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Retorna observación, recompensa, terminado, truncado, info
        para el agente actual.

        Returns:
            Tupla (obs, reward, terminated, truncated, info).
        """
        agent = self.agents[self._current_agent_idx]
        obs = self.observe(agent)
        reward = self._rewards_since_last.get(agent, 0.0)
        # Resetear recompensa pendiente para este agente
        self._rewards_since_last[agent] = 0.0
        terminated = self._terminations.get(agent, False)
        truncated = self._truncations.get(agent, False)
        info = self._infos.get(agent, {})
        return obs, reward, terminated, truncated, info

    def agent_iter(self, max_iter: int = 100000):
        """Generador que produce el siguiente agente en cada paso.

        Args:
            max_iter: Límite máximo de iteraciones para evitar bucles infinitos.

        Yields:
            Nombre del agente al que le toca actuar.
        """
        for _ in range(max_iter):
            if all(self._terminations.values()) or all(self._truncations.values()):
                return
            current = self.agents[self._current_agent_idx]
            yield current

    def render(self) -> None:
        """Renderizado (no implementado para entorno headless)."""
        pass

    def close(self) -> None:
        """Limpieza de recursos."""
        pass

    # ------------------------------------------------------------------
    # Lógica de juego compartida con CorazonesEnv
    # ------------------------------------------------------------------

    def _iniciar_nueva_mano(self) -> None:
        """Inicia una nueva mano."""
        self.motor.repartir()
        self._vacios = [set(), set(), set(), set()]
        self._dama_picas_en = None
        self._puntos_mano_actual = [0, 0, 0, 0]
        for j in self.motor.jugadores:
            j.bazas_ganadas = []

    def _ejecutar_jugada(self, jugador_idx: int, carta: Carta) -> None:
        """Ejecuta una jugada y actualiza tracking de vacíos."""
        if self.motor.mesa and self.motor.palo_de_salida is not None:
            if carta.palo != self.motor.palo_de_salida:
                self._vacios[jugador_idx].add(self.motor.palo_de_salida)
        self.motor.jugar_carta(jugador_idx, carta)

    def _resolver_baza_con_recompensas(self) -> None:
        """Resuelve la baza y acumula recompensas para el ganador."""
        cartas_en_mesa = [c for _, c in self.motor.mesa]
        ganador = self.motor.resolver_baza()

        # Actualizar puntos de la mano
        for i, jug in enumerate(self.motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

        # Tracking dama de picas
        for c in cartas_en_mesa:
            if c.es_dama_de_picas:
                self._dama_picas_en = ganador
                break

        # Recompensas al ganador
        ganador_nombre = self.possible_agents[ganador]
        baza_tiene_corazon = False
        baza_tiene_q_spades = False
        for c in cartas_en_mesa:
            if c.es_corazon:
                self._add_reward(ganador_nombre, self.REWARD_CORAZON)
                baza_tiene_corazon = True
            if c.es_dama_de_picas:
                self._add_reward(ganador_nombre, self.REWARD_DAMA_PICAS)
                baza_tiene_q_spades = True

        # Penalización v5 si pozo no viable
        pozo_ok = self._pozo_viable(ganador)
        if baza_tiene_q_spades and not pozo_ok:
            self._add_reward(ganador_nombre, self.REWARD_Q_SPADES_SIN_POZO)
        if baza_tiene_corazon and not pozo_ok:
            self._add_reward(
                ganador_nombre, self.REWARD_GANAR_BAZA_CON_CORAZON)

    def _finalizar_mano_con_recompensas(self) -> None:
        """Finaliza la mano y aplica recompensa de pleno si corresponde."""
        puntos_crudos = [j.contar_puntos_bazas() for j in self.motor.jugadores]
        for i, pts in enumerate(puntos_crudos):
            if pts == 26:
                nombre = self.possible_agents[i]
                self._add_reward(nombre, self.REWARD_SHOOTING_MOON)
                break

        self.motor.aplicar_puntuacion()
        for i in range(4):
            self._puntuacion_historica[i] = self.motor.jugadores[i].puntuacion_historica
            # Penalización v5 por punto acumulado
            if puntos_crudos[i] > 0:
                nombre = self.possible_agents[i]
                self._add_reward(
                    nombre, self.REWARD_POR_PUNTO_EN_MANO * puntos_crudos[i])

    def _juego_terminado(self) -> bool:
        """True si algún jugador alcanzó 100 puntos."""
        return any(p >= self.PUNTUACION_MAXIMA for p in self._puntuacion_historica)

    def _aplicar_recompensas_finales(self) -> None:
        """Aplica recompensas de fin de partida según la clasificación."""
        puntuaciones = list(self._puntuacion_historica)
        ranking = sorted(range(4), key=lambda i: puntuaciones[i])
        rewards_map = {
            0: self.REWARD_PRIMERO,
            1: self.REWARD_SEGUNDO,
            2: self.REWARD_TERCERO,
            3: self.REWARD_CUARTO,
        }
        for pos, jug_idx in enumerate(ranking):
            nombre = self.possible_agents[jug_idx]
            self._add_reward(nombre, rewards_map[pos])

    def _add_reward(self, agent: str, amount: float) -> None:
        """Acumula recompensa para un agente."""
        self._cumulative_rewards[agent] += amount
        self._rewards_since_last[agent] += amount

    # ------------------------------------------------------------------
    # Construcción del vector de observación (190 dimensiones)
    # ------------------------------------------------------------------

    def _construir_observacion(self, agente_idx: int) -> np.ndarray:
        """Construye el vector de observación desde la perspectiva del agente.

        Misma estructura de 194 dimensiones que CorazonesEnv (v6).
        """
        obs = np.zeros(194, dtype=np.float32)
        a = agente_idx

        # [0:52] Mano del agente
        for c in self.motor.jugadores[a].mano:
            obs[c.id] = 1.0

        # [52:104] Mesa actual
        for _, c in self.motor.mesa:
            obs[52 + c.id] = 1.0

        # [104:156] Cementerio
        for i in range(4):
            for c in self.motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        # [156:172] Vacíos conocidos
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            for palo in self._vacios[jug_idx]:
                obs[156 + rel * 4 + palo] = 1.0

        # [172:176] Puntajes históricos (/100)
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[172 +
                rel] = min(self._puntuacion_historica[jug_idx] / 100.0, 1.0)

        # [176:180] Puntos mano actual (/26)
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[176 + rel] = min(self._puntos_mano_actual[jug_idx] / 26.0, 1.0)

        # [180] Corazones rotos
        obs[180] = 1.0 if self.motor.corazones_rotos else 0.0

        # [181] Posición en la baza
        posiciones = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
        obs[181] = posiciones.get(len(self.motor.mesa), 0.0)

        # [182:187] Dama de Picas
        if self._dama_picas_en is None:
            obs[182] = 1.0
        else:
            rel = (self._dama_picas_en - a) % 4
            obs[183 + rel] = 1.0

        # [187:190] Features estratégicas v5
        obs[187] = 1.0 if self._pozo_viable(agente_idx) else 0.0
        obs[188] = 1.0 if self._debo_arriesgar(agente_idx) else 0.0
        obs[189] = 1.0 if self._puedo_alimentar(agente_idx) else 0.0

        # [190:194] Features all_void v6
        for palo in range(4):
            todos_vacios = all(
                palo in self._vacios[j]
                for j in range(4) if j != agente_idx
            )
            obs[190 + palo] = 1.0 if todos_vacios else 0.0

        return obs

    # ------------------------------------------------------------------
    # Features estratégicas v5
    # ------------------------------------------------------------------

    def _pozo_viable(self, agente_idx: int) -> bool:
        """Determina si es viable intentar shooting the moon."""
        if self.motor.corazones_rotos:
            return False
        mano = self.motor.jugadores[agente_idx].mano
        corazones = [c for c in mano if c.palo == 2]
        altos = sum(1 for c in corazones if c.valor >= 11)
        return (
            len(corazones) >= 6
            and altos >= 3
            and self._puntuacion_historica[agente_idx] < 80
        )

    def _debo_arriesgar(self, agente_idx: int) -> bool:
        """Determina si el agente está tan atrás que debe arriesgarse."""
        mi_score = self._puntuacion_historica[agente_idx]
        return mi_score > 75 and any(
            self._puntuacion_historica[i] < 30 for i in range(4) if i != agente_idx
        )

    def _puedo_alimentar(self, agente_idx: int) -> bool:
        """Determina si puedo darle puntos a un rival que está cerca de 100."""
        mi_score = self._puntuacion_historica[agente_idx]
        for i in range(4):
            if i == agente_idx:
                continue
            if self._puntuacion_historica[i] > 85 and mi_score < 70:
                return True
        return False
