"""
Entorno de Hearts compatible con Ray RLlib.

gym.Env single-agent con:
  - Observation space: Dict({"obs": Box(DIM_ENTORNO,), "action_mask": Box(52,)})
  - Action space: Discrete(52)
  - Un episodio = una mano completa (13 bazas, ~13 pasos del agente).
  - Recompensa: terminal (26 - puntos_agente) + señal por baza (baza_reward_weight).
    Moon exitoso = +52. Señal por baza se suprime cuando P(Moon) >= 0.5.

Los oponentes (3 jugadores) se gestionan dentro del env. Su comportamiento
se controla mediante `opponent_factory` en env_config:

    env_config = {
        "obs_dim": 224,                # dimensión del vector de observación
        "agente_idx": 0,               # posición fija del agente (0-3)
        "random_position": True,       # si True, agente_idx se sortea cada episodio
        "opponent_factory": fn,        # callable(agente_idx) -> dict[int, policy_fn]
    }

`policy_fn` tiene la firma: (motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta

La factory acepta el agente_idx actual para poder sortear posiciones.
"""
from __future__ import annotations

import random as _pyrandom
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_ENTORNO
from src.entorno.observacion import ObservacionBuilder
from src.entorno.recompensas import CalculadoraRecompensas, RewardConfig
from src.agentes.heuristicos import bot_evasivo


PolicyFn = Callable[[MotorCorazones, int, List[Carta]], Carta]
OpponentFactory = Callable[[int], Dict[int, PolicyFn]]


class CorazonesEnvRLlib(gym.Env):
    """Entorno de Hearts para Ray RLlib (single-agent, 13 bazas por episodio).

    Recompensa terminal-only: reward = 26 - puntos_agente al final de la mano.
    Todas las bazas intermedias devuelven reward = 0.0.

    Si random_position=True, el agente_idx se sortea en cada reset(), lo que
    obliga al modelo a aprender a jugar desde las 4 posiciones de la mesa.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict = None, **kwargs):
        super().__init__()
        if env_config is None:
            cfg = kwargs
        elif isinstance(env_config, dict):
            cfg = env_config
        else:
            cfg = {}

        self._agente_idx: int = cfg.get("agente_idx", 0)
        self._random_position: bool = cfg.get("random_position", False)
        self._obs_dim: int = cfg.get("obs_dim", DIM_ENTORNO)
        self._reward_config: RewardConfig = cfg.get("reward_config", RewardConfig())
        self._opponent_factory: Optional[OpponentFactory] = cfg.get(
            "opponent_factory", None
        )

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(52,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(52)

        self._motor = MotorCorazones()
        self._obs_builder = ObservacionBuilder(dim=self._obs_dim)
        # Mantenemos el calculador para _build_obs() (pozo_viable, SCORE_RIVAL_CERCA)
        self._calc = CalculadoraRecompensas(self._reward_config)

        # Estado de episodio
        self._puntuacion_historica: List[int] = [0] * 4
        self._puntos_mano_actual: List[int] = [0] * 4
        self._vacios: List[set] = [set() for _ in range(4)]
        self._dama_picas_en: Optional[int] = None
        self._opponents: Dict[int, PolicyFn] = {}
        # Puntos acumulados al inicio de la baza actual, para calcular delta por baza
        self._puntos_antes_de_baza: int = 0
        # Peso de la señal por baza relativa al terminal (0 = terminal-only)
        self._baza_reward_weight: float = cfg.get("baza_reward_weight", 0.15)
        # Umbral de P(Moon) para suprimir penalización por baza
        self._moon_prob_threshold: float = cfg.get("moon_prob_threshold", 0.5)
        # v9: acumula recompensas baza-level (Q♠ penalty, moon hearts, K♠ discard)
        # para consumirlas al final de step() sea cual sea la posición del agente en la baza
        self._pending_baza_reward: float = 0.0
        self._ultima_carta_agente = None  # carta que jugó el agente en la baza actual

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        rng_seed = int(self.np_random.integers(0, 2**31))
        _pyrandom.seed(rng_seed)

        # Sortear posición si está habilitado
        if self._random_position:
            self._agente_idx = int(self.np_random.integers(0, 4))

        self._motor.repartir()
        # Reiniciar todo el estado por episodio
        self._puntuacion_historica = [0] * 4
        self._puntos_mano_actual = [0] * 4
        self._vacios = [set() for _ in range(4)]
        self._dama_picas_en = None
        self._pending_baza_reward = 0.0
        self._ultima_carta_agente = None

        if self._opponent_factory is not None:
            # La factory acepta el agente_idx actual para asignar los 3 slots rivales
            self._opponents = self._opponent_factory(self._agente_idx)
        else:
            self._opponents = {
                i: bot_evasivo
                for i in range(4) if i != self._agente_idx
            }

        self._auto_step_opponents()

        return self._build_obs(), {}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        carta = Carta._TODAS[action]
        legales = self._motor.obtener_jugadas_legales(self._agente_idx)

        if carta not in legales:
            carta = legales[0]

        self._ultima_carta_agente = carta  # trackear para recompensas baza-level (v9)
        self._motor.jugar_carta(self._agente_idx, carta)
        self._actualizar_vacios(self._agente_idx, carta)

        baza_cerrada = len(self._motor.mesa) == 4
        if baza_cerrada:
            self._puntos_antes_de_baza = self._motor.jugadores[self._agente_idx].contar_puntos_bazas()
            self._resolver_baza()

        if not self._es_fin_de_mano():
            self._auto_step_opponents()

        terminated = self._es_fin_de_mano()

        if terminated:
            reward = self._calcular_recompensa_terminal()
        elif baza_cerrada and self._baza_reward_weight > 0:
            puntos_ahora = self._motor.jugadores[self._agente_idx].contar_puntos_bazas()
            delta = puntos_ahora - self._puntos_antes_de_baza
            if delta > 0:
                moon_prob = self._calcular_moon_prob(self._agente_idx)
                moon_prob_rival = max(
                    self._calcular_moon_prob(i)
                    for i in range(4) if i != self._agente_idx
                )
                if moon_prob >= self._moon_prob_threshold:
                    # Trayectoria Moon activa — no penalizar; el terminal +52 guía.
                    # Si falla, el terminal (26-pts) lo castigará.
                    reward = 0.0
                elif moon_prob_rival >= 0.7:
                    # Tomar puntos para romper el Moon de un rival es neutral.
                    # El modelo aprende por el terminal (evitar el +26 ajeno).
                    reward = 0.0
                else:
                    # Juego normal: penalizar proporcionalmente, atenuando por moon_prob
                    atenuacion = 1.0 - moon_prob
                    reward = -self._baza_reward_weight * (delta / 13.0) * atenuacion
            else:
                reward = 0.0
        else:
            reward = 0.0

        # Consumir recompensas baza-level acumuladas en _resolver_baza() (v9)
        reward += self._pending_baza_reward
        self._pending_baza_reward = 0.0

        return self._build_obs(), reward, terminated, False, {}

    # ------------------------------------------------------------------
    # Internals — avance del juego (sin recompensas intermedias)
    # ------------------------------------------------------------------

    def _auto_step_opponents(self) -> None:
        """Avanza oponentes hasta el turno del agente."""
        while (
            not self._es_fin_de_mano()
            and self._motor.obtener_jugador_actual() != self._agente_idx
        ):
            idx = self._motor.obtener_jugador_actual()
            legales = self._motor.obtener_jugadas_legales(idx)
            carta = self._opponents.get(idx, bot_evasivo)(self._motor, idx, legales)
            self._motor.jugar_carta(idx, carta)
            self._actualizar_vacios(idx, carta)

            if len(self._motor.mesa) == 4:
                self._resolver_baza()

    def _resolver_baza(self) -> None:
        """Avanza el motor al resolver la baza y acumula recompensas baza-level (v9)."""
        cartas_en_mesa = [c for _, c in self._motor.mesa]
        q_activa_antes = self._dama_picas_en is None  # ¿Q♠ aún no capturada?
        ganador = self._motor.resolver_baza()

        if any(c.es_dama_de_picas for c in cartas_en_mesa):
            self._dama_picas_en = ganador

        for i, jug in enumerate(self._motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

        # --- v9: señales baza-level ---
        cfg = self._reward_config
        if ganador == self._agente_idx:
            moon_prob = self._calcular_moon_prob(self._agente_idx)
            for c in cartas_en_mesa:
                if c.es_dama_de_picas and moon_prob < self._moon_prob_threshold:
                    # Capturó Q♠ sin estar persiguiendo la luna → penalizar
                    self._pending_baza_reward += cfg.Q_SPADES_BAZA_PENALTY
                if c.es_corazon and moon_prob >= self._moon_prob_threshold:
                    # Capturó corazón durante intento Moon activo → incentivar
                    self._pending_baza_reward += cfg.MOON_HEARTS_STEP_REWARD
        elif (self._ultima_carta_agente is not None
              and q_activa_antes
              and self._ultima_carta_agente.palo == 2
              and self._ultima_carta_agente.valor >= 13):
            # Agente jugó K♠/A♠ y no ganó la baza, con Q♠ aún activa.
            # Solo recompensar si la carta era un riesgo real: el agente
            # tenía ≤ 2 espadas en total (K♠/A♠ + a lo sumo una más).
            # Si tenía 5 espadas, K♠ no era inminente — no bonus.
            picas_restantes = sum(
                1 for c in self._motor.jugadores[self._agente_idx].mano
                if c.palo == 2
            )
            if picas_restantes <= 1:
                self._pending_baza_reward += cfg.DESCARTAR_REY_PICAS_REWARD

        self._ultima_carta_agente = None  # consumida para esta baza

    def _calcular_recompensa_terminal(self) -> float:
        """Recompensa final de la mano.

        Fórmula: reward = 26 - puntos_agente
          - Rango normal: [0, 26]  (0 = máximo de puntos, 26 = mano perfecta)
          - Shooting the Moon: 52  (el agente capturó los 26 puntos de penalización)

        Los rivales se llevan 26 pts cada uno cuando hay Moon; el agente queda en 0.
        Detectamos Moon por puntos_crudos antes de llamar a aplicar_puntuacion().
        """
        puntos_crudos = [j.contar_puntos_bazas() for j in self._motor.jugadores]

        if puntos_crudos[self._agente_idx] == 26:
            # Agente hizo Shooting the Moon
            self._motor.aplicar_puntuacion()
            return 52.0

        puntuaciones = self._motor.aplicar_puntuacion()
        self._puntuacion_historica = [
            self._puntuacion_historica[i] + puntuaciones[i] for i in range(4)
        ]
        return 26.0 - float(puntuaciones[self._agente_idx])

    def _es_fin_de_mano(self) -> bool:
        return self._motor.numero_baza > 13 or all(
            len(j.mano) == 0 for j in self._motor.jugadores
        )

    def _actualizar_vacios(self, jugador_idx: int, carta: Carta) -> None:
        """Infiere void cuando un jugador no sigue el palo de salida."""
        if self._motor.mesa and self._motor.palo_de_salida is not None:
            if carta.palo != self._motor.palo_de_salida:
                self._vacios[jugador_idx].add(self._motor.palo_de_salida)

    def _calcular_moon_prob(self, jugador_idx: int) -> float:
        """Probabilidad aproximada [0, 1] de que jugador_idx complete Moon.

        Retorna 0.0 inmediatamente si cualquier rival ya tiene puntos de
        penalización (condición necesaria: Moon requiere los 26 puntos completos).

        Score = control de corazones altos (A K Q J 10) × 0.60
              + Q♠ bajo control × 0.20
              + progreso (hearts ya ganados) × 0.10
              - penalización por hearts aún en manos rivales × 0.10
        """
        for i, jug in enumerate(self._motor.jugadores):
            if i != jugador_idx and jug.contar_puntos_bazas() > 0:
                return 0.0

        jug = self._motor.jugadores[jugador_idx]
        todas = list(jug.mano) + list(jug.bazas_ganadas)

        # Corazones altos: A=14 K=13 Q=12 J=11 10=10
        high_hearts = sum(1 for c in todas if c.es_corazon and c.valor >= 10)
        hearts_ganados = sum(1 for c in jug.bazas_ganadas if c.es_corazon)
        qs_control = any(c.es_dama_de_picas for c in todas)

        # Hearts que aún están en manos rivales (vías de escape del Moon)
        hearts_en_rivales = sum(
            1 for i, jug_r in enumerate(self._motor.jugadores)
            if i != jugador_idx
            for c in jug_r.mano if c.es_corazon
        )

        control  = (high_hearts / 5.0) * 0.60
        qs_bonus = 0.20 if qs_control else 0.0
        progreso = min(hearts_ganados / 13.0, 1.0) * 0.10
        escape   = min(hearts_en_rivales * 0.015, 0.10)

        return max(0.0, min(1.0, control + qs_bonus + progreso - escape))

    def _build_obs(self) -> dict:
        agente = self._agente_idx
        moon_prob_agente = self._calcular_moon_prob(agente)
        moon_prob_rival  = max(
            self._calcular_moon_prob(i) for i in range(4) if i != agente
        )
        puedo_alimentar = any(
            self._puntuacion_historica[j] >= self._calc.cfg.SCORE_RIVAL_CERCA
            for j in range(4) if j != agente
        )

        obs_vec = self._obs_builder.construir(
            motor=self._motor,
            agente_idx=agente,
            vacios=self._vacios,
            puntuacion_historica=self._puntuacion_historica,
            puntos_mano_actual=self._puntos_mano_actual,
            dama_picas_en=self._dama_picas_en,
            moon_prob_agente=moon_prob_agente,
            moon_prob_rival=moon_prob_rival,
            puedo_alimentar=puedo_alimentar,
        )

        mask = np.zeros(52, dtype=np.float32)
        for c in self._motor.obtener_jugadas_legales(agente):
            mask[c.id] = 1.0

        return {"obs": obs_vec, "action_mask": mask}
