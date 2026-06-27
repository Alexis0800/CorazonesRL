"""
Entorno de Hearts compatible con Ray RLlib — PARTIDA COMPLETA (v10).

gym.Env single-agent con:
  - Observation space: Dict({"obs": Box(DIM_ENTORNO,), "action_mask": Box(52,)})
  - Action space: Discrete(52)
  - Un episodio = una PARTIDA COMPLETA: se juegan manos hasta que un jugador
    alcanza el límite de puntos (100). El marcador persiste entre manos.
  - Recompensa (ver docs/Rediseño_v10_partida_completa.md):
      * R_terminal: por puesto final (1º=+1.0, 2º=+0.3, 3º=−0.3, 4º=−1.0).
      * Shaping PBRS: F = γ·Φ(s') − Φ(s), con Φ sobre el marcador acumulado.
        Garantizado no-farmeable (Ng/Harada/Russell 1999).

Los oponentes (3 jugadores) se gestionan dentro del env. Su comportamiento
se controla mediante `opponent_factory` en env_config:

    env_config = {
        "obs_dim": 224,                # dimensión del vector de observación
        "agente_idx": 0,               # posición fija del agente (0-3)
        "random_position": True,       # si True, agente_idx se sortea cada partida
        "opponent_factory": fn,        # callable(agente_idx) -> dict[int, policy_fn]
        "gamma": 0.999,                # DEBE coincidir con el gamma de PPO (PBRS)
    }

`policy_fn` tiene la firma: (motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta
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
from src.entorno.recompensas_partida import (
    CalculadoraRecompensasPartida,
    RewardConfigPartida,
)
from src.agentes.heuristicos import bot_evasivo


PolicyFn = Callable[[MotorCorazones, int, List[Carta]], Carta]
OpponentFactory = Callable[[int], Dict[int, PolicyFn]]


class CorazonesEnvRLlib(gym.Env):
    """Entorno de Hearts para Ray RLlib (single-agent, PARTIDA COMPLETA).

    El episodio abarca múltiples manos hasta que un jugador llega a 100 puntos.
    Recompensa = shaping PBRS por mano + recompensa terminal por puesto final.

    Si random_position=True, el agente_idx se sortea en cada reset() (= cada
    partida), obligando al modelo a jugar desde las 4 posiciones de la mesa.
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
        self._reward_config: RewardConfigPartida = cfg.get(
            "reward_config", RewardConfigPartida()
        )
        self._opponent_factory: Optional[OpponentFactory] = cfg.get(
            "opponent_factory", None
        )
        # gamma para el shaping PBRS — DEBE coincidir con el de PPO para que
        # la garantía de invarianza de política se mantenga.
        self._gamma: float = cfg.get("gamma", 0.999)

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(52,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(52)

        self._motor = MotorCorazones()
        self._obs_builder = ObservacionBuilder(dim=self._obs_dim)
        self._calc = CalculadoraRecompensasPartida(self._reward_config)

        # Estado por mano (se reinicia al inicio de cada mano)
        self._puntos_mano_actual: List[int] = [0] * 4
        self._vacios: List[set] = [set() for _ in range(4)]
        self._dama_picas_en: Optional[int] = None

        # Estado por partida (se reinicia en reset())
        self._opponents: Dict[int, PolicyFn] = {}
        self._phi_prev: float = 0.0
        self._manos_jugadas: int = 0
        self._agente_hizo_pozo: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        rng_seed = int(self.np_random.integers(0, 2**31))
        _pyrandom.seed(rng_seed)

        if self._random_position:
            self._agente_idx = int(self.np_random.integers(0, 4))

        # Inicia una partida nueva (marcador a cero) y reparte la primera mano.
        self._motor.nueva_partida()
        self._reiniciar_estado_mano()
        self._manos_jugadas = 0
        self._agente_hizo_pozo = False

        # Potencial inicial: Φ([0,0,0,0]) = 0 por construcción.
        self._phi_prev = self._calc.potencial(
            self._motor.puntuaciones_historicas(), self._agente_idx
        )

        # La factory se llama UNA vez por partida → oponentes fijos toda la partida.
        if self._opponent_factory is not None:
            self._opponents = self._opponent_factory(self._agente_idx)
        else:
            self._opponents = {
                i: bot_evasivo for i in range(4) if i != self._agente_idx
            }

        self._auto_step_opponents()
        return self._build_obs(), {}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        carta = Carta._TODAS[action]
        legales = self._motor.obtener_jugadas_legales(self._agente_idx)
        if carta not in legales:
            carta = legales[0]

        self._motor.jugar_carta(self._agente_idx, carta)
        self._actualizar_vacios(self._agente_idx, carta)

        if len(self._motor.mesa) == 4:
            self._resolver_baza()

        # Avanzar oponentes hasta el turno del agente o el fin de la mano.
        if not self._es_fin_de_mano():
            self._auto_step_opponents()

        terminated = False
        info: dict = {}

        if self._es_fin_de_mano():
            # Cerrar la mano: aplicar puntuación al marcador acumulado.
            self._cerrar_mano()

            if self._motor.partida_terminada(self._reward_config.LIMITE_PARTIDA):
                terminated = True
            else:
                # Repartir la siguiente mano (conserva el marcador) y avanzar
                # hasta el turno del agente.
                self._motor.repartir()
                self._reiniciar_estado_mano()
                self._reset_oponentes_por_mano()
                self._auto_step_opponents()

        # --- Recompensa: shaping PBRS (+ R_terminal si terminó la partida) ---
        scores = self._motor.puntuaciones_historicas()
        reward = self._shaping_step(scores, terminated)

        if terminated:
            reward += self._calc.recompensa_terminal(scores, self._agente_idx)
            puesto = self._calc.puesto(scores, self._agente_idx)
            info = {
                "puesto": puesto,
                "gano_partida": puesto == 1,
                "top2": puesto <= 2,
                "scores_finales": list(scores),
                "manos_jugadas": self._manos_jugadas,
                "shooting_moon": self._agente_hizo_pozo,
                "r_terminal": self._calc.recompensa_terminal(scores, self._agente_idx),
            }

        return self._build_obs(), reward, terminated, False, info

    def _shaping_step(self, scores: List[int], terminal: bool) -> float:
        """F = γ·Φ(s') − Φ(s), manteniendo Φ(s) entre steps (telescopaje exacto)."""
        phi_now = 0.0 if terminal else self._calc.potencial(scores, self._agente_idx)
        f = self._gamma * phi_now - self._phi_prev
        self._phi_prev = phi_now
        return f

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reiniciar_estado_mano(self) -> None:
        """Reinicia el estado táctico que solo vive durante una mano."""
        self._puntos_mano_actual = [0] * 4
        self._vacios = [set() for _ in range(4)]
        self._dama_picas_en = None

    def _reset_oponentes_por_mano(self) -> None:
        """Resetea el estado interno de oponentes con estado por mano (BotExperto, etc.)."""
        for opp in self._opponents.values():
            reset = getattr(opp, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass

    def _auto_step_opponents(self) -> None:
        """Avanza oponentes hasta el turno del agente (o fin de mano)."""
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
        """Resuelve la baza y actualiza estado táctico (Q♠, puntos de mano)."""
        cartas_en_mesa = [c for _, c in self._motor.mesa]
        ganador = self._motor.resolver_baza()
        if any(c.es_dama_de_picas for c in cartas_en_mesa):
            self._dama_picas_en = ganador
        for i, jug in enumerate(self._motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

    def _cerrar_mano(self) -> None:
        """Aplica la puntuación de la mano al marcador acumulado."""
        puntos_crudos = [j.contar_puntos_bazas() for j in self._motor.jugadores]
        if puntos_crudos[self._agente_idx] == 26:
            self._agente_hizo_pozo = True
        self._motor.aplicar_puntuacion()
        self._manos_jugadas += 1

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
        """Probabilidad aproximada [0, 1] de que jugador_idx complete Moon."""
        for i, jug in enumerate(self._motor.jugadores):
            if i != jugador_idx and jug.contar_puntos_bazas() > 0:
                return 0.0

        jug = self._motor.jugadores[jugador_idx]
        todas = list(jug.mano) + list(jug.bazas_ganadas)

        high_hearts = sum(1 for c in todas if c.es_corazon and c.valor >= 10)
        hearts_ganados = sum(1 for c in jug.bazas_ganadas if c.es_corazon)
        qs_control = any(c.es_dama_de_picas for c in todas)
        hearts_en_rivales = sum(
            1 for i, jug_r in enumerate(self._motor.jugadores)
            if i != jugador_idx
            for c in jug_r.mano if c.es_corazon
        )

        control = (high_hearts / 5.0) * 0.60
        qs_bonus = 0.20 if qs_control else 0.0
        progreso = min(hearts_ganados / 13.0, 1.0) * 0.10
        escape = min(hearts_en_rivales * 0.015, 0.10)
        return max(0.0, min(1.0, control + qs_bonus + progreso - escape))

    def _build_obs(self) -> dict:
        agente = self._agente_idx
        # El marcador histórico ahora ESTÁ VIVO (persiste entre manos).
        puntuacion_historica = self._motor.puntuaciones_historicas()
        moon_prob_agente = self._calcular_moon_prob(agente)
        moon_prob_rival = max(
            self._calcular_moon_prob(i) for i in range(4) if i != agente
        )
        puedo_alimentar = any(
            puntuacion_historica[j] >= 85
            for j in range(4) if j != agente
        )

        obs_vec = self._obs_builder.construir(
            motor=self._motor,
            agente_idx=agente,
            vacios=self._vacios,
            puntuacion_historica=puntuacion_historica,
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
