"""
Entorno de Hearts compatible con Ray RLlib.

gym.Env single-agent con:
  - Observation space: Dict({"obs": Box(DIM_ENTORNO,), "action_mask": Box(52,)})
  - Action space: Discrete(52)
  - Un episodio = una mano completa (13 bazas, ~13 pasos del agente).

Los oponentes (3 jugadores) se gestionan dentro del env. Su comportamiento
se controla mediante `opponent_factory` en env_config:

    env_config = {
        "obs_dim": 224,            # dimensión del vector de observación
        "agente_idx": 0,           # posición del agente (0-3)
        "opponent_factory": fn,    # callable() -> dict[int, policy_fn]
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
from src.entorno.recompensas import CalculadoraRecompensas, RewardConfig
from src.agentes.heuristicos import bot_evasivo


PolicyFn = Callable[[MotorCorazones, int, List[Carta]], Carta]
OpponentFactory = Callable[[], Dict[int, PolicyFn]]


class CorazonesEnvRLlib(gym.Env):
    """Entorno de Hearts para Ray RLlib (single-agent, 13 bazas por episodio)."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict = None, **kwargs):
        super().__init__()
        # RLlib puede pasar env_config como dict o como kwargs desempaquetados
        if env_config is None:
            cfg = kwargs
        elif isinstance(env_config, dict):
            cfg = env_config
        else:
            cfg = {}


        self._agente_idx: int = cfg.get("agente_idx", 0)
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
        self._calc = CalculadoraRecompensas(self._reward_config)

        # Estado de episodio
        self._puntuacion_historica: List[int] = [0] * 4
        self._puntos_mano_actual: List[int] = [0] * 4
        self._vacios: List[set] = [set() for _ in range(4)]
        self._dama_picas_en: Optional[int] = None
        self._opponents: Dict[int, PolicyFn] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        # Propagar seed al módulo random para que el reparto sea determinista
        rng_seed = int(self.np_random.integers(0, 2**31))
        _pyrandom.seed(rng_seed)

        self._motor.repartir()
        self._puntos_mano_actual = [0] * 4
        self._vacios = [set() for _ in range(4)]
        self._dama_picas_en = None

        if self._opponent_factory is not None:
            self._opponents = self._opponent_factory()
        else:
            self._opponents = {
                i: bot_evasivo
                for i in range(4) if i != self._agente_idx
            }

        # Avanzar hasta el turno del agente (puede que no salga primero)
        self._auto_step_opponents()

        return self._build_obs(), {}

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        carta = Carta._TODAS[action]
        legales = self._motor.obtener_jugadas_legales(self._agente_idx)

        # Seguridad: nunca debería ocurrir con action masking correcto
        if carta not in legales:
            carta = legales[0]

        self._motor.jugar_carta(self._agente_idx, carta)
        self._actualizar_vacios(self._agente_idx, carta)

        reward = 0.0

        # ¿La jugada del agente completó la baza?
        if len(self._motor.mesa) == 4:
            reward += self._resolver_baza()

        # Avanzar oponentes hasta el próximo turno del agente
        if not self._es_fin_de_mano():
            reward += self._auto_step_opponents()

        # ¿Fin de mano?
        terminated = self._es_fin_de_mano()
        if terminated:
            reward += self._resolver_fin_de_mano()

        return self._build_obs(), reward, terminated, False, {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _auto_step_opponents(self) -> float:
        """Avanza oponentes hasta el turno del agente. Devuelve recompensa acumulada."""
        accumulated = 0.0
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
                accumulated += self._resolver_baza()

        return accumulated

    def _resolver_baza(self) -> float:
        """Resuelve la baza actual y calcula la recompensa para el agente."""
        cartas_en_mesa = [c for _, c in self._motor.mesa]
        numero_baza = self._motor.numero_baza

        # Posición del agente dentro de la mesa (para baza_evitada)
        idx_agente_en_mesa: Optional[int] = None
        for pos, (jug_idx, _) in enumerate(self._motor.mesa):
            if jug_idx == self._agente_idx:
                idx_agente_en_mesa = pos
                break

        ganador = self._motor.resolver_baza()

        # Tracking dama de picas
        if any(c.es_dama_de_picas for c in cartas_en_mesa):
            self._dama_picas_en = ganador

        # Actualizar puntos mano actual
        for i, jug in enumerate(self._motor.jugadores):
            self._puntos_mano_actual[i] = jug.contar_puntos_bazas()

        pozo_viable = self._calcular_pozo_viable()

        if ganador == self._agente_idx:
            return self._calc.recompensa_baza_ganada(
                cartas_baza=cartas_en_mesa,
                agente_idx=self._agente_idx,
                ganador=ganador,
                pozo_viable=pozo_viable,
                numero_baza=numero_baza,
            )
        else:
            return self._calc.recompensa_baza_evitada(
                cartas_baza=cartas_en_mesa,
                agente_idx=self._agente_idx,
                idx_agente_en_mesa=idx_agente_en_mesa,
                numero_baza=numero_baza,
            )

    def _resolver_fin_de_mano(self) -> float:
        """Aplica puntuaciones y devuelve recompensa de fin de mano."""
        puntos_crudos = [j.contar_puntos_bazas() for j in self._motor.jugadores]
        pleno_jugador: Optional[int] = next(
            (i for i, pts in enumerate(puntos_crudos) if pts == 26), None
        )

        puntuaciones = self._motor.aplicar_puntuacion()
        self._puntuacion_historica = [
            self._puntuacion_historica[i] + puntuaciones[i] for i in range(4)
        ]

        reward = self._calc.recompensa_fin_mano(
            agente_idx=self._agente_idx,
            puntuaciones_mano=puntuaciones,
            pleno_jugador=pleno_jugador,
        )

        if pleno_jugador == self._agente_idx:
            reward += self._calc.cfg.REWARD_SHOOTING_MOON

        return reward

    def _es_fin_de_mano(self) -> bool:
        return self._motor.numero_baza > 13 or all(
            len(j.mano) == 0 for j in self._motor.jugadores
        )

    def _actualizar_vacios(self, jugador_idx: int, carta: Carta) -> None:
        """Infiere void cuando un jugador no sigue el palo de salida."""
        if self._motor.mesa and self._motor.palo_de_salida is not None:
            if carta.palo != self._motor.palo_de_salida:
                self._vacios[jugador_idx].add(self._motor.palo_de_salida)

    def _calcular_pozo_viable(self) -> bool:
        """Shooting the moon es viable si alguien acumula ≥6 corazones."""
        return any(
            sum(1 for c in jug.bazas_ganadas if c.es_corazon) >= 6
            for jug in self._motor.jugadores
        )

    def _build_obs(self) -> dict:
        agente = self._agente_idx
        agente_corazones = sum(
            1 for c in self._motor.jugadores[agente].bazas_ganadas if c.es_corazon
        )
        pozo_viable = self._calcular_pozo_viable()
        debo_arriesgar = agente_corazones >= 6
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
            pozo_viable=pozo_viable,
            debo_arriesgar=debo_arriesgar,
            puedo_alimentar=puedo_alimentar,
        )

        mask = np.zeros(52, dtype=np.float32)
        for c in self._motor.obtener_jugadas_legales(agente):
            mask[c.id] = 1.0

        return {"obs": obs_vec, "action_mask": mask}
