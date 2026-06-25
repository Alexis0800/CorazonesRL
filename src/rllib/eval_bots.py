"""
Evaluación periódica del agente contra bots fijos durante el entrenamiento.

Provee una métrica absoluta (independiente de los oponentes de entrenamiento)
para diagnosticar si el modelo realmente mejora a lo largo del tiempo.

Las métricas de entrenamiento (reward contra pool de self-play) son relativas
al pool actual y no son comparables entre fases. Esta evaluación usa siempre
los mismos 3 bots simples como baseline fijo.

Uso en train_rllib.py (cada N snapshots):
    from src.rllib.eval_bots import evaluar_vs_bots
    metrics = evaluar_vs_bots(algo.get_policy(), obs_dim=224, n_partidas=200)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_agresivo, bot_conservador, bot_evasivo
from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_ENTORNO


def _fin_de_mano(motor: MotorCorazones) -> bool:
    return motor.numero_baza > 13 or all(
        len(j.mano) == 0 for j in motor.jugadores
    )


def _jugar_mano(snap, opponent_factory, agente_idx: int) -> float:
    """Juega una mano completa y retorna el reward del agente.

    opponent_factory: callable() -> dict{idx: policy_fn}
    Se llama una vez por mano para permitir instancias con estado (BotExperto).
    """
    motor = MotorCorazones()
    motor.repartir()

    opponents = opponent_factory()

    while not _fin_de_mano(motor):
        jugador_actual = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(jugador_actual)

        if jugador_actual == agente_idx:
            carta = snap(motor, jugador_actual, legales)
        else:
            fn = opponents.get(jugador_actual, bot_evasivo)
            carta = fn(motor, jugador_actual, legales)

        motor.jugar_carta(jugador_actual, carta)

        if len(motor.mesa) == 4:
            motor.resolver_baza()

    puntos_crudos = [j.contar_puntos_bazas() for j in motor.jugadores]
    if puntos_crudos[agente_idx] == 26:
        motor.aplicar_puntuacion()
        return 52.0

    puntuaciones = motor.aplicar_puntuacion()
    return 26.0 - float(puntuaciones[agente_idx])


def evaluar_vs_bots(
    policy,
    obs_dim: int = DIM_ENTORNO,
    n_partidas: int = 200,
    agente_idx: int = 0,
) -> dict:
    """Evalúa la política actual contra los 3 bots simples.

    Corre en el proceso principal (sin Ray actors), tarda ~5-10 segundos.
    Los 3 oponentes son del mismo tipo por escenario para medir el rendimiento
    en condiciones controladas y comparables entre evaluaciones.

    Escenarios:
      - evasivo / conservador / agresivo: 3 bots del mismo tipo (baseline fácil)
      - experto: 3 BotExperto simultáneos (simula jugadores con conocimiento real)
                 BotExperto tiene estado por mano — se crean instancias frescas
                 en cada partida para evitar contaminación entre manos.

    Args:
        policy:     Ray Policy (resultado de algo.get_policy()).
        obs_dim:    Dimensión del vector de observación.
        n_partidas: Manos por escenario (200 por defecto → 800 partidas totales).
        agente_idx: Posición fija del agente (0 para consistencia entre evals).

    Returns:
        Dict con:
          reward_vs_evasivo / conservador / agresivo / experto — reward medio por escenario
          reward_medio    — promedio global sobre los 4 escenarios (800 partidas)
          moon_rate       — fracción de manos con Shooting the Moon
          pct_reward_gt20 — fracción de manos donde el agente sacó > 20 reward
    """
    from src.rllib.opponent_pool import SnapshotPolicy

    snap = SnapshotPolicy(policy, obs_dim=obs_dim)

    def _simple_factory(bot_fn):
        """Factory para bots sin estado — reutiliza la misma función."""
        return lambda: {i: bot_fn for i in range(4) if i != agente_idx}

    def _experto_factory():
        """Factory para BotExperto — crea instancias frescas cada mano."""
        return {i: BotExperto() for i in range(4) if i != agente_idx}

    escenarios = {
        "evasivo":     _simple_factory(bot_evasivo),
        "conservador": _simple_factory(bot_conservador),
        "agresivo":    _simple_factory(bot_agresivo),
        "experto":     _experto_factory,   # 3 BotExperto simultáneos — desafío real
    }

    todos_rewards: List[float] = []
    todos_puntos: List[float] = []
    total_moons = 0
    resultados: dict = {}

    for nombre, factory in escenarios.items():
        rewards = [_jugar_mano(snap, factory, agente_idx) for _ in range(n_partidas)]
        # Moon → 0 puntos para el agente; resto → 26 - reward
        puntos = [0.0 if r == 52.0 else 26.0 - r for r in rewards]

        resultados[f"reward_vs_{nombre}"] = round(float(np.mean(rewards)), 4)
        resultados[f"avg_pts_vs_{nombre}"] = round(float(np.mean(puntos)), 4)
        todos_rewards.extend(rewards)
        todos_puntos.extend(puntos)
        total_moons += sum(1 for r in rewards if r == 52.0)

    total = len(todos_rewards)
    resultados["reward_medio"]    = round(float(np.mean(todos_rewards)), 4)
    resultados["avg_pts_medio"]   = round(float(np.mean(todos_puntos)), 4)
    resultados["reward_std"]      = round(float(np.std(todos_rewards)), 4)
    resultados["moon_rate"]       = round(total_moons / total, 4)
    resultados["pct_reward_gt20"] = round(sum(r > 20 for r in todos_rewards) / total, 4)

    return resultados
