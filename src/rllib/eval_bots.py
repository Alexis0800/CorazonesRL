"""
Evaluación periódica del agente contra bots fijos — PARTIDAS COMPLETAS (v10).

Provee una métrica absoluta (independiente de los oponentes de entrenamiento)
para diagnosticar si el modelo realmente mejora a lo largo del tiempo.

A diferencia de v9 (que medía puntos por mano aislada), aquí se juegan PARTIDAS
COMPLETAS a 100 puntos y se reportan métricas alineadas con el objetivo real:
  - win_rate:    fracción de partidas ganadas (1º puesto).
  - top2_rate:   fracción de partidas en top-2.
  - puesto_medio: puesto promedio (1=mejor, 4=peor).

Uso en train_rllib.py (cada N snapshots):
    from src.rllib.eval_bots import evaluar_vs_bots
    metrics = evaluar_vs_bots(algo.get_policy(), obs_dim=224, n_partidas=100)
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_agresivo, bot_conservador, bot_evasivo
from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_ENTORNO

LIMITE_PARTIDA = 100


def _fin_de_mano(motor: MotorCorazones) -> bool:
    return motor.numero_baza > 13 or all(
        len(j.mano) == 0 for j in motor.jugadores
    )


def _jugar_partida(snap, opponent_factory, agente_idx: int) -> dict:
    """Juega una PARTIDA COMPLETA y devuelve métricas del agente.

    opponent_factory: callable() -> dict{idx: policy_fn}. Se llama una vez por
    partida; los bots con estado por mano se resetean entre manos si exponen reset().
    """
    motor = MotorCorazones()
    motor.nueva_partida()
    opponents = opponent_factory()
    hizo_pozo = False
    manos = 0

    while not motor.partida_terminada(LIMITE_PARTIDA):
        while not _fin_de_mano(motor):
            jugador_actual = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(jugador_actual)
            if jugador_actual == agente_idx:
                carta = snap(motor, jugador_actual, legales)
            else:
                carta = opponents.get(jugador_actual, bot_evasivo)(
                    motor, jugador_actual, legales
                )
            motor.jugar_carta(jugador_actual, carta)
            if len(motor.mesa) == 4:
                motor.resolver_baza()

        if motor.jugadores[agente_idx].contar_puntos_bazas() == 26:
            hizo_pozo = True
        motor.aplicar_puntuacion()
        manos += 1

        if not motor.partida_terminada(LIMITE_PARTIDA):
            motor.repartir()
            for opp in opponents.values():
                reset = getattr(opp, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        pass

    scores = motor.puntuaciones_historicas()
    mi_score = scores[agente_idx]
    puesto = 1 + sum(1 for i, s in enumerate(scores)
                     if i != agente_idx and s < mi_score)
    return {
        "puesto": puesto,
        "gano": puesto == 1,
        "top2": puesto <= 2,
        "score_final": mi_score,
        "manos": manos,
        "pozo": hizo_pozo,
    }


def evaluar_vs_bots(
    policy,
    obs_dim: int = DIM_ENTORNO,
    n_partidas: int = 100,
    agente_idx: int = 0,
) -> dict:
    """Evalúa la política jugando partidas completas contra bots fijos.

    Escenarios (3 bots del mismo tipo por escenario):
      - evasivo / conservador / agresivo: baseline fácil.
      - experto: 3 BotExperto simultáneos (desafío real).

    Args:
        policy:     Ray Policy (resultado de algo.get_policy()).
        obs_dim:    Dimensión del vector de observación.
        n_partidas: Partidas por escenario (4 escenarios → 4×n partidas totales).
        agente_idx: Posición fija del agente (0 para consistencia entre evals).

    Returns:
        Dict con win_rate / top2_rate / puesto_medio por escenario y global,
        más manos_por_partida y moon_rate.
    """
    from src.rllib.opponent_pool import SnapshotPolicy

    # Acepta tanto una Ray Policy (con get_weights) como un SnapshotPolicy ya
    # construido (callable directo, p.ej. cargado desde un checkpoint).
    if hasattr(policy, "get_weights"):
        snap = SnapshotPolicy(policy, obs_dim=obs_dim)
    else:
        snap = policy

    def _simple_factory(bot_fn):
        return lambda: {i: bot_fn for i in range(4) if i != agente_idx}

    def _experto_factory():
        return {i: BotExperto() for i in range(4) if i != agente_idx}

    escenarios = {
        "evasivo": _simple_factory(bot_evasivo),
        "conservador": _simple_factory(bot_conservador),
        "agresivo": _simple_factory(bot_agresivo),
        "experto": _experto_factory,
    }

    resultados: dict = {}
    todos_puestos: List[int] = []
    todas_victorias: List[bool] = []
    todos_top2: List[bool] = []
    total_pozos = 0
    total_manos = 0
    total_partidas = 0

    for nombre, factory in escenarios.items():
        res = [_jugar_partida(snap, factory, agente_idx) for _ in range(n_partidas)]
        puestos = [r["puesto"] for r in res]
        victorias = [r["gano"] for r in res]
        top2 = [r["top2"] for r in res]

        resultados[f"win_rate_vs_{nombre}"] = round(float(np.mean(victorias)), 4)
        resultados[f"top2_rate_vs_{nombre}"] = round(float(np.mean(top2)), 4)
        resultados[f"puesto_medio_vs_{nombre}"] = round(float(np.mean(puestos)), 4)

        todos_puestos.extend(puestos)
        todas_victorias.extend(victorias)
        todos_top2.extend(top2)
        total_pozos += sum(r["pozo"] for r in res)
        total_manos += sum(r["manos"] for r in res)
        total_partidas += len(res)

    resultados["win_rate"] = round(float(np.mean(todas_victorias)), 4)
    resultados["top2_rate"] = round(float(np.mean(todos_top2)), 4)
    resultados["puesto_medio"] = round(float(np.mean(todos_puestos)), 4)
    resultados["manos_por_partida"] = round(total_manos / total_partidas, 2)
    resultados["moon_rate"] = round(total_pozos / total_partidas, 4)

    return resultados
