"""
Evaluación del agente contra bots fijos — PARTIDAS COMPLETAS vía el ENV (v10).

IMPORTANTE (fix de obs): la evaluación se hace paso a paso a través de
CorazonesEnvRLlib, que construye la observación COMPLETA (scores, voids, moon,
tracker de Q♠…) idéntica a la de entrenamiento. La versión anterior usaba
SnapshotPolicy.construir_desde_motor (obs mínima, features estratégicas en cero),
lo que subestimaba al agente. Ahora agente y rivales-snapshot ven la obs completa.

Métricas alineadas con el objetivo real:
  - win_rate:    fracción de partidas ganadas (1er puesto).
  - top2_rate:   fracción de partidas en top-2.
  - puesto_medio: puesto promedio (1=mejor, 4=peor).
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch

from src.agentes.bot_experto import BotExperto
from src.agentes.bot_castigador import BotCastigador
from src.agentes.bot_lunatico import BotLunatico
from src.agentes.bot_atacante_lider import BotAtacanteLider
from src.agentes.heuristicos import bot_agresivo, bot_conservador, bot_evasivo
from src.entorno.corazones_rllib import CorazonesEnvRLlib
from src.entorno.dimensiones import DIM_ENTORNO


def _obtener_modelo(policy, obs_dim: int):
    """Devuelve el torch model, ya sea de una Ray Policy o de un SnapshotPolicy."""
    from src.rllib.opponent_pool import SnapshotPolicy
    if hasattr(policy, "get_weights"):
        model = SnapshotPolicy(policy, obs_dim=obs_dim)._get_model()
    else:
        model = policy._get_model()
    model.eval()
    return model


def _eval_model_vs_factory(
    model,
    opponent_factory: Callable,
    n_partidas: int,
    obs_dim: int = DIM_ENTORNO,
    agente_idx: int = 0,
    con_pase: bool = False,
    pase_memoria: bool = True,
) -> List[dict]:
    """Juega n partidas completas con `model` como agente, vía el ENV.

    opponent_factory: callable(agente_idx) -> dict{idx: policy_fn} (estilo env).
    pase_memoria: si False, ablaciona los planos v13 (memoria del pase) a cero.
    Devuelve la lista de `info` terminal de cada partida (puesto, gano, etc.).
    """
    env = CorazonesEnvRLlib({
        "obs_dim": obs_dim,
        "agente_idx": agente_idx,
        "random_position": False,
        "opponent_factory": opponent_factory,
        "gamma": 0.999,
        "con_pase": con_pase,
        "pase_memoria": pase_memoria,
    })

    initial = model.get_initial_state()
    es_recurrente = bool(initial)
    resultados: List[dict] = []

    for _ in range(n_partidas):
        obs, _ = env.reset()
        state = [s.unsqueeze(0) for s in initial] if es_recurrente else []
        done = False
        info: dict = {}
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
                logits, new_state = model.forward(
                    {"obs": {"obs": o, "action_mask": m}}, state, None
                )
                if es_recurrente:
                    state = new_state
                accion = int(logits.argmax(dim=1).item())
            obs, _, done, _, info = env.step(accion)
        resultados.append(info)

    return resultados


def _agregar(res: List[dict]) -> dict:
    puestos = np.array([r["puesto"] for r in res])
    return {
        "win": float((puestos == 1).mean()),
        "top2": float((puestos <= 2).mean()),
        "puesto": float(puestos.mean()),
        "moon": float(np.mean([r.get("shooting_moon", False) for r in res])),
        "manos": float(np.mean([r.get("manos_jugadas", 0) for r in res])),
    }


def evaluar_vs_bots(
    policy,
    obs_dim: int = DIM_ENTORNO,
    n_partidas: int = 100,
    agente_idx: int = 0,
    con_pase: bool = False,
    pase_memoria: bool = True,
) -> dict:
    """Evalúa la política jugando partidas completas (vía env) contra bots fijos.

    Escenarios (3 bots del mismo tipo): evasivo / conservador / agresivo / experto.

    Returns:
        Dict con win_rate / top2_rate / puesto_medio por escenario y global,
        más manos_por_partida y moon_rate.
    """
    model = _obtener_modelo(policy, obs_dim)

    def _simple_factory(bot_fn):
        return lambda ai=0: {i: bot_fn for i in range(4) if i != ai}

    def _clase_factory(cls):
        # Instancias frescas por partida (bots con estado se reinician solos).
        return lambda ai=0: {i: cls() for i in range(4) if i != ai}

    def _mixto_factory(ai=0):
        # Mesa de 3 arquetipos DISTINTOS (proxy de juego variado/humano).
        clases = [BotExperto, BotCastigador, BotLunatico, BotAtacanteLider]
        idxs = [i for i in range(4) if i != ai]
        return {i: clases[k]() for k, i in enumerate(idxs)}

    escenarios = {
        "evasivo": _simple_factory(bot_evasivo),
        "conservador": _simple_factory(bot_conservador),
        "agresivo": _simple_factory(bot_agresivo),
        "experto": _clase_factory(BotExperto),
        "castigador": _clase_factory(BotCastigador),
        "lunatico": _clase_factory(BotLunatico),
        "atacante": _clase_factory(BotAtacanteLider),
        "mixto": _mixto_factory,
    }

    resultados: dict = {}
    todos: List[dict] = []
    for nombre, factory in escenarios.items():
        res = _eval_model_vs_factory(model, factory, n_partidas, obs_dim,
                                     agente_idx, con_pase=con_pase,
                                     pase_memoria=pase_memoria)
        ag = _agregar(res)
        resultados[f"win_rate_vs_{nombre}"] = round(ag["win"], 4)
        resultados[f"top2_rate_vs_{nombre}"] = round(ag["top2"], 4)
        resultados[f"puesto_medio_vs_{nombre}"] = round(ag["puesto"], 4)
        todos.extend(res)

    glob = _agregar(todos)
    resultados["win_rate"] = round(glob["win"], 4)
    resultados["top2_rate"] = round(glob["top2"], 4)
    resultados["puesto_medio"] = round(glob["puesto"], 4)
    resultados["manos_por_partida"] = round(glob["manos"], 2)
    resultados["moon_rate"] = round(glob["moon"], 4)
    return resultados
