"""
Generación de datasets para Behavioral Cloning usando PIMC.

Provee funciones para generar pares (observación, acción_óptima) usando
PIMC como oracle. Soporta multiprocessing para acelerar la generación.

Flujo:
  1. Para cada mano, en cada turno del agente:
     a. Construir observación (ObservacionBuilder, 220-dim)
     b. Ejecutar PIMC para encontrar la mejor jugada
     c. Guardar (obs, action_id)
  2. Guardar como .npz (arrays) + .json (metadata)
"""

from __future__ import annotations

import json
import os
import random
import time
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.observacion import ObservacionBuilder
from src.entorno.dimensiones import DIM_ENTRENAMIENTO
from src.mcts.pimc import (
    pimc_mejor_jugada,
    mcts_mejor_jugada,
    crear_bots_rollout,
)
from src.agentes.heuristicos import bot_evasivo, bot_conservador, bot_agresivo


# ──────────────────────────────────────────────────────────────
# Generación de una mano
# ──────────────────────────────────────────────────────────────

def _jugar_turno_oponente(
    motor: MotorCorazones,
    idx: int,
    legales: List[Carta],
    bots: Dict[int, Callable],
) -> None:
    """Ejecuta el turno de un oponente usando su bot asignado."""
    carta = bots[idx](motor, idx, legales)
    motor.jugar_carta(idx, carta)


def generar_dataset_una_mano(
    seed: int,
    agente_idx: int = 0,
    num_mundos: int = 30,
    rollout_tipo: str = "mixto",
    tipo_oponentes: str = "mixto",
    use_mcts: bool = False,
    mcts_simulaciones: int = 100,
    soft_labels: bool = False,
    multi_agente: bool = False,
) -> List[Tuple[np.ndarray, Any]]:
    """Genera pares (obs, action/scores) para UNA mano usando PIMC o MCTS.

    Args:
        seed: Semilla para reproducibilidad.
        agente_idx: Indice del agente (0-3). Ignorado si multi_agente=True.
        num_mundos: Mundos PIMC por decision (ignorado si use_mcts=True).
        rollout_tipo: "evasivo", "experto", "mixto" (politica del oracle).
        tipo_oponentes: "heuristicos", "experto", "mixto" (oponentes reales).
        use_mcts: Si True, usa MCTS (multi-step) en vez de PIMC (one-step).
        mcts_simulaciones: Simulaciones MCTS por decision (default 100).
        soft_labels: Si True, retorna scores (52,) en vez de action_id.
        multi_agente: Si True, genera para los 4 jugadores (4x datos).

    Returns:
        Lista de tuplas (observacion_220, action_id) o (obs, scores_52)
        si soft_labels=True.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Si multi_agente, generar para cada posicion y concatenar
    if multi_agente:
        todos: List[Tuple[np.ndarray, Any]] = []
        for idx in range(4):
            sub_seed = seed * 4 + idx
            pares = generar_dataset_una_mano(
                seed=sub_seed, agente_idx=idx,
                num_mundos=num_mundos, rollout_tipo=rollout_tipo,
                tipo_oponentes=tipo_oponentes,
                use_mcts=use_mcts, mcts_simulaciones=mcts_simulaciones,
                soft_labels=soft_labels, multi_agente=False,
            )
            todos.extend(pares)
        return todos

    motor = MotorCorazones()
    motor.repartir()

    obs_builder = ObservacionBuilder(dim=DIM_ENTRENAMIENTO)

    # Oponentes configurables
    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import (
        bot_conservador, bot_agresivo, bot_evasivo,
    )

    bots_disponibles = [bot_conservador, bot_agresivo, bot_evasivo]
    bots_oponentes: Dict[int, Any] = {}

    for i in range(4):
        if i == agente_idx:
            continue
        if tipo_oponentes == "experto":
            bots_oponentes[i] = BotExperto()
        elif tipo_oponentes == "mixto":
            if random.random() < 0.5:
                bots_oponentes[i] = BotExperto()
            else:
                bots_oponentes[i] = random.choice(bots_disponibles)
        else:
            bots_oponentes[i] = random.choice(bots_disponibles)

    pares: List[Tuple[np.ndarray, Any]] = []

    # Jugar 13 bazas
    for _ in range(13):
        for _ in range(4):
            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)

            if idx == agente_idx:
                obs = obs_builder.construir(
                    motor, agente_idx,
                    vacios=[set() for _ in range(4)],
                    puntuacion_historica=[
                        j.puntuacion_historica for j in motor.jugadores],
                    puntos_mano_actual=[
                        j.contar_puntos_bazas() for j in motor.jugadores],
                    dama_picas_en=None,
                )

                if soft_labels:
                    # Obtener scores para TODAS las acciones legales
                    scores = _obtener_scores_acciones(
                        motor, agente_idx, legales, obs_builder,
                        num_mundos=num_mundos, rng=rng,
                        rollout_tipo=rollout_tipo,
                        use_mcts=use_mcts,
                        mcts_simulaciones=mcts_simulaciones,
                    )
                    pares.append((obs, scores))
                    # Jugar la mejor accion segun PIMC
                    mejor_idx = np.nanargmin(scores)
                    mejor_carta = Carta._TODAS[mejor_idx]
                else:
                    if use_mcts:
                        mejor_carta = mcts_mejor_jugada(
                            motor, agente_idx, legales,
                            num_simulaciones=mcts_simulaciones,
                            rollout_tipo=rollout_tipo,
                        )
                    else:
                        mejor_carta = pimc_mejor_jugada(
                            motor, agente_idx, legales,
                            num_mundos=num_mundos,
                            rng=rng,
                            crear_bots=lambda: crear_bots_rollout(
                                tipo=rollout_tipo, rng=rng),
                        )
                    pares.append((obs, mejor_carta.id))

                motor.jugar_carta(idx, mejor_carta)
            else:
                _jugar_turno_oponente(motor, idx, legales, bots_oponentes)

        motor.resolver_baza()

    return pares


def _obtener_scores_acciones(
    motor, agente_idx, legales, obs_builder,
    num_mundos: int, rng,
    rollout_tipo: str,
    use_mcts: bool = False,
    mcts_simulaciones: int = 100,
) -> np.ndarray:
    """Calcula el score PIMC/MCTS esperado para CADA accion legal.

    Retorna array (52,) float32 con:
      - score finito para acciones legales (menor = mejor)
      - +inf para acciones ilegales

    Args:
        motor: MotorCorazones en el estado actual.
        agente_idx: Indice del agente.
        legales: Lista de cartas legales.
        obs_builder: ObservacionBuilder.
        num_mundos: Mundos PIMC.
        rng: numpy Generator.
        rollout_tipo: Tipo de rollout.
        use_mcts: Si True, usa MCTS.
        mcts_simulaciones: Simulaciones MCTS.

    Returns:
        Array (52,) float32 con scores.
    """
    scores = np.full(52, np.inf, dtype=np.float32)

    for carta in legales:
        if use_mcts:
            # MCTS: clonar motor, jugar carta, ejecutar MCTS desde ahi
            clon = _clonar_motor_para_score(motor)
            clon.jugar_carta(agente_idx, carta)
            # Simular el resto de la mano con rollout y medir score
            score = _mcts_score_desde_estado(
                clon, agente_idx, mcts_simulaciones, rollout_tipo, rng)
        else:
            # PIMC: simular N mundos desde este estado
            score = _pimc_score_carta(
                motor, agente_idx, carta, num_mundos, rollout_tipo, rng)
        scores[carta.id] = score

    return scores


def _clonar_motor_para_score(motor: MotorCorazones) -> MotorCorazones:
    """Clona el motor para evaluar una accion sin modificar el original."""
    from src.mcts.pimc import _clonar_motor
    return _clonar_motor(motor)


def _pimc_score_carta(
    motor, agente_idx, carta, num_mundos, rollout_tipo, rng
) -> float:
    """Estima el score esperado si se juega `carta` ahora.

    Crea N mundos determinizados, juega la carta, simula el resto
    de la mano, y promedia los puntos del agente.
    """
    from src.mcts.pimc import determinizar, crear_bots_rollout

    total = 0.0
    for _ in range(num_mundos):
        mundo = determinizar(motor, agente_idx, rng=rng)
        mundo.jugar_carta(agente_idx, carta)

        # Simular el resto de la mano
        bots = crear_bots_rollout(tipo=rollout_tipo, rng=rng)
        _simular_resto_mano(mundo, agente_idx, bots)
        total += mundo.calcular_puntuacion_mano()[agente_idx]

    return total / num_mundos


def _mcts_score_desde_estado(
    motor, agente_idx, num_simulaciones, rollout_tipo, rng
) -> float:
    """Estima el score desde el estado actual usando simulaciones MCTS-light.

    Ejecuta num_simulaciones rollouts desde el estado actual y promedia.
    """
    from src.mcts.pimc import crear_bots_rollout
    bots = crear_bots_rollout(tipo=rollout_tipo, rng=rng)

    total = 0.0
    for _ in range(num_simulaciones):
        clon = _clonar_motor_para_score(motor)
        _simular_resto_mano(clon, agente_idx, bots)
        total += clon.calcular_puntuacion_mano()[agente_idx]

    return total / num_simulaciones


def _simular_resto_mano(motor, agente_idx, bots) -> None:
    """Simula el resto de la mano actual con los bots dados."""
    while motor.numero_baza <= 13 and motor._mano_activa:
        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        if not legales:
            break
        carta = bots[idx](motor, idx, legales)
        motor.jugar_carta(idx, carta)
        if len(motor.mesa) == 4:
            motor.resolver_baza()


# ──────────────────────────────────────────────────────────────
# Generación batch con multiprocessing
# ──────────────────────────────────────────────────────────────

def _worker_generar_mano(args: Tuple) -> List[Tuple[np.ndarray, Any]]:
    """Worker para Pool."""
    (seed, num_mundos, rollout_tipo, tipo_oponentes,
     use_mcts, mcts_sims, soft_labels, multi_agente) = args
    return generar_dataset_una_mano(
        seed=seed,
        num_mundos=num_mundos,
        rollout_tipo=rollout_tipo,
        tipo_oponentes=tipo_oponentes,
        use_mcts=use_mcts,
        mcts_simulaciones=mcts_sims,
        soft_labels=soft_labels,
        multi_agente=multi_agente,
    )


def generar_dataset(
    num_manos: int = 100,
    num_mundos: int = 30,
    rollout_tipo: str = "mixto",
    tipo_oponentes: str = "mixto",
    num_workers: int = 4,
    seed: int = 42,
    use_mcts: bool = False,
    mcts_simulaciones: int = 100,
    soft_labels: bool = False,
    multi_agente: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Genera un dataset BC completo usando PIMC o MCTS con multiprocessing.

    Args:
        num_manos: Numero de manos a generar.
        num_mundos: Mundos PIMC por decision.
        rollout_tipo: Politica del oracle: \"evasivo\", \"experto\", \"mixto\".
        tipo_oponentes: Oponentes reales: \"heuristicos\", \"experto\", \"mixto\".
        num_workers: Numero de procesos paralelos.
        seed: Semilla base.
        use_mcts: Si True, usa MCTS multi-step en vez de PIMC one-step.
        mcts_simulaciones: Simulaciones MCTS por decision.
        soft_labels: Si True, genera scores (N,52) en vez de action ids (N,).
        multi_agente: Si True, genera para las 4 posiciones (4x datos).

    Returns:
        Tuple (observations, actions/scores, metadata).
        Si soft_labels=True, actions es (N,52) float32 con scores.
    """
    t_start = time.time()

    seeds = [seed + i for i in range(num_manos)]
    args = [(s, num_mundos, rollout_tipo, tipo_oponentes,
             use_mcts, mcts_simulaciones, soft_labels, multi_agente)
            for s in seeds]

    todos_pares: List[Tuple[np.ndarray, Any]] = []

    if num_workers <= 1:
        for i, arg in enumerate(args):
            print(f"\r  Mano {i+1}/{num_manos}...", end="", flush=True)
            todos_pares.extend(_worker_generar_mano(arg))
        print()
    else:
        with Pool(processes=num_workers) as pool:
            for i, resultado in enumerate(pool.imap_unordered(_worker_generar_mano, args)):
                todos_pares.extend(resultado)
                print(
                    f"\r  Mano {min(i+1, num_manos)}/{num_manos}...", end="", flush=True)
        print()

    # Convertir a arrays numpy
    observations = np.array([p[0] for p in todos_pares], dtype=np.float32)
    if soft_labels:
        # Scores: (N, 52) float32
        actions = np.array([p[1] for p in todos_pares], dtype=np.float32)
    else:
        # Action ids: (N,) int64
        actions = np.array([p[1] for p in todos_pares], dtype=np.int64)

    elapsed = time.time() - t_start

    metadata: Dict[str, Any] = {
        "num_manos": num_manos,
        "num_mundos": num_mundos,
        "rollout_tipo": rollout_tipo,
        "tipo_oponentes": tipo_oponentes,
        "num_workers": num_workers,
        "seed_base": seed,
        "total_pares": len(todos_pares),
        "obs_dim": DIM_ENTRENAMIENTO,
        "soft_labels": soft_labels,
        "multi_agente": multi_agente,
        "use_mcts": use_mcts,
        "tiempo_total_s": round(elapsed, 1),
        "tiempo_por_mano_s": round(elapsed / max(num_manos, 1), 1),
    }

    return observations, actions, metadata


# ──────────────────────────────────────────────────────────────
# Persistencia
# ──────────────────────────────────────────────────────────────

def guardar_dataset(
    prefix: str,
    observations: np.ndarray,
    actions: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """Guarda el dataset como prefix.npz + prefix.json.

    Args:
        prefix: Ruta base (sin extensión).
        observations: Array (N, 220) float32.
        actions: Array (N,) int64.
        metadata: Diccionario con metadata.
    """
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    np.savez_compressed(
        prefix + ".npz",
        observations=observations,
        actions=actions,  # (N,) int64 o (N,52) float32
    )
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def cargar_dataset(prefix: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Carga un dataset desde prefix.npz + prefix.json.

    Args:
        prefix: Ruta base (sin extensión).

    Returns:
        Tuple (observations, actions, metadata).
    """
    data = np.load(prefix + ".npz")
    observations = data["observations"]
    actions = data["actions"]
    with open(prefix + ".json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return observations, actions, metadata


__all__ = [
    "generar_dataset_una_mano",
    "generar_dataset",
    "guardar_dataset",
    "cargar_dataset",
]
