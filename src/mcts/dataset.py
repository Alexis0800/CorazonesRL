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
) -> List[Tuple[np.ndarray, int]]:
    """Genera pares (obs, action) para UNA mano usando PIMC o MCTS como oracle.

    Args:
        seed: Semilla para reproducibilidad.
        agente_idx: Índice del agente (0-3).
        num_mundos: Mundos PIMC por decisión (ignorado si use_mcts=True).
        rollout_tipo: "evasivo", "experto", "mixto" (política del oracle).
        tipo_oponentes: "heuristicos", "experto", "mixto" (oponentes reales).
        use_mcts: Si True, usa MCTS (multi-step) en vez de PIMC (one-step).
        mcts_simulaciones: Simulaciones MCTS por decisión (default 100).

    Returns:
        Lista de tuplas (observación_220, action_id).
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    motor = MotorCorazones()
    motor.repartir()

    obs_builder = ObservacionBuilder(dim=DIM_ENTRENAMIENTO)

    # Oponentes configurables — deben coincidir con el entorno RL
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
            # 50% BotExperto, 50% heurístico (simula entorno RL)
            if random.random() < 0.5:
                bots_oponentes[i] = BotExperto()
            else:
                bots_oponentes[i] = random.choice(bots_disponibles)
        else:  # "heuristicos"
            bots_oponentes[i] = random.choice(bots_disponibles)

    pares: List[Tuple[np.ndarray, int]] = []

    # Jugar 13 bazas
    for _ in range(13):
        for _ in range(4):
            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)

            if len(motor.mesa) == 0:
                # Rastrear vacíos desde la mesa (simplificado)
                pass

            if idx == agente_idx:
                # Turno del agente: consultar PIMC o MCTS
                obs = obs_builder.construir(
                    motor, agente_idx,
                    vacios=[set() for _ in range(4)],
                    puntuacion_historica=[
                        j.puntuacion_historica for j in motor.jugadores],
                    puntos_mano_actual=[
                        j.contar_puntos_bazas() for j in motor.jugadores],
                    dama_picas_en=None,
                )

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
                # Turno de oponente: usar BotExperto
                _jugar_turno_oponente(motor, idx, legales, bots_oponentes)

        motor.resolver_baza()

    return pares


# ──────────────────────────────────────────────────────────────
# Generación batch con multiprocessing
# ──────────────────────────────────────────────────────────────

def _worker_generar_mano(args: Tuple) -> List[Tuple[np.ndarray, int]]:
    """Worker para Pool."""
    seed, num_mundos, rollout_tipo, tipo_oponentes, use_mcts, mcts_sims = args
    return generar_dataset_una_mano(
        seed=seed,
        num_mundos=num_mundos,
        rollout_tipo=rollout_tipo,
        tipo_oponentes=tipo_oponentes,
        use_mcts=use_mcts,
        mcts_simulaciones=mcts_sims,
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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Genera un dataset BC completo usando PIMC o MCTS con multiprocessing.

    Args:
        num_manos: Número de manos a generar.
        num_mundos: Mundos PIMC por decisión.
        rollout_tipo: Política del oracle: "evasivo", "experto", "mixto".
        tipo_oponentes: Oponentes reales: "heuristicos", "experto", "mixto".
        num_workers: Número de procesos paralelos.
        seed: Semilla base.
        use_mcts: Si True, usa MCTS multi-step en vez de PIMC one-step.
        mcts_simulaciones: Simulaciones MCTS por decisión.

    Returns:
        Tuple (observations, actions, metadata).
    """
    t_start = time.time()

    seeds = [seed + i for i in range(num_manos)]
    args = [(s, num_mundos, rollout_tipo, tipo_oponentes, use_mcts, mcts_simulaciones)
            for s in seeds]

    todos_pares: List[Tuple[np.ndarray, int]] = []

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
        prefix + ".npz", observations=observations, actions=actions)
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
