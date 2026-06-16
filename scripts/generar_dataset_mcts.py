"""
Genera un dataset de (observación 220-dim, acción óptima) usando PIMC.

Ejecuta partidas de Corazones con bots como oponentes. En cada decisión
no trivial del agente (≥2 jugadas legales), aplica PIMC para determinar
la jugada óptima y guarda el par (observación, carta_id).

Uso:
    python scripts/generar_dataset_mcts.py \
        --ejemplos 50000 --mundos 50 \
        --salida datasets/mcts_50k.npz

Formato .npz resultante:
    obs      : float32, shape (N, 220)   ← vector de observación 220-dim
    acciones : int32,   shape (N,)       ← carta_id elegida por PIMC [0-51]
    scores   : float32, shape (N,)       ← puntaje esperado de la acción (PIMC)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set

import numpy as np

# Asegurar que el root del proyecto está en el path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.entorno.single_agent import CorazonesEnv
from src.agentes.heuristicos import bot_evasivo, bot_conservador
from src.mcts.pimc import pimc_mejor_jugada, _puntaje_esperado_por_carta


def _crear_bots_evasivos():
    return {i: bot_evasivo for i in range(4)}


def _crear_bots_conservadores():
    return {i: bot_conservador for i in range(4)}


def generar_dataset(
    n_ejemplos: int,
    num_mundos: int,
    obs_dim: int,
    rollout: str,
    seed: Optional[int],
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera el dataset ejecutando partidas completas.

    Returns:
        (obs_array, acciones_array, scores_array)
    """
    rng = np.random.default_rng(seed)

    crear_bots = _crear_bots_evasivos if rollout == "evasivo" else _crear_bots_conservadores

    # Oponentes del entorno usan bot_evasivo para variedad de estados
    env = CorazonesEnv(
        agente_idx=0,
        politicas_oponentes={1: bot_evasivo, 2: bot_evasivo, 3: bot_evasivo},
        obs_dim=obs_dim,
    )

    obs_list: List[np.ndarray] = []
    acciones_list: List[int] = []
    scores_list: List[float] = []

    partidas = 0
    triviales = 0
    t0 = time.time()

    while len(obs_list) < n_ejemplos:
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done = False

        while not done:
            legales = env.motor.obtener_jugadas_legales(env.agente_idx)

            if len(legales) <= 1:
                # Decisión trivial: única jugada legal
                action = legales[0].id
                triviales += 1
            else:
                # Decisión no trivial: usar PIMC
                vacios: Dict[int, Set[int]] = {
                    i: env._vacios[i]
                    for i in range(4)
                    if i != env.agente_idx and env._vacios[i]
                }

                scores = _puntaje_esperado_por_carta(
                    env.motor,
                    env.agente_idx,
                    legales,
                    vacios=vacios,
                    num_mundos=num_mundos,
                    rng=rng,
                    crear_bots=crear_bots,
                )
                mejor = min(legales, key=lambda c: scores[c.id])
                action = mejor.id

                obs_list.append(obs.copy())
                acciones_list.append(action)
                scores_list.append(float(scores[action]))

                if len(obs_list) >= n_ejemplos:
                    break

            obs, _reward, done, _truncated, _info = env.step(action)

        partidas += 1

        if verbose and partidas % 100 == 0:
            elapsed = time.time() - t0
            rate = len(obs_list) / elapsed if elapsed > 0 else 0
            eta = (n_ejemplos - len(obs_list)) / rate if rate > 0 else float("inf")
            print(
                f"  Partidas: {partidas:,} | Ejemplos: {len(obs_list):,}/{n_ejemplos:,} "
                f"| Triviales saltados: {triviales:,} "
                f"| Rate: {rate:.0f} ej/s | ETA: {eta:.0f}s",
                flush=True,
            )

    env.close()

    obs_array = np.array(obs_list, dtype=np.float32)
    acciones_array = np.array(acciones_list, dtype=np.int32)
    scores_array = np.array(scores_list, dtype=np.float32)

    return obs_array, acciones_array, scores_array


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset MCTS para behavioral cloning"
    )
    parser.add_argument(
        "--ejemplos", type=int, default=50_000,
        help="Número de ejemplos no triviales a generar (default: 50000)"
    )
    parser.add_argument(
        "--mundos", type=int, default=50,
        help="Mundos PIMC por decisión (default: 50). Más mundos = mayor calidad"
    )
    parser.add_argument(
        "--obs-dim", type=int, default=220, choices=[194, 220],
        help="Dimensión del vector de observación (default: 220)"
    )
    parser.add_argument(
        "--rollout", type=str, default="evasivo", choices=["evasivo", "conservador"],
        help="Política de rollout PIMC (default: evasivo)"
    )
    parser.add_argument(
        "--salida", type=str, default="datasets/mcts_dataset.npz",
        help="Ruta del archivo .npz de salida"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Semilla para reproducibilidad"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Mostrar progreso cada 100 partidas"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)

    print(f"Generando {args.ejemplos:,} ejemplos con PIMC ({args.mundos} mundos, rollout={args.rollout})")
    print(f"Observación: {args.obs_dim}-dim | Salida: {args.salida}")

    t0 = time.time()
    obs, acciones, scores = generar_dataset(
        n_ejemplos=args.ejemplos,
        num_mundos=args.mundos,
        obs_dim=args.obs_dim,
        rollout=args.rollout,
        seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    np.savez_compressed(
        args.salida,
        obs=obs,
        acciones=acciones,
        scores=scores,
    )

    print(f"\nDataset guardado en {args.salida}")
    print(f"  obs:      {obs.shape}  dtype={obs.dtype}")
    print(f"  acciones: {acciones.shape}  dtype={acciones.dtype}")
    print(f"  scores:   {scores.shape}  media={scores.mean():.2f} ± {scores.std():.2f}")
    print(f"  Tiempo total: {elapsed:.1f}s ({args.ejemplos / elapsed:.0f} ej/s)")
    print(f"  Distribución de acciones: {len(np.unique(acciones))} cartas distintas usadas")


if __name__ == "__main__":
    main()
