"""
Genera un dataset de Behavioral Cloning para Hearts.

Juega partidas completas a través de CorazonesEnvRLlib (para obtener la
observación COMPLETA de 224 dims, idéntica a entrenamiento), y etiqueta cada
decisión del agente con la carta que elige PIMC (rollout = BotExperto + voids).
El agente juega la jugada de PIMC (on-policy respecto al maestro), de modo que
el dataset cubre estados a lo largo de trayectorias de juego experto.

Salida: un .npz con arrays obs (N,224), mask (N,52), action (N,).

Uso:
    python generar_dataset_bc.py --partidas 300 --mundos 50 --out datasets/bc_v10.npz
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.entorno.corazones_rllib import CorazonesEnvRLlib
from src.entorno.dimensiones import DIM_ENTORNO
from src.agentes.bot_experto import BotExperto
from src.agentes.bot_castigador import BotCastigador
from src.agentes.bot_lunatico import BotLunatico
from src.agentes.bot_atacante_lider import BotAtacanteLider
from src.agentes.heuristicos import bot_evasivo
from src.mcts.pimc import pimc_mejor_jugada, _crear_bots_experto, _crear_bots_mixto


# Arquetipos humanos diversos para los oponentes durante la generación.
_ARQUETIPOS = [
    lambda: BotExperto(),
    lambda: BotCastigador(),
    lambda: BotLunatico(),
    lambda: BotAtacanteLider(),
    lambda: bot_evasivo,
]


def main() -> None:
    p = argparse.ArgumentParser(description="Generar dataset BC (PIMC)")
    p.add_argument("--partidas", type=int, default=300)
    p.add_argument("--mundos", type=int, default=50)
    p.add_argument("--obs-dim", type=int, default=DIM_ENTORNO)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher", choices=["experto", "mixto"], default="mixto",
                   help="Rollout de PIMC para etiquetar: 'mixto'=arquetipos diversos "
                        "(generaliza), 'experto'=solo BotExperto.")
    p.add_argument("--oponentes", choices=["experto", "diverso"], default="diverso",
                   help="Oponentes del agente durante la generación.")
    p.add_argument("--con-pase", action="store_true",
                   help="Generar con fase de pase (v10b, obs_dim=228). Las decisiones "
                        "de pase se etiquetan con BotExperto.pasar; el juego con PIMC.")
    p.add_argument("--out", type=str, default="datasets/bc_v10.npz")
    args = p.parse_args()
    if args.con_pase:
        from src.entorno.dimensiones import DIM_V12
        args.obs_dim = max(args.obs_dim, DIM_V12)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def factory(ai: int = 0):
        if args.oponentes == "experto":
            return {i: BotExperto() for i in range(4) if i != ai}
        # Diverso: cada rival es un arquetipo distinto al azar (instancia fresca).
        idxs = [i for i in range(4) if i != ai]
        elegidos = rng.integers(0, len(_ARQUETIPOS), size=len(idxs))
        return {i: _ARQUETIPOS[int(e)]() for i, e in zip(idxs, elegidos)}

    def _teacher_rollout():
        return (_crear_bots_experto() if args.teacher == "experto"
                else _crear_bots_mixto(rng))

    env = CorazonesEnvRLlib({
        "obs_dim": args.obs_dim,
        "agente_idx": 0,
        "random_position": False,
        "opponent_factory": factory,
        "gamma": 0.999,
        "con_pase": args.con_pase,
    })
    teacher_pase = BotExperto()  # etiqueta-maestra para las decisiones de pase

    obs_list, mask_list, act_list = [], [], []
    t0 = time.time()
    for g in range(args.partidas):
        obs, _ = env.reset()
        done = False
        pase_pendiente = []  # cartas que el maestro quiere pasar esta mano
        while not done:
            idx = env._agente_idx
            if env._fase_pase:
                # Etiqueta de PASE: las 3 cartas que elige BotExperto.pasar,
                # alimentadas una a una. Recalcular al inicio de cada fase.
                if len(env._pase_seleccion) == 0:
                    pase_pendiente = list(teacher_pase.pasar(env._motor, idx))
                seleccionables = [c for c in env._motor.jugadores[idx].mano
                                  if c not in env._pase_seleccion]
                carta = next((c for c in pase_pendiente if c in seleccionables),
                             seleccionables[0])
                label = int(carta.id)
            else:
                # Etiqueta de JUEGO: la mejor carta según PIMC.
                motor = env._motor
                legales = motor.obtener_jugadas_legales(idx)
                vacios = {i: set(env._vacios[i]) for i in range(4)}
                carta = pimc_mejor_jugada(
                    motor, idx, legales, vacios=vacios,
                    num_mundos=args.mundos, rng=rng,
                    crear_bots=_teacher_rollout,
                )
                label = int(carta.id)

            obs_list.append(obs["obs"].astype(np.float32).copy())
            mask_list.append(obs["action_mask"].astype(np.float32).copy())
            act_list.append(label)
            obs, _, done, _, _ = env.step(label)

        if (g + 1) % max(1, args.partidas // 20) == 0:
            n = len(act_list)
            print(f"  partida {g+1}/{args.partidas}  pares={n}  "
                  f"({(time.time()-t0)/(g+1):.1f}s/partida)", flush=True)

    obs_arr = np.stack(obs_list)
    mask_arr = np.stack(mask_list)
    act_arr = np.array(act_list, dtype=np.int64)
    np.savez_compressed(args.out, obs=obs_arr, mask=mask_arr, action=act_arr)
    print(f"\nDataset guardado: {args.out}")
    print(f"  pares: {len(act_arr):,}  obs: {obs_arr.shape}  "
          f"(de {args.partidas} partidas, mundos={args.mundos})")


if __name__ == "__main__":
    main()
