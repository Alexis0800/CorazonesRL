"""
Compara candidatos (elites + últimos snapshots) de una versión para elegir el
MEJOR de forma robusta, con evaluación PAREADA (mismas semillas = mismos repartos
y rivales para todos), que baja muchísimo la varianza.

Métrica primaria: puesto_medio (1=mejor, 4=peor) sobre la mesa MIXTA (proxy de
juego variado). Reporta también win/top2. (PIMC-regret aparte, ver --pimc.)

Uso:
    python comparar_snapshots.py --dir models/v10c --con-pase --partidas 300
    python comparar_snapshots.py --dir models/v10_bc_ppo --partidas 300
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import glob
import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.entorno.corazones_rllib import CorazonesEnvRLlib
from src.agentes.bot_experto import BotExperto
from src.agentes.bot_castigador import BotCastigador
from src.agentes.bot_lunatico import BotLunatico
from src.agentes.bot_atacante_lider import BotAtacanteLider
from src.agentes.heuristicos import bot_evasivo
from src.rllib.utils import cargar_policy_desde_checkpoint, listar_snapshots


def _panel(tipo: str):
    otros = lambda ai: [i for i in range(4) if i != ai]
    if tipo == "experto":
        return lambda ai=0: {i: BotExperto() for i in otros(ai)}
    if tipo == "mixto":
        clases = [BotExperto, BotCastigador, BotLunatico]
        return lambda ai=0: {i: clases[k]() for k, i in enumerate(otros(ai))}
    if tipo == "duro":  # mesa muy difícil
        clases = [BotExperto, BotExperto, BotCastigador]
        return lambda ai=0: {i: clases[k]() for k, i in enumerate(otros(ai))}
    raise ValueError(tipo)


def _eval_pareada(model, factory, M, obs_dim, con_pase):
    env = CorazonesEnvRLlib({
        "obs_dim": obs_dim, "agente_idx": 0, "random_position": False,
        "opponent_factory": factory, "gamma": 0.999, "con_pase": con_pase})
    initial = model.get_initial_state()
    es_rec = bool(initial)
    puestos = []
    for g in range(M):
        obs, _ = env.reset(seed=g)  # PAREADO: misma semilla → mismo juego
        state = [s.unsqueeze(0) for s in initial] if es_rec else []
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
                logits, ns = model.forward({"obs": {"obs": o, "action_mask": m}}, state, None)
                if es_rec:
                    state = ns
                a = int(logits.argmax(1).item())
            obs, _, done, _, info = env.step(a)
        puestos.append(info["puesto"])
    return np.array(puestos)


def _candidatos(model_dir, ult_snaps):
    cands = []
    elite_dir = os.path.join(model_dir, "elite")
    if os.path.isdir(elite_dir):
        for d in sorted(glob.glob(os.path.join(elite_dir, "elite_*"))):
            if os.path.isdir(d):
                cands.append(("elite/" + os.path.basename(d), d))
    snaps = listar_snapshots(os.path.join(model_dir, "snapshots"))
    for paso, ruta in snaps[-ult_snaps:]:
        cands.append(("snap/" + os.path.basename(ruta), ruta))
    return cands


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--con-pase", action="store_true")
    p.add_argument("--partidas", type=int, default=300)
    p.add_argument("--ult-snaps", type=int, default=5, help="últimos N snapshots a incluir")
    p.add_argument("--panel", default="mixto", choices=["mixto", "experto", "duro"])
    args = p.parse_args()
    if args.con_pase:
        from src.entorno.dimensiones import DIM_V12
        obs_dim = DIM_V12
    else:
        from src.entorno.dimensiones import DIM_ENTORNO
        obs_dim = DIM_ENTORNO

    cands = _candidatos(args.dir, args.ult_snaps)
    if not cands:
        print(f"Sin candidatos en {args.dir}"); return
    factory = _panel(args.panel)
    print(f"Comparando {len(cands)} candidatos de {args.dir}  "
          f"({args.partidas} partidas PAREADAS vs panel '{args.panel}', "
          f"{'CON' if args.con_pase else 'SIN'} pase)\n")

    res = []
    for nombre, ruta in cands:
        try:
            snap = cargar_policy_desde_checkpoint(ruta)
            model = snap._get_model()
            puestos = _eval_pareada(model, factory, args.partidas, obs_dim, args.con_pase)
            res.append((nombre, puestos))
            print(f"  {nombre:<32} puesto {puestos.mean():.3f}  "
                  f"win {(puestos==1).mean():.3f}  top2 {(puestos<=2).mean():.3f}", flush=True)
        except Exception as e:
            print(f"  {nombre:<32} ERROR: {e}")

    res.sort(key=lambda r: r[1].mean())
    print("\n=== RANKING (mejor = menor puesto medio) ===")
    base = res[0][1]
    for nombre, puestos in res:
        # error estándar pareado del puesto medio
        ee = puestos.std() / np.sqrt(len(puestos))
        print(f"  {nombre:<32} puesto {puestos.mean():.3f} ± {ee:.3f}  "
              f"win {(puestos==1).mean():.3f}  top2 {(puestos<=2).mean():.3f}")
    print(f"\n>>> MEJOR: {res[0][0]}  (puesto {res[0][1].mean():.3f})")


if __name__ == "__main__":
    main()
