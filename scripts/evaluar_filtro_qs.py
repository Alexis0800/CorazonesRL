"""
A/B pareado del FiltroQS: campeón puro vs campeón con las legales restringidas
por el filtro (igual que producción vía recomendar_jugada). Gate de
NO-REGRESIÓN vs clon (el clon no valida reacciones humanas; el efecto real lo
mide el bridge — ver docs/plan_mejora_vs_humanos_2026-07-25.md Fase 1a).

Uso:
    python scripts/evaluar_filtro_qs.py --modelo models/produccion/v10c_campeon \
        --partidas 3000 --seed-offset 500000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --- bootstrap path ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _jugar(model, env, n_partidas, seed_offset, filtro=None):
    import torch
    from src.dominio.carta import Carta

    resultados = []
    for i in range(n_partidas):
        torch.manual_seed(seed_offset + i)
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        info = {}
        while not done:
            mask = obs["action_mask"]
            if filtro is not None and not env._fase_pase:
                legales = env._motor.obtener_jugadas_legales(env._agente_idx)
                permitidas = filtro.filtrar(env._motor, env._agente_idx, legales)
                if len(permitidas) < len(legales):
                    mask = np.zeros_like(mask)
                    for c in permitidas:
                        mask[c.id] = 1.0
            with torch.no_grad():
                o = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
                logits, _ = model.forward({"obs": {"obs": o, "action_mask": m}}, [], None)
                accion = int(logits.argmax(dim=1).item())
            obs, _, done, _, info = env.step(accion)
        resultados.append(info)
    return resultados


def main() -> None:
    from src.rllib.utils import cargar_policy_desde_checkpoint
    from src.rllib.opponent_pool import SnapshotPolicy
    from src.rllib.eval_bots import _obtener_modelo, _agregar
    from src.entorno.corazones_rllib import CorazonesEnvRLlib
    from src.entorno.dimensiones import DIM_V12, con_pase_de_obs
    from src.agentes.filtro_qs import FiltroQS

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True)
    p.add_argument("--humano", default="models/humano_bc/pesos.npz")
    p.add_argument("--partidas", type=int, default=3000)
    p.add_argument("--obs-dim", type=int, default=DIM_V12)
    p.add_argument("--seed-offset", type=int, default=500000)
    args = p.parse_args()

    policy = cargar_policy_desde_checkpoint(args.modelo)
    model = _obtener_modelo(policy, args.obs_dim)
    d = np.load(args.humano)
    pesos = {k: d[k] for k in d.files}

    def factory(ai=0):
        return {i: SnapshotPolicy.from_weights(pesos, obs_dim=args.obs_dim, temperatura=1.0)
                for i in range(4) if i != ai}

    def make_env():
        return CorazonesEnvRLlib({
            "obs_dim": args.obs_dim, "agente_idx": 0, "random_position": False,
            "opponent_factory": factory, "gamma": 0.999,
            "con_pase": con_pase_de_obs(args.obs_dim),
        })

    res_base = _jugar(model, make_env(), args.partidas, args.seed_offset)
    agg_b = _agregar(res_base)
    print(f"BASELINE: win {agg_b['win']:.3f} top2 {agg_b['top2']:.3f} puesto {agg_b['puesto']:.3f}")

    filtro = FiltroQS()
    res_f = _jugar(model, make_env(), args.partidas, args.seed_offset, filtro=filtro)
    agg_f = _agregar(res_f)
    print(f"FILTRO QS: win {agg_f['win']:.3f} top2 {agg_f['top2']:.3f} puesto {agg_f['puesto']:.3f}")
    print(f"  intervenciones: {filtro.stats['intervenciones']}/{filtro.stats['consultas']} "
          f"({filtro.stats['intervenciones']/max(filtro.stats['consultas'],1)*100:.1f}%)")

    wb = np.array([float(r["gano_partida"]) for r in res_base])
    wf = np.array([float(r["gano_partida"]) for r in res_f])
    dd = wf - wb
    se = float(dd.std(ddof=1) / np.sqrt(len(dd)))
    pb = np.array([r["puesto"] for r in res_base]); pf = np.array([r["puesto"] for r in res_f])
    print(f"\nPAREADO (n={len(dd)}): Δwin {dd.mean():+.4f} ± {se:.4f} (1 SE)  "
          f"Δpuesto {(pf-pb).mean():+.4f}  idénticas: {int((pb==pf).sum())}/{len(dd)}")
    print(json.dumps({"baseline": agg_b, "filtro": agg_f,
                      "delta_win": float(dd.mean()), "se": se}, indent=2))


if __name__ == "__main__":
    main()
