"""
Evalúa un modelo jugando partidas completas contra 3 bots de IMITACIÓN HUMANA
(scripts/entrenar_bc_humano.py) — el proxy más cercano a los humanos reales que
enfrentamos (campeón v10c: ~0.46–0.52 según lote de semillas, vs 24% real, vs
92% bots simples). Ver docs/auditoria_moon_2026-07-20.md.

⚠ Esta eval NO está pareada: los clones muestrean del RNG global de torch, así
que comparar dos corridas sueltas tiene un piso de ruido de ~±3pp — del orden
de los efectos que se suelen buscar. Para medir un DELTA entre dos políticas,
usar el patrón pareado de scripts/evaluar_modo_lunar.py (torch.manual_seed por
partida + delta pareado). Este script sirve para números absolutos gruesos.

Uso:
    python scripts/evaluar_vs_humano_bc.py --modelo models/produccion/v10c_campeon \
        --humano models/humano_bc/pesos.npz --partidas 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --- bootstrap path ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---


def main() -> None:
    from src.rllib.utils import cargar_policy_desde_checkpoint
    from src.rllib.opponent_pool import SnapshotPolicy
    from src.rllib.eval_bots import _eval_model_vs_factory, _obtener_modelo, _agregar
    from src.entorno.dimensiones import DIM_V12, con_pase_de_obs

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True, help="Checkpoint RLlib a evaluar")
    p.add_argument("--humano", default="models/humano_bc/pesos.npz")
    p.add_argument("--partidas", type=int, default=300)
    p.add_argument("--obs-dim", type=int, default=DIM_V12)
    args = p.parse_args()

    policy = cargar_policy_desde_checkpoint(args.modelo)
    model = _obtener_modelo(policy, args.obs_dim)
    d = np.load(args.humano)
    pesos = {k: d[k] for k in d.files}

    def factory(ai=0):
        return {i: SnapshotPolicy.from_weights(pesos, obs_dim=args.obs_dim)
                for i in range(4) if i != ai}

    res = _eval_model_vs_factory(model, factory, args.partidas, obs_dim=args.obs_dim,
                                 con_pase=con_pase_de_obs(args.obs_dim))
    agg = _agregar(res)
    print(f"\n=== {args.modelo} vs 3 bots humanos ({args.partidas} partidas) ===")
    print(f"  win_rate:    {agg['win']:.3f}  (campeón base: 0.46-0.52 según lote; no pareado, ±3pp)")
    print(f"  top2_rate:   {agg['top2']:.3f}")
    print(f"  puesto_medio:{agg['puesto']:.3f}")
    print(f"  moon (agente dispara): {agg['moon']:.3f}")
    print(f"  manos/partida: {agg['manos']:.2f}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
