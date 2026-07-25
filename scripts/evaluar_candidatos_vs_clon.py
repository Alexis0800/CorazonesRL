"""
Fase 1b (plan 2026-07-25): re-evaluar los elite archivados vs el clon humano v3.

Protocolo anti-winner's-curse:
  1. Screening: N partidas PAREADAS (mismas semillas para todos los candidatos)
     → comparables entre sí y con la vara del campeón (0.468, 600 partidas).
  2. Si alguno supera 0.50: confirmar con --partidas 400 --seed-offset 300000
     --solo <nombre> (semillas frescas; el máximo de ~9 con n=200 infla ~4-6pp).

Pareo: torch.manual_seed(seed) ANTES de env.reset(seed) por partida — los
clones muestrean del RNG global de torch (patrón de evaluar_modo_lunar.py).

Uso:
    python scripts/evaluar_candidatos_vs_clon.py                    # screening
    python scripts/evaluar_candidatos_vs_clon.py --partidas 400 \
        --seed-offset 300000 --solo elite_000023732224              # confirmacion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# --- bootstrap path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

sys.stdout.reconfigure(encoding="utf-8")

ELITE_DIR = Path("models/v10c_finetune_pozo/elite")
HUMANO_NPZ = "models/humano_bc/pesos.npz"


def _jugar_partidas(model, env, n_partidas: int, seed_offset: int) -> list:
    """Patrón pareado de evaluar_modo_lunar._jugar_partidas, sin modo."""
    import torch

    resultados = []
    for i in range(n_partidas):
        # CRÍTICO: los clones muestrean de torch.multinomial (RNG global de
        # torch); sin resembrar por partida los brazos no son comparables.
        torch.manual_seed(seed_offset + i)
        obs, _ = env.reset(seed=seed_offset + i)
        done = False
        info: dict = {}
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
                logits, _ = model.forward({"obs": {"obs": o, "action_mask": m}}, [], None)
                accion = int(logits.argmax(dim=1).item())
            obs, _, done, _, info = env.step(accion)
        resultados.append(info)
    return resultados


def main() -> None:
    import numpy as np
    from src.rllib.utils import cargar_policy_desde_checkpoint
    from src.rllib.opponent_pool import SnapshotPolicy
    from src.rllib.eval_bots import _obtener_modelo
    from src.entorno.corazones_rllib import CorazonesEnvRLlib
    from src.entorno.dimensiones import DIM_V12, con_pase_de_obs
    from src.entorno.moon_model import RUTA_MOON

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partidas", type=int, default=200)
    p.add_argument("--seed-offset", type=int, default=200000)
    p.add_argument("--obs-dim", type=int, default=DIM_V12)
    p.add_argument("--solo", default=None,
                   help="Evaluar solo este candidato (para la confirmacion)")
    p.add_argument("--salida", default="models/v10c_finetune_pozo/eval_vs_clon.jsonl")
    args = p.parse_args()

    candidatos = sorted(d for d in ELITE_DIR.iterdir()
                        if d.is_dir() and d.name.startswith("elite_"))
    if args.solo:
        candidatos = [c for c in candidatos if c.name == args.solo]
    if not candidatos:
        print("Sin candidatos"); sys.exit(1)

    d = np.load(HUMANO_NPZ)
    pesos_clon = {k: d[k] for k in d.files}

    def factory(ai=0):
        return {i: SnapshotPolicy.from_weights(pesos_clon, obs_dim=args.obs_dim,
                                               temperatura=1.0)
                for i in range(4) if i != ai}

    env = CorazonesEnvRLlib({
        "obs_dim": args.obs_dim, "agente_idx": 0, "random_position": False,
        "opponent_factory": factory, "gamma": 0.999,
        "con_pase": con_pase_de_obs(args.obs_dim), "moon_dir": RUTA_MOON,
    })

    print(f"{len(candidatos)} candidatos | {args.partidas} partidas | "
          f"semillas {args.seed_offset}+ | vara campeon: 0.468")
    filas = []
    with open(args.salida, "a", encoding="utf-8") as f:
        for cand in candidatos:
            t0 = time.time()
            policy = cargar_policy_desde_checkpoint(str(cand))
            model = _obtener_modelo(policy, args.obs_dim)
            res = _jugar_partidas(model, env, args.partidas, args.seed_offset)
            puestos = np.array([r["puesto"] for r in res])
            fila = {"candidato": cand.name,
                    "win_rate": round(float((puestos == 1).mean()), 4),
                    "top2_rate": round(float((puestos <= 2).mean()), 4),
                    "puesto_medio": round(float(puestos.mean()), 4),
                    "partidas": args.partidas, "seed_offset": args.seed_offset}
            filas.append(fila)
            f.write(json.dumps(fila) + "\n"); f.flush()
            print(f"{cand.name}  win={fila['win_rate']:.4f}  "
                  f"top2={fila['top2_rate']:.4f}  puesto={fila['puesto_medio']:.3f}  "
                  f"({time.time()-t0:.0f}s)")

    filas.sort(key=lambda x: -x["win_rate"])
    print("\n=== Ranking ===")
    for x in filas:
        marca = "  <-- supera 0.50, confirmar con semillas frescas" \
            if x["win_rate"] > 0.50 else ""
        print(f"{x['candidato']}  {x['win_rate']:.4f}{marca}")


if __name__ == "__main__":
    main()
