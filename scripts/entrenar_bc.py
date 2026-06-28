"""
Entrena una política por Behavioral Cloning (BC) desde un dataset PIMC-experto.

Entrena un HeartsActionMaskModel (misma arquitectura que la política PPO) para
imitar la carta elegida por PIMC. Guarda los pesos en formato compatible con
SnapshotPolicy.from_weights / policy.set_weights, para poder:
  (a) evaluarlo directamente vía el env, y
  (b) cargarlo como init de PPO (train_rllib --bc-weights).

Al terminar, evalúa el modelo BC vs bots (partidas completas, obs completa).

Uso:
    python entrenar_bc.py --dataset datasets/bc_v10.npz --epochs 15 --out models/bc/bc_v10.pkl
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from gymnasium import spaces
from src.entorno.dimensiones import DIM_ENTORNO
from src.rllib.model import HeartsActionMaskModel


def _build_model(obs_dim: int) -> HeartsActionMaskModel:
    obs_space = spaces.Dict({
        "obs": spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32),
        "action_mask": spaces.Box(0.0, 1.0, shape=(52,), dtype=np.float32),
    })
    model_config = {"fcnet_hiddens": [512, 512, 256],
                    "fcnet_activation": "relu", "vf_share_layers": False}
    return HeartsActionMaskModel(obs_space, spaces.Discrete(52), 52, model_config, "bc")


def main() -> None:
    p = argparse.ArgumentParser(description="Entrenar BC desde dataset PIMC")
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--obs-dim", type=int, default=DIM_ENTORNO)
    p.add_argument("--con-pase", action="store_true",
                   help="Modelo v10b con fase de pase (obs 228; eval con pase).")
    p.add_argument("--out", type=str, default="models/bc/bc_v10.pkl")
    p.add_argument("--eval-partidas", type=int, default=100,
                   help="Partidas por escenario para la eval final (0 = no evaluar)")
    args = p.parse_args()
    if args.con_pase:
        from src.entorno.dimensiones import DIM_V12
        args.obs_dim = max(args.obs_dim, DIM_V12)

    data = np.load(args.dataset)
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    mask = torch.tensor(data["mask"], dtype=torch.float32)
    action = torch.tensor(data["action"], dtype=torch.long)
    N = obs.shape[0]
    print(f"Dataset: {N:,} pares, obs_dim={obs.shape[1]}")

    # Split train/val
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    n_val = int(N * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = _build_model(args.obs_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def _logits(o, m):
        return model.forward({"obs": {"obs": o, "action_mask": m}}, [], None)[0]

    import copy
    mejor_vacc = -1.0
    mejor_state = None
    mejor_epoch = 0
    for ep in range(args.epochs):
        model.train()
        tr_perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(tr_perm), args.batch):
            b = tr_perm[i:i + args.batch]
            logits = _logits(obs[b], mask[b])
            loss = loss_fn(logits, action[b])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(b)
            correct += int((logits.argmax(1) == action[b]).sum())
            tot += len(b)
        # Validación
        model.eval()
        with torch.no_grad():
            vlogits = _logits(obs[val_idx], mask[val_idx])
            vacc = float((vlogits.argmax(1) == action[val_idx]).float().mean()) if n_val else float("nan")
        print(f"  epoch {ep+1:>2}/{args.epochs}  loss {loss_sum/tot:.4f}  "
              f"train_acc {correct/tot:.3f}  val_acc {vacc:.3f}", flush=True)
        # Early-stopping: conservar el mejor modelo por val_acc (evita overfitting)
        if n_val and vacc > mejor_vacc:
            mejor_vacc = vacc
            mejor_state = copy.deepcopy(model.state_dict())
            mejor_epoch = ep + 1

    if mejor_state is not None:
        model.load_state_dict(mejor_state)
        print(f"\nMejor modelo: epoch {mejor_epoch} (val_acc {mejor_vacc:.3f}) — usado para guardar/eval")

    # Guardar pesos (numpy) compatibles con SnapshotPolicy.from_weights
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    with open(args.out, "wb") as f:
        pickle.dump({"weights": weights, "obs_dim": args.obs_dim}, f)
    print(f"\nPesos BC guardados: {args.out}")

    # Eval del modelo BC vs bots (partidas completas, obs completa via env)
    if args.eval_partidas > 0:
        from src.rllib.opponent_pool import SnapshotPolicy
        from src.rllib.eval_bots import evaluar_vs_bots
        snap = SnapshotPolicy.from_weights(weights, obs_dim=args.obs_dim)
        print(f"\nEvaluando BC-solo ({args.eval_partidas} partidas/escenario)...")
        m = evaluar_vs_bots(snap, obs_dim=args.obs_dim,
                            n_partidas=args.eval_partidas, con_pase=args.con_pase)
        print(f"  vs experto:  win {m['win_rate_vs_experto']:.3f}  "
              f"top2 {m['top2_rate_vs_experto']:.3f}  puesto {m['puesto_medio_vs_experto']:.3f}")
        print(f"  GLOBAL:      win {m['win_rate']:.3f}  top2 {m['top2_rate']:.3f}  "
              f"puesto {m['puesto_medio']:.3f}")
        print(f"  (referencia MLP v10 self-play: top2 vs experto ~0.72-0.75)")
        print(f"  (techo PIMC-experto: top2 0.975 / win 0.825)")


if __name__ == "__main__":
    main()
