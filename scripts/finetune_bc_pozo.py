"""
Corrección supervisada dirigida: parte de un checkpoint YA entrenado (no
desde cero, a diferencia de entrenar_bc.py) y hace pocas épocas a LR bajo
sobre el dataset de `generar_dataset_pozo.py`, etiquetado por el oráculo PIMC.

`--peso-lider` sobre-pesa las decisiones de LIDERAZGO en la pérdida sin
excluir el resto del dataset (`--filtro todos` en generar_dataset_pozo.py) --
el análisis 2026-07-06 mostró que liderar tiene 22x más regret alto en manos
reales que en estados simulados (vs bots), pero reaccionar también tiene
brecha (9x); filtrar a solo liderazgo repite la sobre-especialización que ya
vimos con el dataset filtrado a solo pozo (mejoraba pozo, empeoraba el resto).

Uso:
    python scripts/finetune_bc_pozo.py \
        --dataset datasets/bc_correccion_train.npz \
        --checkpoint-base models/v10c_finetune_pozo/snapshots/snapshot_000025001984 \
        --epochs 40 --lr 1e-4 --peso-lider 2.5 \
        --out models/v10c_correccion_lider
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import copy
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
from src.rllib.model import HeartsActionMaskModel
from src.rllib.utils import cargar_policy_desde_checkpoint


def _build_model(obs_dim: int) -> HeartsActionMaskModel:
    obs_space = spaces.Dict({
        "obs": spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32),
        "action_mask": spaces.Box(0.0, 1.0, shape=(52,), dtype=np.float32),
    })
    model_config = {"fcnet_hiddens": [512, 512, 256],
                    "fcnet_activation": "relu", "vf_share_layers": False}
    return HeartsActionMaskModel(obs_space, spaces.Discrete(52), 52, model_config, "bc_pozo")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint-base", required=True,
                   help="Checkpoint RLlib del que partir (pesos iniciales, no entrenar desde cero)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--peso-lider", type=float, default=1.0,
                   help="Peso relativo de las decisiones de liderazgo en la pérdida "
                        "(1.0 = sin sobre-peso). Requiere `es_lider` en el dataset.")
    p.add_argument("--out", required=True,
                   help="Directorio de checkpoint de salida (formato liviano: "
                        "solo policies/default_policy/policy_state.pkl -- suficiente "
                        "para Recomendador/pimc_regret_real.py, no para algo.restore())")
    args = p.parse_args()

    data = np.load(args.dataset)
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    mask = torch.tensor(data["mask"], dtype=torch.float32)
    action = torch.tensor(data["action"], dtype=torch.long)
    es_lider = torch.tensor(data["es_lider"], dtype=torch.bool) if "es_lider" in data else torch.zeros(len(action), dtype=torch.bool)
    peso = torch.where(es_lider, torch.tensor(args.peso_lider), torch.tensor(1.0))
    N = obs.shape[0]
    obs_dim = obs.shape[1]
    print(f"Dataset: {N} pares, obs_dim={obs_dim}, {int(es_lider.sum())} liderando "
          f"(peso {args.peso_lider}x)")

    base = cargar_policy_desde_checkpoint(args.checkpoint_base)
    assert base._obs_dim == obs_dim, f"obs_dim del checkpoint ({base._obs_dim}) != dataset ({obs_dim})"

    model = _build_model(obs_dim)
    model.load_state_dict({k: torch.tensor(v) for k, v in base._weights.items()}, strict=True)
    pesos_iniciales = copy.deepcopy(model.state_dict())

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    n_val = max(1, int(N * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    def _logits(o, m):
        return model.forward({"obs": {"obs": o, "action_mask": m}}, [], None)[0]

    def _acc(idx):
        model.eval()
        with torch.no_grad():
            logits = _logits(obs[idx], mask[idx])
            correcto = (logits.argmax(1) == action[idx])
            acc = float(correcto.float().mean()) if len(idx) else float("nan")
            lid = es_lider[idx]
            acc_lider = float(correcto[lid].float().mean()) if lid.any() else float("nan")
            acc_reac = float(correcto[~lid].float().mean()) if (~lid).any() else float("nan")
            return acc, acc_lider, acc_reac

    acc_inicial, acc_lider_inicial, acc_reac_inicial = _acc(val_idx)
    print(f"val_acc ANTES de la corrección: {acc_inicial:.3f}  "
          f"(liderando {acc_lider_inicial:.3f} / reaccionando {acc_reac_inicial:.3f})")

    mejor_vacc = acc_inicial
    mejor_state = pesos_iniciales
    for ep in range(args.epochs):
        model.train()
        tr_perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(tr_perm), args.batch):
            b = tr_perm[i:i + args.batch]
            logits = _logits(obs[b], mask[b])
            perdidas = loss_fn(logits, action[b])
            loss = (perdidas * peso[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(b)
            correct += int((logits.argmax(1) == action[b]).sum())
            tot += len(b)
        vacc, vacc_lider, vacc_reac = _acc(val_idx)
        print(f"  epoch {ep+1:>2}/{args.epochs}  loss {loss_sum/tot:.4f}  "
              f"train_acc {correct/tot:.3f}  val_acc {vacc:.3f}  "
              f"(lidera {vacc_lider:.3f} / reacciona {vacc_reac:.3f})", flush=True)
        if vacc > mejor_vacc:
            mejor_vacc = vacc
            mejor_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(mejor_state)
    print(f"\nMejor val_acc: {mejor_vacc:.3f} (inicial sin corrección: {acc_inicial:.3f})")

    weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    policy_dir = os.path.join(args.out, "policies", "default_policy")
    os.makedirs(policy_dir, exist_ok=True)
    with open(os.path.join(policy_dir, "policy_state.pkl"), "wb") as f:
        pickle.dump({"weights": weights}, f)
    print(f"Checkpoint (liviano) guardado en {args.out}  "
          f"(usable con Recomendador/pimc_regret_real.py --modelo)")


if __name__ == "__main__":
    main()
