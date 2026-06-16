"""
Preentrenamiento supervisado (Behavioral Cloning) sobre el dataset MCTS.

Entrena el actor de MaskablePPO con CrossEntropyLoss usando los pares
(observación, jugada_óptima_PIMC). El modelo resultante sirve como punto
de partida para el fine-tuning RL (Phase 10).

Uso:
    python train_bc.py \
        --dataset datasets/mcts_50k.npz \
        --output modelos/bc_pretrain \
        --epochs 30 --lr 1e-3 --batch 512 --obs-dim 220

El archivo .zip resultante puede cargarse con MaskablePPO.load() para
continuar entrenamiento RL.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sb3_contrib import MaskablePPO
from src.entorno.single_agent import CorazonesEnv
from src.red import obtener_policy_kwargs


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

def cargar_dataset(path: str):
    """Carga el .npz y retorna (obs, acciones) como tensores."""
    datos = np.load(path)
    obs = torch.FloatTensor(datos["obs"])
    acciones = torch.LongTensor(datos["acciones"])
    print(f"Dataset cargado: {obs.shape[0]:,} ejemplos, obs_dim={obs.shape[1]}")
    return obs, acciones


def crear_dataloaders(obs, acciones, val_frac: float = 0.1, batch_size: int = 512):
    dataset = TensorDataset(obs, acciones)
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Split: {n_train:,} train / {n_val:,} val")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────
# Forward pass del actor (SB3 internals)
# ──────────────────────────────────────────────────────────────

def actor_logits(policy, obs_tensor: torch.Tensor) -> torch.Tensor:
    """Calcula los logits del actor para un batch de observaciones.

    Ruta: features_extractor → mlp_extractor (actor head) → action_net
    """
    features = policy.features_extractor(obs_tensor)
    latent_pi, _ = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


# ──────────────────────────────────────────────────────────────
# Evaluación
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluar(policy, loader, device: str) -> tuple[float, float]:
    """Retorna (loss_promedio, accuracy) en el dataloader."""
    policy.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for obs_b, acc_b in loader:
        obs_b = obs_b.to(device)
        acc_b = acc_b.to(device)
        logits = actor_logits(policy, obs_b)
        loss = loss_fn(logits, acc_b)
        total_loss += loss.item() * len(acc_b)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == acc_b).sum().item()
        total_n += len(acc_b)

    policy.train()
    return total_loss / total_n, total_correct / total_n


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def entrenar(
    dataset_path: str,
    output_dir: str,
    obs_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    val_frac: float,
    device_str: str,
    paciencia: int,
):
    device = torch.device(device_str)

    # 1. Dataset
    obs, acciones = cargar_dataset(dataset_path)
    train_loader, val_loader = crear_dataloaders(obs, acciones, val_frac, batch_size)

    # 2. Modelo MaskablePPO (solo para obtener la arquitectura de política)
    env = CorazonesEnv(agente_idx=0, obs_dim=obs_dim)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=obtener_policy_kwargs(),
        verbose=0,
        device=device_str,
    )
    env.close()

    policy = model.policy.to(device)
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Política inicializada | {n_params:,} parámetros entrenables")

    # 3. Optimizador + scheduler
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Entrenamiento
    os.makedirs(output_dir, exist_ok=True)
    mejor_val_acc = 0.0
    sin_mejora = 0
    t0 = time.time()

    print(f"\nEntrenando {epochs} épocas en {device_str}")
    print(f"{'Época':>6} | {'Loss Train':>11} | {'Acc Train':>10} | {'Loss Val':>9} | {'Acc Val':>8} | {'LR':>8}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        policy.train()
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for obs_b, acc_b in train_loader:
            obs_b = obs_b.to(device)
            acc_b = acc_b.to(device)

            logits = actor_logits(policy, obs_b)
            loss = loss_fn(logits, acc_b)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(acc_b)
            preds = logits.detach().argmax(dim=-1)
            total_correct += (preds == acc_b).sum().item()
            total_n += len(acc_b)

        scheduler.step()

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        val_loss, val_acc = evaluar(policy, val_loader, device_str)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>6} | {train_loss:>11.4f} | {train_acc:>10.4f} | "
            f"{val_loss:>9.4f} | {val_acc:>8.4f} | {current_lr:>8.2e}"
        )

        # Guardar mejor modelo por val accuracy
        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc
            sin_mejora = 0
            model.policy = policy
            model.save(os.path.join(output_dir, "bc_best"))
            print(f"          * Mejor modelo guardado (val_acc={val_acc:.4f})")
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print(f"\nEarly stopping: sin mejora en {paciencia} épocas consecutivas.")
                break

    elapsed = time.time() - t0
    print(f"\nEntrenamiento completado en {elapsed:.1f}s")
    print(f"Mejor val_acc: {mejor_val_acc:.4f}")
    print(f"Modelo guardado en: {output_dir}/bc_best.zip")

    # Guardar también el modelo final (última época)
    model.policy = policy
    model.save(os.path.join(output_dir, "bc_final"))
    print(f"Modelo final guardado en: {output_dir}/bc_final.zip")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning sobre dataset MCTS")
    parser.add_argument("--dataset", required=True, help="Ruta al .npz generado por generar_dataset_mcts.py")
    parser.add_argument("--output", default="modelos/bc_pretrain", help="Directorio de salida")
    parser.add_argument("--obs-dim", type=int, default=220, choices=[194, 220])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--paciencia", type=int, default=10, help="Early stopping: épocas sin mejora")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo PyTorch (cpu, cuda, dml)")
    args = parser.parse_args()

    entrenar(
        dataset_path=args.dataset,
        output_dir=args.output,
        obs_dim=args.obs_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        val_frac=args.val_frac,
        device_str=args.device,
        paciencia=args.paciencia,
    )


if __name__ == "__main__":
    main()
