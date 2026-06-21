"""
Behavioral Cloning — entrena el actor con dataset PIMC y guarda modelo.

Entrena SOLO la policy network (actor) usando BC loss sobre un dataset
generado por PIMC. El value head se inicializa pero no se entrena aquí
(lo hará el RL después).

Uso:
    python scripts/entrenar_bc.py --dataset datasets/bc_dataset --output models/v22_bc

Luego para fine-tuning con RL:
    python train.py --resume models/v22_bc/snapshots/bc_model --total-steps 10000000 --output-dir models/v22
"""

from __future__ import annotations
from src.entorno.single_agent import CorazonesEnv
from src.entorno.dimensiones import DIM_ENTRENAMIENTO
from src.red import CorazonesFeatureExtractor
from src.mcts.dataset import cargar_dataset

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BCPolicy(nn.Module):
    """Policy network idéntica a la de MaskablePPO: extractor + action head."""

    def __init__(self, obs_dim: int, features_dim: int = 256, n_actions: int = 52):
        super().__init__()
        self.features_extractor = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, features_dim), nn.ReLU(),
        )
        self.action_net = nn.Linear(features_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        return self.action_net(features)


def entrenar_bc(
    dataset_prefix: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Entrena el actor con BC loss y guarda el modelo.

    Args:
        dataset_prefix: Prefijo del dataset (sin .npz/.json).
        output_dir: Directorio donde guardar el modelo.
        epochs: Épocas de entrenamiento BC.
        batch_size: Tamaño de batch.
        lr: Learning rate.
        device: 'cpu', 'cuda', 'dml', 'xpu'.
    """
    print("=" * 60)
    print("  BEHAVIORAL CLONING — Entrenamiento del Actor")
    print("=" * 60)

    # Cargar dataset
    obs, actions, meta = cargar_dataset(dataset_prefix)
    print(f"  Dataset: {len(obs)} pares (obs, action)")
    print(f"  Obs dim: {obs.shape[1]}")
    print(f"  Acciones únicas: {len(np.unique(actions))}")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR: {lr}")
    print("-" * 60)

    # Preparar datos
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    dataset = TensorDataset(obs_tensor, actions_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modelo
    model = BCPolicy(obs_dim=DIM_ENTRENAMIENTO, n_actions=52)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Entrenar
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            optimizer.zero_grad()
            logits = model(batch_obs)
            loss = criterion(logits, batch_act)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_obs.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_act).sum().item()
            total += batch_obs.size(0)

        acc = 100 * correct / total
        avg_loss = total_loss / total
        print(
            f"  Época {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")

    print("-" * 60)

    # Guardar modelo BC standalone (.pt)
    os.makedirs(output_dir, exist_ok=True)
    bc_path = os.path.join(output_dir, "bc_policy.pt")
    torch.save(model.state_dict(), bc_path)
    print(f"  ✅ BC policy guardada: {bc_path}")

    # Crear MaskablePPO con los pesos BC para poder usar --resume
    print("  Creando MaskablePPO con pesos BC...")
    try:
        from sb3_contrib import MaskablePPO

        # Crear env dummy
        env = CorazonesEnv(agente_idx=0, obs_dim=DIM_ENTRENAMIENTO)

        # Crear MaskablePPO con la misma arquitectura
        from src.red import obtener_policy_kwargs
        hp = obtener_policy_kwargs()

        ppo_model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=hp,
            learning_rate=3e-5,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.12,
            vf_coef=0.25,
            max_grad_norm=0.3,
            verbose=0,
            device=device,
        )

        # Transferir pesos del actor BC al MaskablePPO
        bc_state = model.state_dict()
        ppo_state = ppo_model.policy.state_dict()

        # Mapear pesos: BC usa features_extractor + action_net
        # MaskablePPO usa mlp_extractor.policy_net + action_net
        # Las keys son diferentes, necesitamos mapearlas
        transferred = 0
        for key_bc, value_bc in bc_state.items():
            # BC: features_extractor.0.weight, features_extractor.0.bias, ...
            # PPO: mlp_extractor.policy_net.0.weight, ...
            # PPO: features_extractor.net.0.weight, ...
            # Intentar varios mapeos
            ppo_key = None
            for candidate in [
                key_bc,  # mismo nombre
                key_bc.replace("features_extractor.",
                               "features_extractor.net."),
                key_bc.replace("features_extractor.",
                               "mlp_extractor.policy_net."),
                key_bc.replace("action_net.", "action_net."),
            ]:
                if candidate in ppo_state:
                    ppo_key = candidate
                    break

            if ppo_key and ppo_state[ppo_key].shape == value_bc.shape:
                ppo_state[ppo_key].copy_(value_bc)
                transferred += 1

        ppo_model.policy.load_state_dict(ppo_state)
        print(f"  Pesos transferidos: {transferred}/{len(bc_state)}")

        # Guardar snapshot para --resume
        snap_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        snap_path = os.path.join(snap_dir, "bc_model")
        ppo_model.save(snap_path)
        print(f"  ✅ Snapshot guardado: {snap_path}.zip")

        env.close()
    except ImportError:
        print("  ⚠️  sb3-contrib no disponible — solo se guardó bc_policy.pt")

    print("=" * 60)
    print(f"  LISTO. Para fine-tuning:")
    print(
        f"  python train.py --resume {output_dir}/snapshots/bc_model --total-steps 10000000 --output-dir models/v22 --eval-experto")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning desde dataset PIMC")
    parser.add_argument("--dataset", type=str, default="datasets/bc_dataset",
                        help="Prefijo del dataset (sin .npz/.json)")
    parser.add_argument("--output", type=str, default="models/v22_bc",
                        help="Directorio de salida")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Épocas de entrenamiento BC")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo: cpu, cuda, dml, xpu")
    args = parser.parse_args()

    entrenar_bc(
        dataset_prefix=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
