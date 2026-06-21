"""
Behavioral Cloning pretraining para v2_ronda.

Entrena un modelo MaskablePPO por imitacion (BC) usando un dataset
generado con PIMC/MCTS como oracle. Luego guarda el modelo para
fine-tuning con RL.

El dataset contiene pares (observacion_220, action_id) donde la action
es la jugada optima segun el oracle (minimiza puntos esperados).

Uso:
    # Generar dataset primero
    python scripts/generar_dataset_bc.py --manos 3000 --mundos 100 \\
        --rollout experto --oponentes experto --workers 8 \\
        --output datasets/bc_v2_mcts

    # BC pretraining
    python -m src.v2_ronda.train_bc --dataset datasets/bc_v2_mcts \\
        --output models/v2_bc --epochs 20

    # RL fine-tune
    python -m src.v2_ronda.train --resume models/v2_bc/snapshots/bc_model \\
        --total-steps 5000000 --output-dir models/v2_rl
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Asegurar que el proyecto esta en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def cargar_dataset_v2(prefix: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Carga dataset BC. Espera prefix.npz + prefix.json."""
    data = np.load(prefix + ".npz")
    obs = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    with open(prefix + ".json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return obs, actions, meta


def preentrenar_bc(
    dataset_path: str,
    output_dir: str = "models/v2_bc",
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> str:
    """Entrena modelo MaskablePPO por Behavioral Cloning.

    Flujo:
      1. Cargar dataset (obs, actions).
      2. Crear entorno single-hand y VecNormalize.
      3. Pre-ajustar VecNormalize con todas las observaciones del dataset.
      4. Crear MaskablePPO con vf_coef=0 (sin value head, solo actor).
      5. Entrenar policy por BC (maximizar log prob de acciones expertas).
      6. Guardar modelo + vecnorm.

    Args:
        dataset_path: Ruta al dataset (sin extension .npz).
        output_dir: Directorio de salida.
        epochs: Epocas de BC.
        batch_size: Tamaño de batch.
        learning_rate: Learning rate del optimizador BC.
        device: Dispositivo (cpu, cuda).
        seed: Semilla.

    Returns:
        Ruta al modelo guardado.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.v2_ronda.entorno import CorazonesEnvSingleHand

    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Directorios ---
    dir_snapshots = os.path.join(output_dir, "snapshots")
    dir_vecnorm = os.path.join(output_dir, "vecnorm")
    os.makedirs(dir_snapshots, exist_ok=True)
    os.makedirs(dir_vecnorm, exist_ok=True)

    # --- Cargar dataset ---
    print("=" * 60)
    print("  BC PRETRAINING v2_ronda")
    print("=" * 60)
    print(f"  Dataset: {dataset_path}")
    obs_all, actions_all, meta = cargar_dataset_v2(dataset_path)
    print(f"  Pares (obs, action): {len(obs_all):,}")
    print(f"  Manos: {meta.get('num_manos', '?')}")
    print(f"  Rollout: {meta.get('rollout_tipo', '?')}")
    print(f"  Acciones unicas: {len(np.unique(actions_all))}")
    print(f"  Obs shape: {obs_all.shape}")
    print("-" * 60)

    # --- Crear VecNormalize y pre-ajustar ---
    print("  Ajustando VecNormalize...")

    def _make_env():
        return CorazonesEnvSingleHand(agente_idx=0)
    venv = DummyVecEnv([_make_env])
    venv = VecNormalize(
        venv, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
    )

    # Pre-llenar RunningMeanStd del VecNormalize con todas las observaciones
    # del dataset para que las stats reflejen la distribucion real de BC
    venv.obs_rms = _compute_rms(obs_all, epsilon=1e-8)
    # Para ret_rms (reward running stats), inicializar con rango razonable
    # Rewards en v2: [-31, +52] → inicializar con media 0, var ~400
    venv.ret_rms = _compute_rms(
        np.zeros((1, 1), dtype=np.float32), epsilon=1e-8)
    # Ajustar manualmente para que el rango sea razonable
    venv.ret_rms.mean = np.array([0.0], dtype=np.float64)
    venv.ret_rms.var = np.array([400.0], dtype=np.float64)
    venv.ret_rms.count = 1000

    # Normalizar el dataset para BC
    obs_norm = venv.normalize_obs(obs_all)
    print(
        f"  Obs normalizadas: mean={obs_norm.mean():.3f}, std={obs_norm.std():.3f}")

    # --- Crear MaskablePPO (solo actor, sin value head) ---
    print("  Creando modelo MaskablePPO...")
    modelo = MaskablePPO(
        "MlpPolicy", venv,
        learning_rate=learning_rate,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,       # sin entropia en BC
        vf_coef=0.0,        # sin value head en BC
        max_grad_norm=0.5,
        target_kl=0.02,
        verbose=0,
        device=device,
    )

    # --- Behavioral Cloning ---
    print(f"  Entrenando BC ({epochs} epochs, batch={batch_size})...")
    dataset = TensorDataset(
        torch.from_numpy(obs_norm),
        torch.from_numpy(actions_all),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Usar el optimizador de la policy directamente
    optimizer = torch.optim.Adam(
        modelo.policy.parameters(), lr=learning_rate,
    )

    modelo.policy.train()
    n_batches = len(loader)

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0
        total_samples = 0

        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            optimizer.zero_grad()

            # Forward: obtener distribucion de acciones
            # MaskableActorCriticPolicy.evaluate_actions devuelve
            # (values, log_prob, entropy)
            # Pero necesitamos solo la policy (action distribution)
            # Usamos el feature extractor + action_net directamente
            features = modelo.policy.extract_features(batch_obs)
            latent = modelo.policy.mlp_extractor.policy_net(features)
            logits = modelo.policy.action_net(latent)

            # Cross-entropy loss (maximizar prob de accion experta)
            loss = F.cross_entropy(logits, batch_act)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_acc += (pred == batch_act).sum().item()
            total_samples += len(batch_act)

        avg_loss = total_loss / n_batches
        accuracy = total_acc / total_samples
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"loss={avg_loss:.4f} | acc={accuracy:.1%}")

    # --- Guardar ---
    vecnorm_path = os.path.join(dir_vecnorm, "bc_vecnorm.pkl")
    venv.save(vecnorm_path)

    modelo_path = os.path.join(dir_snapshots, "bc_model")
    modelo.save(modelo_path)
    venv.save(modelo_path + "_vecnorm.pkl")

    # Guardar metadata
    meta_path = os.path.join(output_dir, "bc_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_path,
            "num_pares": len(obs_all),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "timestamp": datetime.now().isoformat(),
            **meta,
        }, f, indent=2)

    print("-" * 60)
    print(f"  Modelo guardado: {modelo_path}.zip")
    print(f"  VecNormalize:    {modelo_path}_vecnorm.pkl")
    print(f"  Metadata:        {meta_path}")
    print("=" * 60)

    return modelo_path


def _compute_rms(
    data: np.ndarray, epsilon: float = 1e-8,
) -> Any:
    """Crea un RunningMeanStd pre-poblado con los datos.

    Args:
        data: Array (N, dims) de observaciones.
        epsilon: Valor pequeno para evitar division por cero.

    Returns:
        Objeto con atributos .mean, .var, .count compatibles con VecNormalize.
    """
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    rms = RunningMeanStd(shape=(data.shape[1],))
    # Poblar con todos los datos
    for i in range(data.shape[0]):
        rms.update(data[i:i+1])
    return rms


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BC Pretraining para v2_ronda",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Ruta al dataset BC (sin extension .npz).")
    parser.add_argument("--output", type=str, default="models/v2_bc",
                        help="Directorio de salida.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epocas de BC (default: 20).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3).")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Dispositivo (default: cpu).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla (default: 42).")

    args = parser.parse_args()

    preentrenar_bc(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
