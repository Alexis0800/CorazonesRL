"""
Encoder offline: convierte partidas capturadas (`.jsonl`) en un dataset
`(obs, acción)` `.npz` para Behavioral Cloning / fine-tune.

Re-juega cada partida en el motor y construye la observación con el mismo
`ObservacionBuilder` del entorno (consistencia sim↔real). Capa pura: no usa ADB.

Uso:
    python scripts/jsonl_a_dataset.py --jsonl datasets/humano/partidas.jsonl \
        --salida datasets/humano/bc_humano.npz
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")  # consola Windows cp1252
except Exception:
    pass

import argparse

import numpy as np

from src.entorno.dimensiones import DIM_ENTORNO
from src.captura.escritor import leer_partidas
from src.captura.replay import partidas_a_arrays


def main() -> None:
    p = argparse.ArgumentParser(description="JSONL de partidas -> dataset BC .npz")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--salida", default="datasets/humano/bc_humano.npz")
    p.add_argument("--dim", type=int, default=DIM_ENTORNO)
    args = p.parse_args()

    partidas = list(leer_partidas(args.jsonl))
    X, y = partidas_a_arrays(partidas, dim=args.dim)

    salida = _Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida, obs=X, acciones=y)
    print(f"✅ {len(partidas)} partidas → {X.shape[0]} ejemplos (dim={args.dim})")
    print(f"   guardado en {salida}")


if __name__ == "__main__":
    main()
