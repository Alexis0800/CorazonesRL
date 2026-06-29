"""
Ablación de la memoria del pase (v13): evalúa el MISMO modelo con los planos
[228:332] poblados (ON) vs en cero (OFF). La diferencia mide cuánto USA el modelo
esa información — el test limpio de "¿la memoria del pase ayuda?".

Uso:
    python scripts/ablacion_pase.py --snapshot models/v13/elite/elite_000017080320 \
        --partidas 200
"""
from __future__ import annotations

# --- bootstrap path ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse

from src.entorno.dimensiones import DIM_V13
from src.rllib.eval_bots import evaluar_vs_bots
from src.rllib.utils import cargar_policy_desde_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Ablación de la memoria del pase (v13).")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--obs-dim", type=int, default=DIM_V13)
    p.add_argument("--partidas", type=int, default=200)
    args = p.parse_args()

    snap = cargar_policy_desde_checkpoint(args.snapshot)
    print(f"Snapshot: {args.snapshot}  ({args.partidas} partidas/escenario)\n")

    print("Evaluando CON memoria de pase (ON)...")
    on = evaluar_vs_bots(snap, obs_dim=args.obs_dim, n_partidas=args.partidas,
                         con_pase=True, pase_memoria=True)
    print("Evaluando SIN memoria de pase (OFF, ablación)...")
    off = evaluar_vs_bots(snap, obs_dim=args.obs_dim, n_partidas=args.partidas,
                          con_pase=True, pase_memoria=False)

    claves = [
        ("top2_rate_vs_experto", "top2 vs experto"),
        ("win_rate_vs_experto", "win  vs experto"),
        ("top2_rate", "top2 global"),
        ("win_rate", "win  global"),
        ("puesto_medio", "puesto medio (↓)"),
    ]
    print(f"\n{'métrica':<20} {'ON':>8} {'OFF':>8} {'Δ (ON-OFF)':>12}")
    print("-" * 52)
    for k, label in claves:
        a, b = on.get(k, 0.0), off.get(k, 0.0)
        print(f"{label:<20} {a:>8.3f} {b:>8.3f} {a-b:>+12.3f}")
    print("\nΔ>0 en win/top2 ⇒ el modelo USA la memoria del pase (ayuda).")
    print("Δ≈0 ⇒ el modelo la ignora (no aprendió a aprovecharla en este run).")


if __name__ == "__main__":
    main()
