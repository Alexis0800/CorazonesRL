"""
Entrena un opponent de IMITACIÓN HUMANA (behavior cloning) sobre las partidas
reales del puente SFS, para meterlo al pool de self-play y romper la burbuja:
el modelo campeón le gana 79% a los bots pero solo 24% a humanos (ver
`docs/auditoria_moon_2026-07-20.md`) porque entrena contra oponentes que no
juegan como humanos. Este bot clona el estilo de los 3 asientos humanos.

Clave (arquitectura): entrena un `HeartsActionMaskModel` IDÉNTICO al campeón, así
sus pesos se envuelven con `SnapshotPolicy.from_weights(...)` y entran al pool
como un snapshot cualquiera — cero cirugía en el env/pool.

Clave (obs): usa `replay.partidas_a_arrays(..., seats_de="rivales")`, que ahora
emite la obs COMPLETA alineada con el env (ver `src/captura/replay.py`). Sin esa
alineación el BC no sirve como oponente en el env.

Uso:
    python scripts/entrenar_bc_humano.py --partidas data/partidas_bridge_full.jsonl \
        --out-dir models/humano_bc --obs-dim 228 --epocas 40
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


def _split_por_partida(partidas, val_frac, rng):
    idx = np.arange(len(partidas))
    rng.shuffle(idx)
    corte = int(len(idx) * (1 - val_frac))
    tr = [partidas[i] for i in idx[:corte]]
    va = [partidas[i] for i in idx[corte:]]
    return tr, va


def main() -> None:
    import torch
    import torch.nn as nn
    from gymnasium import spaces

    from src.captura.escritor import cargar_partidas
    from src.captura import replay
    from src.entorno.dimensiones import DIM_V12, NUM_CARTAS
    from src.entorno.moon_model import RUTA_MOON
    from src.rllib.model import HeartsActionMaskModel

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partidas", required=True)
    p.add_argument("--out-dir", default="models/humano_bc")
    p.add_argument("--seats-de", default="rivales", choices=["agente", "rivales", "luna"],
                   help="'rivales' = clon humano (default); 'luna' = BC de persecución "
                        "de pozo (solo manos con luna, perspectiva del luneador)")
    p.add_argument("--obs-dim", type=int, default=DIM_V12)
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paciencia", type=int, default=8)
    p.add_argument("--moon-dir", default=RUTA_MOON,
                   help="Pesos moon para [187:189]. DEBE coincidir con el --moon-dir "
                        "del fine-tune (train_rllib) o esas 2 features quedan OOD.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    partidas = cargar_partidas(args.partidas)
    print(f"{len(partidas)} partidas cargadas de {args.partidas}")
    tr_p, va_p = _split_por_partida(partidas, args.val_frac, rng)

    # obs COMPLETA desde la perspectiva de los 3 rivales humanos, CON la máscara
    # legal de cada decisión: entrenar enmascarado alinea el BC con la inferencia
    # del env (que siempre enmascara) y concentra la probabilidad en las legales.
    Xtr, ytr, Mtr = replay.partidas_a_arrays(
        tr_p, dim=args.obs_dim, seats_de=args.seats_de, moon_dir=args.moon_dir, con_mask=True)
    Xva, yva, Mva = replay.partidas_a_arrays(
        va_p, dim=args.obs_dim, seats_de=args.seats_de, moon_dir=args.moon_dir, con_mask=True)
    print(f"Ejemplos (jugadas humanas): train {len(ytr)}  val {len(yva)}")
    if len(ytr) == 0:
        raise SystemExit("Sin ejemplos: ¿partidas reconstruibles?")

    obs_space = spaces.Dict({
        "obs": spaces.Box(0.0, 1.0, shape=(args.obs_dim,), dtype=np.float32),
        "action_mask": spaces.Box(0.0, 1.0, shape=(NUM_CARTAS,), dtype=np.float32),
    })
    model_config = {"fcnet_hiddens": [512, 512, 256], "fcnet_activation": "relu", "vf_share_layers": False}
    model = HeartsActionMaskModel(obs_space, spaces.Discrete(NUM_CARTAS), NUM_CARTAS, model_config, "humano_bc")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.long)
    # Máscara legal REAL de cada decisión: entrenar enmascarado (igual que la
    # inferencia del env) concentra la probabilidad en las cartas legales y
    # elimina el ~10% de "primera preferencia ilegal" medido con el BC sin máscara.
    mask_tr = torch.tensor(Mtr, dtype=torch.float32)
    mask_va = torch.tensor(Mva, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    def _acc(X, y, M):
        model.train(False)
        with torch.no_grad():
            logits, _ = model.forward({"obs": {"obs": X, "action_mask": M}}, [], None)
            pred = logits.argmax(1)
            return (pred == y).float().mean().item()

    n = len(ytr_t)
    mejor_acc = -1.0
    mejor_state = None
    sin_mejora = 0
    for ep in range(args.epocas):
        model.train(True)
        perm = torch.randperm(n)
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            logits, _ = model.forward(
                {"obs": {"obs": Xtr_t[b], "action_mask": mask_tr[b]}}, [], None
            )
            loss = lossf(logits, ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        va_acc = _acc(Xva_t, yva_t, mask_va)
        if va_acc > mejor_acc:
            mejor_acc = va_acc
            mejor_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
        print(f"  epoca {ep+1:02d}: val top-1 acc = {va_acc:.4f}  (mejor {mejor_acc:.4f})")
        if sin_mejora >= args.paciencia:
            print(f"  early stop (paciencia {args.paciencia})")
            break

    tr_acc = _acc(Xtr_t, ytr_t, mask_tr)
    print(f"\n=== BC humano: train top-1 {tr_acc:.4f}  |  val top-1 {mejor_acc:.4f} ===")
    print("(top-1 = fraccion de jugadas donde el bot elige la MISMA carta que el humano)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pesos = {k: v.numpy() for k, v in mejor_state.items()}
    np.savez(out / "pesos.npz", **pesos)
    (out / "metadata.json").write_text(json.dumps({
        "obs_dim": args.obs_dim, "arquitectura": "HeartsActionMaskModel",
        "fcnet_hiddens": [512, 512, 256],
        "val_top1": mejor_acc, "train_top1": tr_acc,
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "fuente": args.partidas, "seats_de": args.seats_de, "moon_dir": args.moon_dir,
        "mask_legal": True,
    }, indent=2), encoding="utf-8")
    print(f"Pesos -> {out/'pesos.npz'}  (envolver con SnapshotPolicy.from_weights)")


if __name__ == "__main__":
    main()
