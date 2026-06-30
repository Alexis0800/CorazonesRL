"""
Recorta la región del banner de todos los fotogramas importantes y guarda
solo los ÚNICOS (deduplicados por hash perceptual). Ideal para crear o mejorar
plantillas de banner para `BannerClasificador`.

Los recortes van a `calibracion/hearts_app/banners_crudos/`. Luego los revisás
a mano, los renombras con el tag correcto (pase_derecha__0.png, turno_agente__1.png,
etc.) y los copiás a `banners/`.

Uso:
  python scripts/extraer_banners.py
  python scripts/extraer_banners.py --umbral-dedup 4   # mas sensible (menos duplicados)
  python scripts/extraer_banners.py --umbral-dedup 12  # mas laxo
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# --- bootstrap path ---
_sys = sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np


_DEFAULT_REGIONES = "calibracion/hearts_app/regiones.json"
_DEFAULT_FRAMES = "videos/fotogramas importantes"
_DEFAULT_SALIDA = "calibracion/hearts_app/banners_crudos"


def _dhash(img_gray: np.ndarray, size: int = 8) -> int:
    """Hash perceptual (dHash): diferencia horizontal entre píxeles adyacentes.

    Redimensiona a (size+1)×size, compara columnas adyacentes → 64 bits.
    Distancia de Hamming < umbral → misma imagen (incluso con leve ruido)."""
    resized = cv2.resize(img_gray, (size + 1, size))
    diff = resized[:, 1:] > resized[:, :-1]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extrae banners únicos de fotogramas importantes.")
    p.add_argument("--frames", default=_DEFAULT_FRAMES,
                   help="Directorio con los PNG de los fotogramas.")
    p.add_argument("--regiones", default=_DEFAULT_REGIONES)
    p.add_argument("--salida", default=_DEFAULT_SALIDA)
    p.add_argument("--umbral-dedup", type=int, default=6,
                   help="Distancia de Hamming máxima para considerar dos banners "
                        "idénticos (default: 6). Menor = más estricto.")
    args = p.parse_args()

    from src.captura.vision_hearts import Regiones

    reg = Regiones.cargar(args.regiones)
    frames_dir = Path(args.frames)
    salida_dir = Path(args.salida)
    salida_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise SystemExit(f"No hay PNGs en {frames_dir}")

    print(f"Fotogramas: {len(frames)}")
    print(f"Región banner: {reg.banner}  (fracciones)")
    print(f"Dedup: distancia Hamming ≤ {args.umbral_dedup}")
    print()

    seen: list[tuple[int, str, int, int]] = []  # (hash, nombre_frame, W, H)
    guardados = 0
    saltados = 0

    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        # ── recortar banner ──
        bx0 = int(reg.banner[0] * W)
        by0 = int(reg.banner[1] * H)
        bw = int(reg.banner[2] * W)
        bh = int(reg.banner[3] * H)
        bx1, by1 = bx0 + bw, by0 + bh

        if bx1 > W or by1 > H or bw <= 0 or bh <= 0:
            continue

        crop = img[by0:by1, bx0:bx1]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h = _dhash(crop_gray)

        # ── ¿ya existe uno suficientemente parecido? ──
        dup = False
        for seen_h, seen_name, sw, sh in seen:
            if _hamming(h, seen_h) <= args.umbral_dedup:
                dup = True
                break

        if dup:
            saltados += 1
            continue

        # ── nuevo ──
        seen.append((h, fp.name, bw, bh))
        nombre = salida_dir / f"banner_{len(seen):04d}__{fp.stem}.png"
        cv2.imwrite(str(nombre), crop)
        guardados += 1

    print(f"Guardados: {guardados}  |  Saltados (duplicados): {saltados}")
    print(f"Salida: {salida_dir.resolve()}")
    print()
    print("Ahora revisalos a mano, renombralos con el tag correcto y copialos a")
    print(f"  calibracion/hearts_app/banners/")


if __name__ == "__main__":
    main()
