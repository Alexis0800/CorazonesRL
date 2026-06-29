"""
Genera la base de plantillas de cartas a partir del sprite ORIGINAL de la APK
(`calibracion/Corazones/recursos/assets/cards_0.png`). Mucho mejor que recortar
del video: arte limpio, sin compresion ni animacion, y las 52 con etiqueta exacta.

Layout del sprite (868x2184, RGBA): rejilla 4 filas x 13 columnas, carta 168x217.
  - filas: 0=picas(P), 1=corazones(C), 2=treboles(T), 3=diamantes(D)
  - columnas: A,2,3,4,5,6,7,8,9,10,J,Q,K

Salida:
  - `calibracion/hearts_app/cartas_completas/<carta>.png` — carta entera sobre blanco
    (versionada: la usa `vision_cartas.ReconocedorPlantilla` para leer la mano).
  - `<salida_esquinas>/<carta>__sprite.png` — esquina sup-izq canonica (reconocer).

Uso:
  python scripts/cartas_desde_sprite.py \
      --sprite calibracion/Corazones/recursos/assets/cards_0.png \
      --esquinas calibracion/hearts_app/cartas
"""
from __future__ import annotations

# --- bootstrap path ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse

import numpy as np

# fila -> palo (carta_a_str); columna -> valor
_FILA_PALO = {0: "P", 1: "C", 2: "T", 3: "D"}
_COL_VALOR = {0: "A", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7",
              7: "8", 8: "9", 9: "10", 10: "J", 11: "Q", 12: "K"}

# Recorte de esquina (mismas fracciones que vision_cartas.recortar_esquina).
_ESQ_W, _ESQ_H = 0.42, 0.30
_CW, _CH = 70, 96


def _sobre_blanco(bgra: np.ndarray) -> np.ndarray:
    if bgra.shape[2] == 4:
        a = bgra[:, :, 3:4] / 255.0
        return (bgra[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return bgra


def main() -> None:
    p = argparse.ArgumentParser(description="Genera plantillas de cartas del sprite.")
    p.add_argument("--sprite",
                   default="calibracion/Corazones/recursos/assets/cards_0.png")
    p.add_argument("--completas",
                   default="calibracion/hearts_app/cartas_completas")
    p.add_argument("--esquinas", default="calibracion/hearts_app/cartas")
    p.add_argument("--limpiar-esquinas", action="store_true",
                   help="Borra los PNG previos de la carpeta de esquinas.")
    args = p.parse_args()

    import cv2

    sprite = cv2.imread(args.sprite, cv2.IMREAD_UNCHANGED)
    if sprite is None:
        raise SystemExit(f"No se pudo leer el sprite: {args.sprite}")
    sprite = _sobre_blanco(sprite)
    H, W = sprite.shape[:2]
    ch, cw = H // 4, W // 13            # 217 x 168
    comp = _Path(args.completas); comp.mkdir(parents=True, exist_ok=True)
    esq = _Path(args.esquinas); esq.mkdir(parents=True, exist_ok=True)
    if args.limpiar_esquinas:
        for f in esq.glob("*.png"):
            f.unlink()

    n = 0
    for fila in range(4):
        for col in range(13):
            carta = _COL_VALOR[col] + _FILA_PALO[fila]
            y0, x0 = fila * ch, col * cw
            celda = sprite[y0:y0 + ch, x0:x0 + cw]
            cv2.imwrite(str(comp / f"{carta}.png"), celda)
            e = celda[0:int(_ESQ_H * ch), 0:int(_ESQ_W * cw)]
            e = cv2.resize(e, (_CW, _CH), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(esq / f"{carta}__sprite.png"), e)
            n += 1
    print(f"{n} cartas: completas en {comp}, esquinas en {esq}")


if __name__ == "__main__":
    main()
