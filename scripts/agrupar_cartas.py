"""
Descubre las plantillas de esquina de las 52 cartas agrupando, por similitud,
los recortes de esquina de las cartas de la MESA (una carta limpia por slot) a
lo largo de muchos fotogramas. Resultado: un representante por grupo en
`--salida`, listo para etiquetar a mano con el nombre de la carta (p.ej. 'QP').

Las cartas de la mesa son ideales como fuente: van aisladas (no solapadas) en
las 4 posiciones en cruz, asi que la esquina sale limpia.

Uso:
  python scripts/agrupar_cartas.py \
      --frames "videos/fotogramas importantes" \
      --regiones calibracion/hearts_app/regiones.json \
      --salida calibracion/hearts_app/cartas_descubiertas \
      --umbral 22
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
import json

import numpy as np

from src.captura.vision_cartas import detectar_cartas, firma_esquina

# Tamano canonico del recorte de esquina guardado (ancho x alto).
_CW, _CH = 70, 96
_ESQ_W, _ESQ_H = 0.42, 0.30


def _esquina_de_slot(img, caja_frac):
    import cv2

    h, w = img.shape[:2]
    x, y, cw, ch = caja_frac
    x0, y0 = int(x * w), int(y * h)
    roi = img[y0:y0 + int(ch * h), x0:x0 + int(cw * w)]
    cartas = detectar_cartas(roi, min_area_frac=0.05)
    if not cartas:
        return None
    cx, cy, ccw, cch = max(cartas, key=lambda b: b[2] * b[3])
    carta = roi[cy:cy + cch, cx:cx + ccw]
    esq = carta[0:int(_ESQ_H * cch), 0:int(_ESQ_W * ccw)]
    if esq.size == 0:
        return None
    return cv2.resize(esq, (_CW, _CH), interpolation=cv2.INTER_AREA)


def main() -> None:
    p = argparse.ArgumentParser(description="Agrupa esquinas de cartas de la mesa.")
    p.add_argument("--frames", required=True)
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--salida", default="calibracion/hearts_app/cartas_descubiertas")
    p.add_argument("--umbral", type=int, default=22)
    args = p.parse_args()

    import cv2

    reg = json.loads(_Path(args.regiones).read_text(encoding="utf-8"))
    mesa = reg["mesa"]
    frames = sorted(_Path(args.frames).glob("*.png"))
    if not frames:
        raise SystemExit(f"No hay PNG en {args.frames}")

    grupos = []  # {"firma","repr","frame","n"}
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for nombre, caja in mesa.items():
            esq = _esquina_de_slot(img, caja)
            if esq is None:
                continue
            f = firma_esquina(esq)
            mejor, mejor_d = None, 10 ** 9
            for gidx, g in enumerate(grupos):
                d = int(np.count_nonzero(f != g["firma"]))
                if d < mejor_d:
                    mejor, mejor_d = gidx, d
            if mejor is not None and mejor_d <= args.umbral:
                grupos[mejor]["n"] += 1
            else:
                grupos.append({"firma": f, "repr": esq.copy(),
                               "frame": fp.name, "n": 1})

    salida = _Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    grupos.sort(key=lambda g: -g["n"])
    manifiesto = {}
    for i, g in enumerate(grupos):
        nombre = f"grupo_{i:02d}__n{g['n']}__{g['frame']}"
        cv2.imwrite(str(salida / nombre), g["repr"])
        manifiesto[nombre] = {"n": g["n"], "frame": g["frame"]}
    (salida / "_manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(frames)} fotogramas -> {len(grupos)} grupos de carta. "
          f"Representantes en {salida}")


if __name__ == "__main__":
    main()
