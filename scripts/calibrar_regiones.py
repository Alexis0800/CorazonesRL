"""
Overlay de calibracion: dibuja las regiones de `regiones.json` sobre un
fotograma real y guarda un PNG anotado para verificar (y ajustar) las cajas.

Las cajas del JSON estan en FRACCIONES (0..1) del ancho/alto, asi que el mismo
fichero sirve para cualquier resolucion. Este script las convierte a pixeles
sobre la imagen concreta y las dibuja con su etiqueta.

Uso:
  python scripts/calibrar_regiones.py \
      --frame "videos/fotogramas importantes/frame_000000.png" \
      --regiones calibracion/hearts_app/regiones.json \
      --salida calibracion/hearts_app/_overlay.png

Requiere opencv-python (requirements-captura.txt).
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

# Colores BGR por grupo de region.
_COLORES = {
    "banner": (0, 0, 255),       # rojo
    "marcador": (0, 200, 0),     # verde
    "mesa": (255, 128, 0),       # azul
    "mano": (0, 200, 255),       # amarillo
}


def _frac_a_px(caja, w, h):
    x, y, cw, ch = caja
    return int(x * w), int(y * h), int(cw * w), int(ch * h)


def _dibujar(img, caja_px, color, etiqueta):
    import cv2

    x, y, w, h = caja_px
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    cv2.putText(img, etiqueta, (x + 4, max(y - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay de regiones sobre un fotograma.")
    p.add_argument("--frame", required=True, help="PNG del fotograma a anotar.")
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--salida", default="calibracion/hearts_app/_overlay.png")
    args = p.parse_args()

    import cv2

    img = cv2.imread(args.frame, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"No se pudo leer el fotograma: {args.frame}")
    h, w = img.shape[:2]
    print(f"Fotograma: {w}x{h}")

    reg = json.loads(_Path(args.regiones).read_text(encoding="utf-8"))

    # banner (caja simple)
    if "banner" in reg:
        _dibujar(img, _frac_a_px(reg["banner"], w, h), _COLORES["banner"], "banner")

    # mano (caja simple)
    if "mano" in reg:
        _dibujar(img, _frac_a_px(reg["mano"], w, h), _COLORES["mano"], "mano")

    # marcador y mesa (dicts de cajas)
    for grupo in ("marcador", "mesa"):
        for nombre, caja in reg.get(grupo, {}).items():
            _dibujar(img, _frac_a_px(caja, w, h),
                     _COLORES[grupo], f"{grupo[:4]}:{nombre}")

    salida = _Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida), img)
    print(f"Overlay guardado en {salida}")


if __name__ == "__main__":
    main()
