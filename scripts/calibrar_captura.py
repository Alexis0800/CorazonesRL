"""
Ayuda de calibración del parser ADB: toma un screenshot del móvil y lo guarda
para que recortes las plantillas de las 52 cartas y definas las regiones.

Flujo sugerido:
  1. Conecta el móvil (USB, depuración ADB activada) con la app en una mano.
  2. `python scripts/calibrar_captura.py --salida calibracion/captura.png`
  3. Recorta cada carta a `calibracion/cartas/<id>.png` (id 0-51, ver
     src/captura/modelos.carta_a_str para el mapeo).
  4. Define las cajas en `calibracion/regiones.json`, p.ej.:
       {"mano": [x, y, w, h], "mesa": [x, y, w, h], "marcador": [x, y, w, h]}
  5. Usa `scripts/capturar.py --fuente adb ...`.

Requiere `opencv-python` (requirements-captura.txt) y `adb` en el PATH.
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

from src.captura.adb import ClienteADB


def main() -> None:
    p = argparse.ArgumentParser(description="Captura un screenshot vía ADB.")
    p.add_argument("--salida", default="calibracion/captura.png")
    p.add_argument("--serial", default=None)
    args = p.parse_args()

    import cv2  # dep opcional

    cliente = ClienteADB(serial=args.serial)
    disp = cliente.dispositivos()
    if not disp:
        raise SystemExit("No hay dispositivos ADB conectados (revisa `adb devices`).")
    print(f"Dispositivos: {disp}")

    img = cliente.captura()
    salida = _Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida), img)
    print(f"✅ Screenshot {img.shape[1]}x{img.shape[0]} guardado en {salida}")


if __name__ == "__main__":
    main()
