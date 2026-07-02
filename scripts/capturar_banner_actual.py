"""
Captura el BANNER que se ve AHORA en el dispositivo y guarda dos cosas:
  - el screenshot completo en `debug/<tag>_full.png` (para recortar con precisión)
  - la banda del banner en `debug/<tag>_banda.png` (recorte de `regiones.banner`)

Úsalo cuando aparezca un banner que el clasificador NO conoce (p.ej.
«Tú se llevará todo el resto»): deja el banner en pantalla y ejecuta:

  python scripts/capturar_banner_actual.py --serial R52W90431FP --tag remate_agente

Luego, la PLANTILLA definitiva (86x680 aprox., como las de
`calibracion/hearts_app/banners/*.png`) se recorta del screenshot completo y se
guarda como `calibracion/hearts_app/banners/<tag>__0.png`.

⚠ Requiere `adb` en el PATH y `opencv-python`.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Captura el banner actual del móvil.")
    p.add_argument("--tag", required=True,
                   help="Etiqueta del banner (p.ej. remate_agente, remate_rival).")
    p.add_argument("--serial", default=None)
    p.add_argument("--adb", default="adb")
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--salida", default="debug")
    args = p.parse_args()

    import cv2
    from src.captura.adb import ClienteADB
    from src.captura.vision_hearts import Regiones, BannerClasificador

    cliente = ClienteADB(serial=args.serial, adb=args.adb)
    if not cliente.dispositivos():
        raise SystemExit("No hay dispositivos ADB (revisa `adb devices`).")

    reg = Regiones.cargar(args.regiones)
    img = cliente.captura()
    out = _Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)

    full = out / f"{args.tag}_full.png"
    banda = out / f"{args.tag}_banda.png"
    cv2.imwrite(str(full), img)
    cv2.imwrite(str(banda), Regiones.recortar(img, reg.banner))
    print(f"✅ screenshot  → {full}  ({img.shape[1]}x{img.shape[0]})")
    print(f"✅ banda banner → {banda}")

    # Pista: ¿cómo clasifica ahora? (debería ser 'desconocido' o algo lejano)
    clf = BannerClasificador("calibracion/hearts_app/banners")
    res, top = clf.clasificar_verbose_con_regiones(img, reg)
    print(f"   clasificación actual: {res.tag} (d={res.distancia:.2f}, "
          f"cat={res.categoria})")
    print("   top:", "  ".join(f"{t}={d:.2f}" for t, d in top))
    print(f"\nSiguiente paso: recorta el banner de {full} y guárdalo como "
          f"calibracion/hearts_app/banners/{args.tag}__0.png")


if __name__ == "__main__":
    main()
