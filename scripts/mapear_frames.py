"""
Mapea cada fotograma de una carpeta a su ESTADO de alto nivel leyendo el banner
(Fase 1 de la vision). Es la validacion del catalogo de mensajes y el primer
"mapa" de la partida: por cada frame, que esta pasando.

Uso:
  python scripts/mapear_frames.py \
      --frames "videos/fotogramas importantes" \
      --regiones calibracion/hearts_app/regiones.json \
      --banners calibracion/hearts_app/banners \
      --salida calibracion/hearts_app/mapa_frames.csv

Imprime un resumen por categoria y avisa de banners 'desconocido' (candidatos a
nuevas plantillas: vuelve a correr agrupar_banners.py sobre esos frames).
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
import csv
from collections import Counter

from src.captura.vision_hearts import BannerClasificador, Regiones


def main() -> None:
    p = argparse.ArgumentParser(description="Mapea fotogramas -> estado por banner.")
    p.add_argument("--frames", required=True)
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--banners", default="calibracion/hearts_app/banners")
    p.add_argument("--salida", default="calibracion/hearts_app/mapa_frames.csv")
    p.add_argument("--umbral", type=float, default=1.5)
    args = p.parse_args()

    import cv2

    reg = Regiones.cargar(args.regiones)
    clf = BannerClasificador(args.banners, umbral=args.umbral)
    frames = sorted(_Path(args.frames).glob("*.png"))
    if not frames:
        raise SystemExit(f"No hay PNG en {args.frames}")

    filas = []
    cat_cnt, tag_cnt = Counter(), Counter()
    desconocidos = []
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = clf.clasificar_con_regiones(img, reg)
        filas.append((fp.name, r.categoria, r.dato or "", r.tag, r.distancia))
        cat_cnt[r.categoria] += 1
        tag_cnt[r.tag] += 1
        if r.categoria == "desconocido":
            desconocidos.append((fp.name, r.distancia))

    salida = _Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "categoria", "dato", "tag", "distancia"])
        w.writerows(filas)

    print(f"{len(filas)} fotogramas mapeados -> {salida}\n")
    print("Por categoria:")
    for c, n in cat_cnt.most_common():
        print(f"  {c:12s} {n}")
    print("\nPor tag:")
    for t, n in tag_cnt.most_common():
        print(f"  {t:18s} {n}")
    if desconocidos:
        print(f"\n{len(desconocidos)} DESCONOCIDOS (candidatos a nueva plantilla):")
        for nombre, d in desconocidos[:20]:
            print(f"  {nombre}  (dist={d})")


if __name__ == "__main__":
    main()
