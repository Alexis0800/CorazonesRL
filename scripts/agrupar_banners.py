"""
Descubre el catalogo de mensajes (banner) de la app agrupando, por similitud
visual, el recorte del banner de muchos fotogramas. Resultado: un representante
por grupo en `--salida`, listo para etiquetar a mano (cada PNG = un mensaje
distinto: pase izq/der/enfrente, "El turno de X", etc.).

No usa OCR ni dependencias de sistema: firma = banner en gris, reescalado a una
rejilla pequena y binarizado; agrupa por distancia de Hamming (greedy).

Uso:
  python scripts/agrupar_banners.py \
      --frames "videos/fotogramas importantes" \
      --regiones calibracion/hearts_app/regiones.json \
      --region banner \
      --salida calibracion/hearts_app/banners_descubiertos \
      --umbral 12

`--region` puede ser 'banner' o una clave de 'marcador'/'mesa' (p.ej. para
descubrir variantes de una caja). El manifiesto (grupo -> frames) se escribe en
`<salida>/_manifiesto.json`.
"""
from __future__ import annotations
import numpy as np
import json
import argparse

# --- bootstrap path ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Rejilla de la firma perceptual (ancho x alto en celdas).
_GW, _GH = 64, 16


def _firma(roi) -> np.ndarray:
    """Firma binaria del recorte: gris -> NxM -> umbral por la media."""
    import cv2

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (_GW, _GH), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).astype(np.uint8).ravel()


def _caja_px(reg, region, w, h):
    if region in reg and isinstance(reg[region], list):
        caja = reg[region]
    else:
        # region anidada: "marcador.abajo", "mesa.izquierda"
        grupo, _, nombre = region.partition(".")
        caja = reg[grupo][nombre]
    x, y, cw, ch = caja
    return int(x * w), int(y * h), int(cw * w), int(ch * h)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Agrupa recortes de banner por similitud.")
    p.add_argument("--frames", required=True,
                   help="Directorio de fotogramas PNG.")
    p.add_argument(
        "--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--region", default="banner")
    p.add_argument(
        "--salida", default="calibracion/hearts_app/banners_descubiertos")
    p.add_argument("--umbral", type=int, default=48,
                   help="Distancia Hamming maxima para considerar el mismo grupo.")
    args = p.parse_args()

    import cv2

    reg = json.loads(_Path(args.regiones).read_text(encoding="utf-8"))
    frames = sorted(_Path(args.frames).glob("*.png"))
    if not frames:
        raise SystemExit(f"No hay PNG en {args.frames}")

    # cada uno: {"firma", "repr_roi", "repr_frame", "frames":[...]}
    grupos = []
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        x, y, cw, ch = _caja_px(reg, args.region, w, h)
        roi = img[y:y + ch, x:x + cw]
        if roi.size == 0:
            continue
        f = _firma(roi)
        mejor, mejor_d = None, 10 ** 9
        for gidx, g in enumerate(grupos):
            d = int(np.count_nonzero(f != g["firma"]))
            if d < mejor_d:
                mejor, mejor_d = gidx, d
        if mejor is not None and mejor_d <= args.umbral:
            grupos[mejor]["frames"].append(fp.name)
        else:
            grupos.append({"firma": f, "repr_roi": roi.copy(),
                           "repr_frame": fp.name, "frames": [fp.name]})

    salida = _Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    manifiesto = {}
    grupos.sort(key=lambda g: -len(g["frames"]))
    for i, g in enumerate(grupos):
        nombre = f"grupo_{i:02d}__n{len(g['frames'])}__{g['repr_frame']}"
        cv2.imwrite(str(salida / nombre), g["repr_roi"])
        manifiesto[nombre] = {"n": len(g["frames"]),
                              "repr": g["repr_frame"],
                              "frames": g["frames"]}
    (salida / "_manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(frames)} fotogramas -> {len(grupos)} grupos. "
          f"Representantes en {salida}")


if __name__ == "__main__":
    main()
