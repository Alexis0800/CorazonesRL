"""
Diagnostico de UN solo fotograma: dime que ve la vision en una captura.

Pensado para probar rapido una captura real (ADB/screenshot) y ver:
  - que dice el BANNER (fase, direccion de pase, turno, ganador de baza)
  - que CARTAS hay en la mesa (las 4 en cruz)
  - intento (best-effort) de leer tu MANO

Tambien guarda un overlay con las regiones dibujadas para verificar calibracion.

Uso:
  python scripts/diagnostico_captura.py --frame calibracion/hearts_app/captura_real.png
  python scripts/diagnostico_captura.py --frame mi_captura.png --overlay
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

# Banner tag -> descripcion humana de que significa.
_DESC_BANNER = {
    "pase_izquierda": "FASE DE PASE -> pasa 3 cartas a la IZQUIERDA",
    "pase_derecha": "FASE DE PASE -> pasa 3 cartas a la DERECHA",
    "pase_enfrente": "FASE DE PASE -> pasa 3 cartas de ENFRENTE",
    "pase_recibido": "Te han pasado cartas (pase recibido)",
    "turno_agente": "Es TU turno",
    "turno_arriba": "Turno del jugador de ARRIBA (enfrente)",
    "turno_izquierda": "Turno del jugador de la IZQUIERDA",
    "turno_derecha": "Turno del jugador de la DERECHA",
    "baza_agente": "TU recoges la baza",
    "baza_arriba": "Recoge la baza el de ARRIBA",
    "baza_izquierda": "Recoge la baza el de la IZQUIERDA",
    "baza_derecha": "Recoge la baza el de la DERECHA",
    "vacio": "(banner vacio)",
    "desconocido": "(banner no reconocido -> banner nuevo, re-correr agrupar_banners.py)",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnostico de un fotograma.")
    p.add_argument("--frame", required=True, help="PNG de la captura.")
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--banners", default="calibracion/hearts_app/banners")
    p.add_argument("--cartas", default="calibracion/hearts_app/cartas")
    p.add_argument("--completas", default="calibracion/hearts_app/cartas_completas",
                   help="Naipes completos del sprite (para leer la mano).")
    p.add_argument("--overlay", action="store_true",
                   help="Guardar overlay de regiones (_overlay.png junto al frame).")
    args = p.parse_args()

    import cv2

    from src.captura.modelos import carta_a_str
    from src.captura.vision_cartas import Reconocedor, ReconocedorPlantilla
    from src.captura.vision_hearts import (BannerClasificador, Regiones,
                                           leer_estado, leer_mano)

    img = cv2.imread(args.frame, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"No se pudo leer el frame: {args.frame}")
    h, w = img.shape[:2]
    print(f"Frame: {w}x{h}  ({args.frame})")

    reg = Regiones.cargar(args.regiones)
    clf = BannerClasificador(args.banners)
    rec = Reconocedor(args.cartas)
    rec_mano = ReconocedorPlantilla(args.completas)

    est = leer_estado(img, reg, clf, rec)

    print("\n== BANNER ==")
    desc = _DESC_BANNER.get(est.banner.tag, est.banner.tag)
    print(f"  {desc}")
    print(f"  (tag={est.banner.tag}  distancia={est.banner.distancia}  "
          f"categoria={est.banner.categoria}  dato={est.banner.dato})")

    print("\n== MESA (cartas jugadas, en cruz) ==")
    if all(v is None for v in est.mesa.values()):
        print("  (mesa vacia)")
    else:
        for pos, cid in est.mesa.items():
            print(f"  {pos:10s} -> {carta_a_str(cid) if cid is not None else '-'}")

    print("\n== TU MANO ==")
    mano = leer_mano(img, reg, rec_mano)
    n_ok = sum(1 for c in mano if c is not None)
    print(f"  cartas localizadas: {len(mano)}  |  rango reconocido: {n_ok}")
    print("  " + " ".join(carta_a_str(c) if c is not None else "??" for c in mano))
    if n_ok < len(mano):
        print("  (los '??' = rango no reconocido: re-genera plantillas de cartas "
              "de ESTE dispositivo con scripts/agrupar_cartas.py)")

    if args.overlay:
        from subprocess import run
        out = str(_Path(args.frame).with_name("_overlay.png"))
        run([_sys.executable, "scripts/calibrar_regiones.py",
             "--frame", args.frame, "--regiones", args.regiones,
             "--salida", out], check=False)


if __name__ == "__main__":
    main()
