"""
Monitor en vivo por ADB: imprime en tiempo real lo que la vision ve en la app
(banner -> fase/direccion de pase/turno, cartas de la mesa y tu mano). SOLO
observa (screencap); no toca la pantalla.

A diferencia de `capturar_visual.py` (que graba un dataset JSONL), esto es un
visor: refresca la consola cada poll para que veas, mientras juegas, que esta
leyendo el cerebro de vision.

Uso:
  python scripts/monitor_vivo.py                       # primer dispositivo ADB
  python scripts/monitor_vivo.py --serial XXXX --poll 0.5
  python scripts/monitor_vivo.py --frame captura.png   # un frame fijo (sin ADB)
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
import os
import time

_DESC_BANNER = {
    "pase_izquierda": "FASE DE PASE -> pasa 3 a la IZQUIERDA",
    "pase_derecha": "FASE DE PASE -> pasa 3 a la DERECHA",
    "pase_enfrente": "FASE DE PASE -> pasa 3 de ENFRENTE",
    "pase_recibido": "Pase recibido",
    "turno_agente": "Es TU turno",
    "turno_arriba": "Turno del de ARRIBA",
    "turno_izquierda": "Turno del de la IZQUIERDA",
    "turno_derecha": "Turno del de la DERECHA",
    "baza_agente": "TU recoges la baza",
    "baza_arriba": "Recoge la baza el de ARRIBA",
    "baza_izquierda": "Recoge la baza el de la IZQUIERDA",
    "baza_derecha": "Recoge la baza el de la DERECHA",
    "vacio": "(banner vacio)",
    "desconocido": "(banner no reconocido)",
}


def _render(est, mano, carta_a_str, poll, n):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"== MONITOR VIVO (poll {poll}s, frame #{n})  Ctrl+C para salir ==\n")
    desc = _DESC_BANNER.get(est.banner.tag, est.banner.tag)
    print(f"BANNER: {desc}   (dist={est.banner.distancia})\n")
    mesa = {p: (carta_a_str(c) if c is not None else "-")
            for p, c in est.mesa.items()}
    print("MESA:")
    print(f"        {mesa.get('arriba','-'):>4}")
    print(f"  {mesa.get('izquierda','-'):>4}        {mesa.get('derecha','-'):>4}")
    print(f"        {mesa.get('abajo','-'):>4}\n")
    n_ok = sum(1 for c in mano if c is not None)
    print(f"TU MANO ({len(mano)} cartas, {n_ok} con rango leido):")
    print("  " + " ".join(carta_a_str(c) if c is not None else "??" for c in mano))


def _frames_adb(serial, adb, poll, max_frames):
    from src.captura.adb import ClienteADB
    cli = ClienteADB(serial=serial, adb=adb)
    if not cli.dispositivos():
        raise SystemExit("No hay dispositivos ADB (revisa `adb devices`).")
    n = 0
    while max_frames <= 0 or n < max_frames:
        yield cli.captura()
        n += 1
        time.sleep(poll)


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor en vivo de la vision (ADB).")
    p.add_argument("--serial", default=None)
    p.add_argument("--adb", default="adb")
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--frame", default=None,
                   help="Un PNG fijo (modo prueba sin ADB).")
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--banners", default="calibracion/hearts_app/banners")
    p.add_argument("--cartas", default="calibracion/hearts_app/cartas")
    p.add_argument("--completas", default="calibracion/hearts_app/cartas_completas")
    args = p.parse_args()

    import cv2

    from src.captura.modelos import carta_a_str
    from src.captura.vision_cartas import Reconocedor, ReconocedorPlantilla
    from src.captura.vision_hearts import (BannerClasificador, Regiones,
                                           leer_estado, leer_mano)

    reg = Regiones.cargar(args.regiones)
    clf = BannerClasificador(args.banners)
    rec = Reconocedor(args.cartas)
    rec_mano = ReconocedorPlantilla(args.completas)

    if args.frame:
        img = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"No se pudo leer el frame: {args.frame}")
        frames = iter([img])
    else:
        frames = _frames_adb(args.serial, args.adb, args.poll, args.max_frames)

    try:
        for n, img in enumerate(frames):
            if img is None:
                continue
            est = leer_estado(img, reg, clf, rec)
            mano = leer_mano(img, reg, rec_mano)
            _render(est, mano, carta_a_str, args.poll, n)
            if args.frame:
                break
    except KeyboardInterrupt:
        print("\nFin.")


if __name__ == "__main__":
    main()
