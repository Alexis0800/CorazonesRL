"""
Captura por vision (banner + mesa) desde una fuente de fotogramas y, opcional,
VALIDA cada partida re-jugandola con el motor (replay). Mismo cerebro para:
  - validar offline sobre el video:   --fuente carpeta --carpeta videos/fotogramas
  - capturar en vivo por ADB:         --fuente adb

La validacion por replay es la prueba de que el dataset NO falla: si una mano
quedo mal leida, `reconstruir_manos` lo detecta (cartas != 13/asiento, orden).

Uso:
  python scripts/capturar_visual.py --fuente carpeta \
      --carpeta "videos/fotogramas" --paso 1 \
      --salida datos/capturas/video.jsonl --validar

  python scripts/capturar_visual.py --fuente adb --serial XXXX \
      --salida datos/capturas/sesion.jsonl
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

from src.captura.adaptador_visual import AdaptadorVisual, fuente_carpeta
from src.captura.escritor import EscritorJsonl
from src.captura.maquina import ROTACION_ANTIHORARIA, ROTACION_HORARIA
from src.captura.recolector import RecolectorPartidas
from src.captura.vision_cartas import Reconocedor
from src.captura.vision_hearts import BannerClasificador, Regiones


def _fuente_adb(serial, app, poll_s, max_frames, adb="adb"):
    import time

    from src.captura.adb import ClienteADB

    cli = ClienteADB(serial=serial, adb=adb)
    if not cli.dispositivos():
        raise SystemExit("No hay dispositivos ADB (revisa `adb devices`).")
    n = 0
    while max_frames <= 0 or n < max_frames:
        yield cli.captura()
        n += 1
        time.sleep(poll_s)


def _validar(partidas) -> None:
    from src.captura.replay import reconstruir_manos

    ok_manos = mal_manos = 0
    for p in partidas:
        for m in p.manos:
            try:
                reconstruir_manos(m)
                ok_manos += 1
            except Exception as e:
                mal_manos += 1
                print(f"  [mano {m.numero_mano}] jugadas={len(m.jugadas)} "
                      f"FALLA: {e}")
    total = ok_manos + mal_manos
    print(f"\nValidacion replay: {ok_manos}/{total} manos reconstruibles "
          f"({mal_manos} con problemas).")


def main() -> None:
    p = argparse.ArgumentParser(description="Captura por vision (carpeta/ADB).")
    p.add_argument("--fuente", choices=["carpeta", "adb"], default="carpeta")
    p.add_argument("--carpeta", default="videos/fotogramas")
    p.add_argument("--patron", default="*.png")
    p.add_argument("--paso", type=int, default=1)
    p.add_argument("--serial", default=None)
    p.add_argument("--adb", default="adb",
                   help="Ruta al ejecutable adb si no esta en el PATH.")
    p.add_argument("--poll", type=float, default=0.4)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--banners", default="calibracion/hearts_app/banners")
    p.add_argument("--cartas", default="calibracion/hearts_app/cartas")
    p.add_argument("--rotacion", choices=["horaria", "antihoraria"],
                   default="horaria")
    p.add_argument("--asiento-agente", type=int, default=0)
    p.add_argument("--salida", default=None, help="JSONL de salida (opcional).")
    p.add_argument("--validar", action="store_true",
                   help="Re-jugar cada partida con el motor para validarla.")
    p.add_argument("--solo-validas", action="store_true",
                   help="Escribir solo las manos reconstruibles (dataset limpio).")
    args = p.parse_args()

    reg = Regiones.cargar(args.regiones)
    banner_clf = BannerClasificador(args.banners)
    rec = Reconocedor(args.cartas)
    rot = ROTACION_HORARIA if args.rotacion == "horaria" else ROTACION_ANTIHORARIA

    if args.fuente == "carpeta":
        frames = fuente_carpeta(args.carpeta, args.patron, args.paso)
    else:
        frames = _fuente_adb(args.serial, "hearts", args.poll, args.max_frames,
                             adb=args.adb)

    adaptador = AdaptadorVisual(
        frames, reg, banner_clf, rec,
        asiento_agente=args.asiento_agente, rotacion=rot,
    )

    # Recolectamos en memoria; el filtrado/escritura va despues para poder
    # descartar manos no reconstruibles (dataset limpio) si --solo-validas.
    partidas = RecolectorPartidas(adaptador).ejecutar()
    n_manos = sum(len(p.manos) for p in partidas)
    n_jug = sum(len(m.jugadas) for p in partidas for m in p.manos)
    print(f"Capturado -> partidas: {len(partidas)} | manos: {n_manos} | "
          f"jugadas: {n_jug}")

    if args.solo_validas:
        from src.captura.replay import mano_reconstruible
        for p in partidas:
            p.manos = [m for m in p.manos if mano_reconstruible(m)]
        validas = sum(len(p.manos) for p in partidas)
        print(f"Tras filtrar por replay: {validas}/{n_manos} manos validas.")

    if args.salida:
        esc = EscritorJsonl(args.salida)
        for p in partidas:
            if p.manos or not args.solo_validas:
                esc.escribir(p)
        print(f"Escrito en {args.salida}")
    if args.validar:
        _validar(partidas)


if __name__ == "__main__":
    main()
