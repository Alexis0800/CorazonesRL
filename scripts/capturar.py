"""
Captura un dataset de partidas reales de Corazones a un `.jsonl`.

Fuentes:
  - manual : narras la partida por consola (sin ADB; usable hoy).
  - adb    : automático vía ADB sobre una app móvil (requiere calibrar el parser
             y deps de requirements-captura.txt).

Uso:
    python scripts/capturar.py --fuente manual --salida datasets/humano/partidas.jsonl
    python scripts/capturar.py --fuente adb --plantillas calibracion/cartas \
        --regiones calibracion/regiones.json --salida datasets/humano/partidas.jsonl
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
import json

from src.captura.escritor import EscritorJsonl
from src.captura.manual import AdaptadorManual
from src.captura.recolector import RecolectorPartidas


def _construir_adaptador(args):
    if args.fuente == "manual":
        return AdaptadorManual(asiento_agente=args.asiento, limite=args.limite)
    if args.fuente == "demo":
        from src.captura.simulado import AdaptadorSimulado
        return AdaptadorSimulado(asiento_agente=args.asiento, limite=args.limite,
                                 max_manos=args.manos_demo, seed=args.seed)
    # adb
    from src.captura.adb import AdaptadorADB, ClienteADB, ParserPlantillas

    regiones = {}
    if args.regiones:
        regiones = json.loads(_Path(args.regiones).read_text(encoding="utf-8"))
    parser = ParserPlantillas(plantillas_dir=args.plantillas, regiones=regiones)
    cliente = ClienteADB(serial=args.serial)
    return AdaptadorADB(parser=parser, cliente=cliente,
                        asiento_agente=args.asiento, app=args.app,
                        max_partidas=args.partidas)


def main() -> None:
    p = argparse.ArgumentParser(description="Recolector de partidas de Corazones.")
    p.add_argument("--fuente", choices=["manual", "demo", "adb"], default="manual")
    p.add_argument("--salida", default="datasets/humano/partidas.jsonl")
    p.add_argument("--asiento", type=int, default=0, help="Asiento del agente (0-3).")
    p.add_argument("--limite", type=int, default=100, help="Puntos para fin de partida.")
    p.add_argument("--partidas", type=int, default=1, help="(adb) nº de partidas.")
    # demo
    p.add_argument("--manos-demo", type=int, default=2, help="(demo) manos a simular.")
    p.add_argument("--seed", type=int, default=0, help="(demo) semilla.")
    # adb
    p.add_argument("--serial", default=None, help="(adb) serial del dispositivo.")
    p.add_argument("--app", default="desconocida", help="(adb) etiqueta de la app.")
    p.add_argument("--plantillas", default="calibracion/cartas")
    p.add_argument("--regiones", default=None, help="(adb) JSON con cajas calibradas.")
    args = p.parse_args()

    escritor = EscritorJsonl(args.salida)
    adaptador = _construir_adaptador(args)
    recolector = RecolectorPartidas(adaptador=adaptador, escritor=escritor)

    completadas = recolector.ejecutar()
    print(f"\n✅ {len(completadas)} partida(s) guardada(s) en {args.salida}")


if __name__ == "__main__":
    main()
