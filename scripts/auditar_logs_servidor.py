"""Audita los JSONL de `servidor_inferencia.py` en busca de desincronización
con la partida real, SIN necesitar anotaciones externas: se apoya en
invariantes del propio dominio (ver `calcular_puntuacion_mano` en motor.py).

Uso:
    python scripts/auditar_logs_servidor.py logs/servidor_inferencia_*.jsonl
    python scripts/auditar_logs_servidor.py --dir logs
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _auditar_partida(path: Path) -> None:
    lineas = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n=== {path.name} ({len(lineas)} eventos) ===")

    errores = [l for l in lineas if l["evento"].startswith("error:")]
    for e in errores:
        print(f"  ⚠ ERROR en {e['evento']}: {e['salida'].get('error')}")

    n_manos = 0
    for l in lineas:
        if l["evento"] == "reset_mano":
            n_manos += 1
            n_cartas = len(l["entrada"].get("cartas", []))
            if n_cartas not in (0, 13):
                print(f"  ⚠ reset_mano con {n_cartas} cartas (se esperaban 13)")
        elif l["evento"] == "registrar_puntos_mano":
            pts = l["entrada"]["puntos"]
            suma = sum(pts)
            if suma not in (26, 78):
                print(f"  ⚠ puntos de mano suman {suma} (válido: 26 o 78) → {pts}")

    finales = [l["estado"]["scores"] for l in lineas if l["evento"] == "registrar_puntos_mano"]
    if finales:
        scores = finales[-1]
        orden = sorted(range(4), key=lambda i: scores[i])
        puesto_me = orden.index(0) + 1  # ajustar si tu asiento no es 0
        print(f"  manos jugadas: {n_manos}  |  marcador final: {scores}  |  puesto agente (asiento 0): {puesto_me}")
    if not errores and n_manos:
        print("  ✓ sin errores ni invariantes rotas")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("patrones", nargs="*", help="Rutas o globs a archivos .jsonl")
    p.add_argument("--dir", help="Carpeta donde buscar servidor_inferencia_*.jsonl")
    args = p.parse_args()

    rutas = []
    for pat in args.patrones:
        rutas.extend(glob.glob(pat))
    if args.dir:
        rutas.extend(str(f) for f in Path(args.dir).glob("servidor_inferencia_*.jsonl"))
    if not rutas:
        p.error("pasa al menos un archivo/glob o --dir")

    for r in sorted(set(rutas)):
        _auditar_partida(Path(r))


if __name__ == "__main__":
    main()
