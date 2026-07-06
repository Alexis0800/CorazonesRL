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


def _auditar_partida(path: Path) -> dict:
    lineas = []
    corruptas = []
    for n, l in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not l.strip():
            continue
        try:
            lineas.append(json.loads(l))
        except json.JSONDecodeError:
            corruptas.append(n)

    print(f"\n=== {path.name} ({len(lineas)} eventos, {len(corruptas)} corruptas) ===")
    for n in corruptas:
        print(f"  ⚠ línea {n} corrupta (2 requests concurrentes escribiendo al log a la vez), se ignora")

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
        elif l["evento"] in ("registrar_baza", "registrar_resto") and "puntos_mano" in l["salida"]:
            # Estos dos endpoints cierran la mano (13ª baza o remate del resto) y devuelven
            # puntos_mano en la SALIDA -- no hay un evento "registrar_puntos_mano" separado
            # desde que el cálculo se movió a Python (commit 0e4b9f7).
            pts = l["salida"]["puntos_mano"]
            suma = sum(pts)
            if suma not in (26, 78):
                print(f"  ⚠ puntos de mano suman {suma} (válido: 26 o 78) → {pts}")

    finales = [l["salida"]["scores"] for l in lineas
               if l["evento"] in ("registrar_baza", "registrar_resto") and "scores" in l["salida"]]
    scores = finales[-1] if finales else None
    puesto_me = None
    if scores:
        orden = sorted(range(4), key=lambda i: scores[i])
        puesto_me = orden.index(0) + 1  # ajustar si tu asiento no es 0
        print(f"  manos jugadas: {n_manos}  |  marcador final: {scores}  |  puesto agente (asiento 0): {puesto_me}")
    if not errores and not corruptas and n_manos:
        print("  ✓ sin errores ni invariantes rotas")

    return {
        "archivo": path.name,
        "n_manos": n_manos,
        "n_errores": len(errores),
        "n_corruptas": len(corruptas),
        "scores": scores,
        "puesto": puesto_me,
    }


def _imprimir_resumen(resultados: list) -> None:
    con_marcador = [r for r in resultados if r["puesto"] is not None]
    print(f"\n=== Resumen ({len(resultados)} partidas, {len(con_marcador)} con marcador final) ===")
    if not con_marcador:
        return

    conteo_puesto = {p: 0 for p in (1, 2, 3, 4)}
    for r in con_marcador:
        conteo_puesto[r["puesto"]] += 1
    n = len(con_marcador)
    for p in (1, 2, 3, 4):
        pct = 100 * conteo_puesto[p] / n
        print(f"  puesto {p}: {conteo_puesto[p]:3d}/{n} ({pct:5.1f}%)")
    promedio = sum(r["puesto"] for r in con_marcador) / n
    print(f"  puesto promedio: {promedio:.2f}  (1.0=siempre 1º, 4.0=siempre último; azar ≈ 2.50)")

    n_errores = sum(r["n_errores"] for r in resultados)
    n_corruptas = sum(r["n_corruptas"] for r in resultados)
    if n_errores or n_corruptas:
        print(f"  ⚠ {n_errores} errores y {n_corruptas} líneas corruptas en total (ver detalle arriba)")


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

    resultados = [_auditar_partida(Path(r)) for r in sorted(set(rutas))]
    _imprimir_resumen(resultados)


if __name__ == "__main__":
    main()
