"""
Visor de un dataset de captura (`.jsonl`): decodifica los ids de carta a texto
y muestra cada partida de forma legible para verificar que se capturó bien.

Uso:
    python scripts/inspeccionar_captura.py --jsonl datasets/humano/partidas.jsonl
    python scripts/inspeccionar_captura.py --jsonl ... --partida 0 --detalle
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

from src.captura.escritor import cargar_partidas
from src.captura.modelos import carta_a_str

_DIR = {"izquierda": "←", "derecha": "→", "enfrente": "↑", None: "·"}


def _cartas(ids):
    return " ".join(carta_a_str(c) for c in ids)


def _mostrar_partida(p, idx, detalle):
    print(f"\n══ Partida #{idx}  id={p.partida_id}  fuente={p.fuente} "
          f"agente=asiento{p.asiento_agente} ══")
    print(f"   marcador final: {p.marcador_final}   ranking(mejor→peor): {p.ranking_final}")
    for m in p.manos:
        n_jug = len(m.jugadas)
        print(f"  · Mano {m.numero_mano} (pase {_DIR.get(m.direccion_pase, '?')}) "
              f"| jugadas={n_jug}/52 | puntuación={m.puntuacion_mano}")
        print(f"      mano agente: {_cartas(m.mano_inicial_agente)}")
        if m.pase_dado:
            print(f"      pasa: {_cartas(m.pase_dado)}"
                  + (f"   recibe: {_cartas(m.pase_recibido)}" if m.pase_recibido else ""))
        if m.remate_asiento is not None:
            print(f"      resto → asiento {m.remate_asiento} (tras {len(m.jugadas)//4} bazas)")
            for s, h in enumerate(m.manos_restantes):
                print(f"        restantes a{s}: {_cartas(h)}")
        if detalle:
            por_baza = {}
            for j in m.jugadas:
                por_baza.setdefault(j.baza, []).append(j)
            for baza in sorted(por_baza):
                trozos = []
                for j in por_baza[baza]:
                    marca = "*" if j.asiento == p.asiento_agente else " "
                    trozos.append(f"{marca}a{j.asiento}:{carta_a_str(j.carta_id)}")
                print(f"      baza {baza:2d}: " + "  ".join(trozos))


def main() -> None:
    ap = argparse.ArgumentParser(description="Visor de capturas .jsonl")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--partida", type=int, default=None, help="Mostrar solo esta (índice).")
    ap.add_argument("--detalle", action="store_true", help="Mostrar baza por baza.")
    args = ap.parse_args()

    partidas = cargar_partidas(args.jsonl)
    print(f"{len(partidas)} partida(s) en {args.jsonl}")
    indices = [args.partida] if args.partida is not None else range(len(partidas))
    for i in indices:
        _mostrar_partida(partidas[i], i, args.detalle)

    # Resumen de integridad
    total_jug = sum(len(m.jugadas) for p in partidas for m in p.manos)
    manos = sum(len(p.manos) for p in partidas)

    def _cartas_totales(m):
        return len(m.jugadas) + sum(len(h) for h in m.manos_restantes)

    incompletas = [(i, m.numero_mano) for i, p in enumerate(partidas)
                   for m in p.manos if _cartas_totales(m) != 52]
    con_resto = sum(1 for p in partidas for m in p.manos if m.remate_asiento is not None)
    print(f"\nResumen: {manos} manos ({con_resto} con concesión), {total_jug} jugadas.")
    if incompletas:
        print(f"⚠ Manos con cartas != 52 (jugadas+restantes, revisar): {incompletas}")
    else:
        print("✅ Todas las manos suman 52 cartas (jugadas + restantes).")


if __name__ == "__main__":
    main()
