"""
Escritura/lectura del dataset de partidas en formato JSONL (una partida/línea).

Append-only: cada partida capturada se añade al final, de modo que una caída a
mitad de sesión conserva todo lo ya recolectado.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from src.captura.modelos import RegistroPartida


class EscritorJsonl:
    """Escribe `RegistroPartida` en append a un `.jsonl` (UTF-8, una por línea)."""

    def __init__(self, ruta: str | Path) -> None:
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)

    def escribir(self, partida: RegistroPartida) -> None:
        linea = json.dumps(partida.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self.ruta.open("a", encoding="utf-8") as f:
            f.write(linea + "\n")


def leer_partidas(ruta: str | Path) -> Iterator[RegistroPartida]:
    """Itera las partidas de un `.jsonl` (salta líneas en blanco)."""
    p = Path(ruta)
    with p.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            yield RegistroPartida.from_dict(json.loads(linea))


def cargar_partidas(ruta: str | Path) -> List[RegistroPartida]:
    return list(leer_partidas(ruta))


__all__ = ["EscritorJsonl", "leer_partidas", "cargar_partidas"]
