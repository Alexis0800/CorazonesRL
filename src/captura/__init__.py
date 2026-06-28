"""
Captura — recolección de dataset de partidas reales (vs humanos).

Arquitectura (SOLID):
    puerto.py     — AdaptadorJuego (ABC) + eventos. Frontera DIP.
    manual.py     — AdaptadorManual (consola, usable sin ADB).
    adb.py        — ClienteADB + ParserPantalla + AdaptadorADB (vía ADB).
    recolector.py — RecolectorPartidas: eventos → RegistroPartida.
    escritor.py   — EscritorJsonl / lectura del dataset .jsonl.
    modelos.py    — DTOs persistibles + helpers carta<->texto.
    replay.py     — encoder offline: RegistroPartida → (obs, acción) (pura).

`adb.py` necesita deps opcionales (`requirements-captura.txt`); el resto solo
depende de `dominio`/`entorno`. Ver docs/ROADMAP.md (Fase 2-3).
"""
from src.captura.puerto import (
    AdaptadorJuego, InicioPartida, InicioMano, PaseAgente,
    JugadaObservada, FinMano, FinPartida,
)
from src.captura.modelos import (
    Jugada, RegistroMano, RegistroPartida, carta_a_str, str_a_carta_id,
)
from src.captura.escritor import EscritorJsonl, leer_partidas, cargar_partidas
from src.captura.recolector import RecolectorPartidas
from src.captura.manual import AdaptadorManual

__all__ = [
    "AdaptadorJuego", "InicioPartida", "InicioMano", "PaseAgente",
    "JugadaObservada", "FinMano", "FinPartida",
    "Jugada", "RegistroMano", "RegistroPartida", "carta_a_str", "str_a_carta_id",
    "EscritorJsonl", "leer_partidas", "cargar_partidas",
    "RecolectorPartidas", "AdaptadorManual",
]
