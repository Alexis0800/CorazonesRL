"""
Puerto (interfaz) entre el recolector y la fuente del juego.

Aplica DIP: el `RecolectorPartidas` depende de esta abstracción, no de ADB ni
de la consola. Cualquier fuente (app móvil vía ADB, entrada manual, un mock de
test) implementa `AdaptadorJuego` emitiendo un **stream de eventos**.

La granularidad de evento (una jugada observada a la vez) es la unidad mínima
común a todas las fuentes: la cámara/parsing ve cartas aparecer en la mesa una a
una, y un humano puede narrarlas igual. El recolector agrega esos eventos en
`RegistroPartida` (ver `modelos.py`) sin saber de dónde vienen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Union


# --- Eventos del stream ---------------------------------------------------

@dataclass
class InicioPartida:
    asiento_agente: int          # asiento del agente/jugador observado (0-3)
    fuente: str                  # "manual" | "adb:<app>" | ...
    partida_id: Optional[str] = None


@dataclass
class InicioMano:
    numero_mano: int
    direccion_pase: Optional[str]        # izquierda/derecha/enfrente/None
    mano_agente: List[int] = field(default_factory=list)  # 13 ids PRE-pase


@dataclass
class PaseAgente:
    dadas: List[int]                      # 3 ids que el agente pasó
    recibidas: List[int] = field(default_factory=list)    # 3 ids recibidas (si se ven)


@dataclass
class JugadaObservada:
    asiento: int                          # quién jugó (0-3 absoluto)
    carta_id: int
    baza: int


@dataclass
class RemateResto:
    """Concesión: `asiento` se lleva todas las bazas restantes ("se llevará el resto").

    `manos_restantes` son las cartas reveladas de cada asiento (4 listas) en el
    momento de conceder. Se emite tras la última `JugadaObservada` real y antes
    de `FinMano`. Las bazas concedidas NO se emiten como jugadas.
    """
    asiento: int
    manos_restantes: List[List[int]]      # cartas reveladas por asiento (4 listas)


@dataclass
class FinMano:
    puntuacion: List[int]                 # 4, por asiento


@dataclass
class FinPartida:
    marcador: List[int]                   # 4, acumulado final


Evento = Union[
    InicioPartida, InicioMano, PaseAgente, JugadaObservada,
    RemateResto, FinMano, FinPartida,
]


# --- Puerto ---------------------------------------------------------------

class AdaptadorJuego(ABC):
    """Fuente de eventos de una partida real. Implementaciones: manual, ADB."""

    @abstractmethod
    def eventos(self) -> Iterator[Evento]:
        """Genera los eventos de una o varias partidas hasta agotar la sesión.

        Debe emitir, por partida: `InicioPartida`, y por mano `InicioMano`,
        opcional `PaseAgente`, una `JugadaObservada` por cada carta jugada
        (52 en una mano completa), `FinMano`, y al terminar la partida
        `FinPartida`. Si un asiento concede el resto, emite `RemateResto`
        (tras la última `JugadaObservada`, antes de `FinMano`); en ese caso
        las bazas concedidas NO se emiten como jugadas, por lo que la mano
        tiene menos de 52 `JugadaObservada`.
        """
        raise NotImplementedError

    def cerrar(self) -> None:
        """Libera recursos (conexión ADB, ficheros, etc.). Idempotente."""


__all__ = [
    "AdaptadorJuego", "Evento",
    "InicioPartida", "InicioMano", "PaseAgente",
    "JugadaObservada", "RemateResto", "FinMano", "FinPartida",
]
