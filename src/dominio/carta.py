"""
Representación inmutable de una carta de la baraja francesa.

Cada carta tiene un palo (0-3) y un valor (2-14), donde:
    Palos: 0 = Tréboles, 1 = Diamantes, 2 = Picas, 3 = Corazones.
    Valores: 2-10 numéricos, 11 = J, 12 = Q, 13 = K, 14 = A.

Proporciona propiedades de conveniencia para las reglas del juego de Corazones.

Las 52 instancias se crean una sola vez a nivel de módulo (cache) y se reutilizan
en todas las manos para maximizar el rendimiento.
"""

from __future__ import annotations
from typing import List


class Carta:
    """Representa una carta única e inmutable de la baraja francesa."""

    __slots__ = ("_palo", "_valor", "_hash", "_id")

    # Cache global de las 52 cartas (inicializado al final del módulo)
    _TODAS: List[Carta] = []

    def __init__(self, palo: int, valor: int) -> None:
        if not (0 <= palo <= 3):
            raise ValueError(f"Palo inválido: {palo}. Debe estar entre 0 y 3.")
        if not (2 <= valor <= 14):
            raise ValueError(
                f"Valor inválido: {valor}. Debe estar entre 2 y 14.")
        self._palo: int = palo
        self._valor: int = valor
        self._hash: int = hash((palo, valor))
        self._id: int = palo * 13 + (valor - 2)

    @property
    def palo(self) -> int:
        return self._palo

    @property
    def valor(self) -> int:
        return self._valor

    @property
    def id(self) -> int:
        return self._id

    @property
    def es_corazon(self) -> bool:
        return self._palo == 3

    @property
    def es_dama_de_picas(self) -> bool:
        return self._palo == 2 and self._valor == 12

    @property
    def puntos(self) -> int:
        if self._palo == 3:
            return 1
        if self._palo == 2 and self._valor == 12:
            return 13
        return 0

    @property
    def es_dos_de_treboles(self) -> bool:
        return self._palo == 0 and self._valor == 2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Carta):
            return NotImplemented
        return self._palo == other._palo and self._valor == other._valor

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        nombres_palo = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
        nombres_valor = {11: "J", 12: "Q", 13: "K", 14: "A"}
        v = nombres_valor.get(self._valor, str(self._valor))
        p = nombres_palo.get(self._palo, str(self._palo))
        return f"Carta({v}{p})"


# --- Inicializar el cache global de las 52 cartas ---
Carta._TODAS = [Carta(palo, valor) for palo in range(4)
                for valor in range(2, 15)]

# Arrays precomputados para acceso rápido en el motor (indexados por carta.id)
_PALOS: List[int] = [c.palo for c in Carta._TODAS]
_PUNTOS: List[int] = [c.puntos for c in Carta._TODAS]
_ES_CORAZON: List[bool] = [c.es_corazon for c in Carta._TODAS]

__all__ = ["Carta", "_PALOS", "_PUNTOS", "_ES_CORAZON"]
