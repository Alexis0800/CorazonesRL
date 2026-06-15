"""
Baraja francesa de 52 cartas para el juego de Corazones.

Garantiza la unicidad de las cartas y permite barajar y repartir
manos completas a los 4 jugadores.

Utiliza las instancias cacheadas de Carta._TODAS para evitar
creación de objetos por mano (optimización de rendimiento).
"""

from __future__ import annotations

import random
from typing import List
from src.dominio.carta import Carta
from src.dominio.jugador import Jugador


class Baraja:
    """Baraja completa de 52 cartas francesas (referencias cacheadas)."""

    def __init__(self) -> None:
        self.cartas: List[Carta] = list(Carta._TODAS)

    def barajar(self) -> None:
        """Baraja las cartas aleatoriamente in-place."""
        random.shuffle(self.cartas)

    def repartir(self, jugadores: List[Jugador]) -> None:
        """Baraja y reparte 13 cartas a cada uno de los 4 jugadores."""
        if len(jugadores) != 4:
            raise ValueError(
                "Se requieren exactamente 4 jugadores para repartir.")
        self.barajar()
        for i, jugador in enumerate(jugadores):
            inicio = i * 13
            fin = inicio + 13
            jugador.recibir_mano(self.cartas[inicio:fin])


__all__ = ["Baraja"]
