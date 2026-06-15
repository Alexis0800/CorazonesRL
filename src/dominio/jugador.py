"""
Representación de un jugador en el juego de Corazones.

Almacena la mano actual, las bazas ganadas y la puntuación histórica acumulada
a lo largo de múltiples manos.
"""

from __future__ import annotations

from typing import List
from src.dominio.carta import Carta


class Jugador:
    """Jugador del juego de Corazones con su mano y estado de puntuación."""

    def __init__(self, nombre: str) -> None:
        self.nombre: str = nombre
        self.mano: List[Carta] = []
        self.bazas_ganadas: List[Carta] = []
        self.puntuacion_historica: int = 0

    def recibir_mano(self, cartas: List[Carta]) -> None:
        """Asigna una nueva mano de cartas al jugador."""
        self.mano = list(cartas)

    def jugar_carta(self, carta: Carta) -> Carta:
        """Remueve y retorna la carta especificada de la mano del jugador."""
        try:
            self.mano.remove(carta)
            return carta
        except ValueError:
            raise ValueError(
                f"El jugador {self.nombre} no tiene la carta {carta} en su mano."
            )

    def sumar_puntos(self, puntos: int) -> None:
        """Acumula puntos al historial del jugador."""
        self.puntuacion_historica += puntos

    def contar_puntos_bazas(self) -> int:
        """Calcula los puntos totales de las bazas ganadas en esta mano."""
        return sum(c.puntos for c in self.bazas_ganadas)


__all__ = ["Jugador"]
