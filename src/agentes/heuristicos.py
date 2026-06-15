"""
Bots heurísticos para el juego de Corazones.

Proporciona estrategias de juego basadas en reglas que sirven como
oponentes de entrenamiento para el agente RL y como baseline en torneos.

Strategy Pattern: Cada bot es una función con la misma firma
    (motor, jugador_idx, legales) → Carta

Esto permite intercambiarlos sin modificar el código que los consume
(Open/Closed Principle).
"""

from __future__ import annotations

from typing import Callable, List
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones

# Tipo para políticas de juego (Strategy)
PoliticaJuego = Callable[[MotorCorazones, int, List[Carta]], Carta]


def bot_conservador(
    motor: MotorCorazones,
    jugador_idx: int,
    legales: List[Carta],
) -> Carta:
    """Estrategia conservadora: jugar la carta legal de menor valor.

    Prioriza deshacerse de cartas bajas para evitar ganar bazas no deseadas.
    """
    return min(legales, key=lambda c: c.valor)


def bot_agresivo(
    motor: MotorCorazones,
    jugador_idx: int,
    legales: List[Carta],
) -> Carta:
    """Estrategia agresiva: ganar bazas o fugarse de palos no deseados.

    Si puede seguir el palo, juega la más alta para ganar la baza.
    Si está void, juega la más alta para deshacerse de cartas peligrosas.
    """
    if motor.mesa and motor.palo_de_salida is not None:
        mismo_palo = [c for c in legales if c.palo == motor.palo_de_salida]
        if mismo_palo:
            return max(mismo_palo, key=lambda c: c.valor)
    return max(legales, key=lambda c: c.valor)


def bot_evasivo(
    motor: MotorCorazones,
    jugador_idx: int,
    legales: List[Carta],
) -> Carta:
    """Estrategia evasiva: evitar cartas de puntos a toda costa.

    Prioriza jugar cartas sin puntos. Si debe seguir el palo, elige la
    más baja sin puntos. Si está void, descarta la más alta con puntos.
    """
    sin_puntos = [c for c in legales if c.puntos == 0]
    con_puntos = [c for c in legales if c.puntos > 0]

    if motor.mesa and motor.palo_de_salida is not None:
        mismo_palo = [c for c in legales if c.palo == motor.palo_de_salida]
        if mismo_palo:
            sin_puntos_mismo_palo = [c for c in mismo_palo if c.puntos == 0]
            if sin_puntos_mismo_palo:
                return min(sin_puntos_mismo_palo, key=lambda c: c.valor)
            return min(mismo_palo, key=lambda c: c.valor)

    if sin_puntos:
        return min(sin_puntos, key=lambda c: c.valor)
    return max(con_puntos, key=lambda c: c.valor)


# Pool de bots disponibles
BOTS_DISPONIBLES: List[PoliticaJuego] = [
    bot_conservador, bot_agresivo, bot_evasivo]
BOT_NOMBRES: List[str] = ["conservador", "agresivo", "evasivo"]

__all__ = [
    "PoliticaJuego",
    "bot_conservador", "bot_agresivo", "bot_evasivo",
    "BOTS_DISPONIBLES", "BOT_NOMBRES",
]
