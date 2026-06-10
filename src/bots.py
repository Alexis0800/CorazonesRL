"""
Bots heurísticos para el juego de Corazones (Módulo 3).

Proporciona tres estrategias básicas basadas en reglas que sirven como
oponentes de entrenamiento para el agente RL en la Fase 1 del pipeline.

Cada bot recibe (motor, jugador_idx, legales) y retorna una Carta.

No dependen de PyTorch ni de Gymnasium.
"""

from __future__ import annotations

from typing import List
from src.carta import Carta
from src.motor import MotorCorazones


def bot_conservador(
    motor: MotorCorazones,
    jugador_idx: int,
    legales: List[Carta],
) -> Carta:
    """Estrategia conservadora: jugar la carta legal de menor valor.

    Prioriza deshacerse de cartas bajas para evitar ganar bazas
    no deseadas. Es un bot pasivo que minimiza riesgos.

    Args:
        motor: Referencia al motor del juego (no utilizada directamente).
        jugador_idx: Índice del jugador controlado por el bot.
        legales: Lista de cartas que puede jugar legalmente.

    Returns:
        La carta de menor valor entre las legales.
    """
    return min(legales, key=lambda c: c.valor)


def bot_agresivo(
    motor: MotorCorazones,
    jugador_idx: int,
    legales: List[Carta],
) -> Carta:
    """Estrategia agresiva: intentar ganar bazas o fugarse de palos no deseados.

    Si puede seguir el palo, juega la carta más alta para ganar la baza.
    Si está void (no puede seguir el palo), juega la carta de mayor valor
    para deshacerse de cartas peligrosas (corazones, dama de picas).

    Args:
        motor: Referencia al motor del juego.
        jugador_idx: Índice del jugador controlado por el bot.
        legales: Lista de cartas que puede jugar legalmente.

    Returns:
        La carta seleccionada según estrategia agresiva.
    """
    # Si puede seguir el palo de salida, jugar la más alta
    if motor.mesa and motor.palo_de_salida is not None:
        mismo_palo = [c for c in legales if c.palo == motor.palo_de_salida]
        if mismo_palo:
            return max(mismo_palo, key=lambda c: c.valor)

    # Si está void o abre la baza, jugar la carta más alta
    return max(legales, key=lambda c: c.valor)


def bot_evasivo(
    motor: MotorCorazones,
    jugador_idx: int,
    legales: List[Carta],
) -> Carta:
    """Estrategia evasiva: evitar cartas de puntos a toda costa.

    Prioriza jugar cartas sin puntos. Si debe seguir el palo y tiene
    opciones sin puntos, elige la más baja entre ellas. Si está void,
    descarta la carta de puntos más alta para minimizar daño futuro.

    Args:
        motor: Referencia al motor del juego.
        jugador_idx: Índice del jugador controlado por el bot.
        legales: Lista de cartas que puede jugar legalmente.

    Returns:
        La carta seleccionada según estrategia evasiva.
    """
    # Separar cartas con y sin puntos
    sin_puntos = [c for c in legales if c.puntos == 0]
    con_puntos = [c for c in legales if c.puntos > 0]

    # Si está siguiendo el palo (mesa no vacía) y es el palo de salida
    if motor.mesa and motor.palo_de_salida is not None:
        mismo_palo = [c for c in legales if c.palo == motor.palo_de_salida]
        if mismo_palo:
            sin_puntos_mismo_palo = [c for c in mismo_palo if c.puntos == 0]
            if sin_puntos_mismo_palo:
                # Jugar la más baja sin puntos del palo de salida
                return min(sin_puntos_mismo_palo, key=lambda c: c.valor)
            # Si todas las del palo tienen puntos, jugar la más baja
            return min(mismo_palo, key=lambda c: c.valor)

    # Si está void (no puede seguir palo) o abre la baza
    if sin_puntos:
        # Jugar la más baja sin puntos
        return min(sin_puntos, key=lambda c: c.valor)

    # Si todas las legales tienen puntos, jugar la más alta
    # (deshacerse de cartas peligrosas)
    return max(con_puntos, key=lambda c: c.valor)
