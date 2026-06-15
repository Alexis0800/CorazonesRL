"""
Dominio — Pure game rules for the card game Hearts (Corazones).

No dependencies on RL, ML, or Gymnasium.
"""
from src.dominio.carta import Carta
from src.dominio.baraja import Baraja
from src.dominio.jugador import Jugador
from src.dominio.motor import MotorCorazones

__all__ = ["Carta", "Baraja", "Jugador", "MotorCorazones"]
