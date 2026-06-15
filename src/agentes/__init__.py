"""
Agentes — Estrategias de juego para Corazones.

Incluye bots heurísticos (Strategy Pattern) y adaptadores para modelos RL.
"""
from src.agentes.heuristicos import (
    bot_conservador, bot_agresivo, bot_evasivo,
    BOTS_DISPONIBLES, BOT_NOMBRES, PoliticaJuego,
)
from src.agentes.politica_rl import PoliticaSB3

__all__ = [
    "bot_conservador", "bot_agresivo", "bot_evasivo",
    "BOTS_DISPONIBLES", "BOT_NOMBRES", "PoliticaJuego",
    "PoliticaSB3",
]
