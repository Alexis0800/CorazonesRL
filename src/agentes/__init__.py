"""
Agentes — Estrategias de juego para Corazones.

Incluye bots heurísticos (Strategy Pattern): (motor, idx, legales) → Carta.
"""
from src.agentes.heuristicos import (
    bot_conservador, bot_agresivo, bot_evasivo,
    BOTS_DISPONIBLES, BOT_NOMBRES, PoliticaJuego,
)
from src.agentes.bot_castigador import BotCastigador
from src.agentes.bot_experto import BotExperto

__all__ = [
    "bot_conservador", "bot_agresivo", "bot_evasivo",
    "BotCastigador", "BotExperto",
    "BOTS_DISPONIBLES", "BOT_NOMBRES", "PoliticaJuego",
]
