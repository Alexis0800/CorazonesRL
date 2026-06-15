"""
Corazones — Reinforcement Learning agent for the card game Hearts.

Arquitectura:
    dominio/   — Pure game rules (no RL dependencies)
    entorno/   — Gymnasium/PettingZoo RL environments
    agentes/   — Playing strategies (heuristic bots, RL policies)
    torneo/    — Evaluation and Elo rating system
    entrenamiento/ — Training pipeline (self-play, auto-training)
    cli/       — Command-line interfaces (play, evaluate)

Uso:
    from src.dominio import Carta, MotorCorazones
    from src.agentes import bot_conservador, PoliticaSB3
    from src.entorno import CorazonesEnv
"""
# Re-export para acceso rápido
from src.dominio import Carta, Baraja, Jugador, MotorCorazones
from src.agentes import bot_conservador, bot_agresivo, bot_evasivo
