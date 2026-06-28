"""
Corazones — Reinforcement Learning agent for the card game Hearts.

Arquitectura:
    dominio/   — Pure game rules (no RL dependencies)
    entorno/   — Gymnasium RL environment (observation, rewards, dimensions)
    agentes/   — Playing strategies (heuristic bots)
    rllib/     — Ray RLlib pipeline (model, config, self-play pool)
    torneo/    — Evaluation and Elo rating system
    mcts/      — PIMC oracle for dataset generation and analysis

Uso:
    from src.dominio import Carta, MotorCorazones
    from src.agentes import bot_conservador, BotCastigador
    from src.entorno import ObservacionBuilder, CalculadoraRecompensasPartida
"""
from src.dominio import Carta, Baraja, Jugador, MotorCorazones
from src.agentes import bot_conservador, bot_agresivo, bot_evasivo, BotCastigador
