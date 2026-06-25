"""
src/rllib/ — Módulos específicos de Ray RLlib para Corazones.

Estructura:
    model.py         — Modelo PyTorch con action masking (MLP, luego LSTM)
    config.py        — PPOConfig builder
    opponent_pool.py — Pool de oponentes para self-play
    callbacks.py     — Custom callbacks (métricas, snapshots)
    utils.py         — Utilidades de snapshots
"""
