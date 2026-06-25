"""v5 — 4-model shared-weights self-play simultáneo.

Cambios vs v4:
  - 4 entornos en paralelo (DummyVecEnv × 4 posiciones)
  - El modelo recibe reward de las 4 sillas simultáneamente
  - Zero-sum rewards: suma de las 4 rewards = 78
  - 4× más datos por update de PPO
  - Sin BC, sin MCTS fine-tune, terminal-only rewards
"""
