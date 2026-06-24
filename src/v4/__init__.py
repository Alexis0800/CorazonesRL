"""v4 — Terminal-only rewards + Self-play puro con bots heurísticos.

Cambios vs v3.1:
  - Recompensa terminal-only: 26 - puntos_agente
  - Sin BC pre-training
  - Sin MCTS BC fine-tune
  - Sin BotExperto (solo bots heurísticos en training)
  - Cosine LR schedule (1e-4 → 1e-6)
  - Multi-position rotation (4 sillas)
  - Curriculum 3 fases con self-play progresivo
"""
