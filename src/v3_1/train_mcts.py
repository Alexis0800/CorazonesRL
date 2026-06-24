"""
MCTS-guided training para v3.1 — Buffer BC + Oráculo PIMC.

Re-exporta desde v3.train_mcts (la lógica de MCTS no depende de la dimensión).
"""

from __future__ import annotations

from src.v3.train_mcts import (
    MCTSBuffer,
    calcular_bc_loss,
    entrenar_bc_dataset,
    entrenar_bc_epoch,
    evaluar_con_oraculo,
    evaluar_y_guardar_batch,
)

__all__ = [
    "MCTSBuffer",
    "calcular_bc_loss",
    "entrenar_bc_dataset",
    "entrenar_bc_epoch",
    "evaluar_con_oraculo",
    "evaluar_y_guardar_batch",
]
