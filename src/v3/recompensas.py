"""
Sistema de recompensas para v3 — re-exporta desde v2_1.

v3 usa exactamente las mismas 8 señales que v2_1 (4 base + 4 tácticas).
No hay cambios en la función de recompensa respecto a v2_1.
"""

from __future__ import annotations

from src.v2_1.recompensas import (
    CalculadoraRecompensasV21,
    RewardConfigV21,
)

__all__ = [
    "CalculadoraRecompensasV21",
    "RewardConfigV21",
]
