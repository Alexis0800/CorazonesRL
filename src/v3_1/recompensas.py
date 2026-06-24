"""
Sistema de recompensas para v3.1 — re-exporta desde v2_1.

v3.1 usa el mismo sistema de recompensas que v2_1/v3.
La simplificación de rewards (Fase 3) se implementará después.
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
