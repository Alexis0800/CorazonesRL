"""
Dimensiones del vector de observación — Single Source of Truth.

Centraliza las constantes de dimensionalidad para que todos los módulos
(ObservacionBuilder, CorazonesEnv, config.py, train.py) referencien
un único lugar.

Principio: DRY (Don't Repeat Yourself) + SSOT (Single Source of Truth).

Estándar actual: 220 dimensiones (DIM_V10).
"""

from __future__ import annotations

from typing import Tuple

# --- Versiones de dimensionalidad (históricas) ---
DIM_V5: int = 190   # features básicas + flags estratégicos (pozo_viable, etc.)
DIM_V6: int = 194   # + all_void por palo (obsoleto, solo referencia histórica)
DIM_V10: int = 220  # + bloque avanzado (conteo, prob Q♠, alertas)
# + quien_jugo_mesa (4 flags: qué jugadores ya jugaron en la baza)
DIM_V11: int = 224
# + 40 features enriquecidas (v3): cartas_restantes, peligro_qs, liderazgo, etc.
DIM_V3: int = 265

# --- Defaults ---
DIM_ENTRENAMIENTO: int = DIM_V11  # usado por train.py
DIM_ENTORNO: int = DIM_V11        # usado por CorazonesEnv (estándar actual)

# --- Dimensiones aceptadas ---
DIMS_VALIDAS: Tuple[int, ...] = (DIM_V6, DIM_V10, DIM_V11, DIM_V3)

# --- Mapeo para referencia humana ---
DIM_NOMBRES = {
    DIM_V5: "v5 (190) — features básicas + flags estratégicos",
    DIM_V6: "v6 (194) — + all_void por palo (obsoleto)",
    DIM_V10: "v10 (220) — + bloque avanzado (conteo, prob Q♠, alertas)",
    DIM_V11: "v11 (224) — + quien_jugo_mesa (4 flags de jugadores en baza actual) ← actual",
}

__all__ = [
    "DIM_V5",
    "DIM_V6",
    "DIM_V10",
    "DIM_V11",
    "DIM_V3",
    "DIM_ENTRENAMIENTO",
    "DIM_ENTORNO",
    "DIMS_VALIDAS",
    "DIM_NOMBRES",
]
