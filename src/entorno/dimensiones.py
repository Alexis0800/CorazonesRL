"""
Dimensiones del vector de observación — Single Source of Truth.

Centraliza las constantes de dimensionalidad para que todos los módulos
(ObservacionBuilder, CorazonesEnvRLlib, config.py, scripts/train_rllib.py)
referencien un único lugar.

Principio: DRY (Don't Repeat Yourself) + SSOT (Single Source of Truth).

Estándar actual:
  - 224 (DIM_V11) = DIM_ENTORNO, el default del entorno SIN fase de pase.
  - 228 (DIM_V12) = entorno CON fase de pase (con_pase=True), usado por el
    modelo campeón v10c (partida completa + pase).
"""

from __future__ import annotations

from typing import Tuple

# --- Versiones de dimensionalidad (históricas) ---
DIM_V5: int = 190   # features básicas + flags estratégicos (pozo_viable, etc.)
DIM_V6: int = 194   # + all_void por palo (obsoleto, solo referencia histórica)
DIM_V10: int = 220  # + bloque avanzado (conteo, prob Q♠, alertas)
# + quien_jugo_mesa (4 flags: qué jugadores ya jugaron en la baza)
DIM_V11: int = 224
# v10b: + 4 features de la fase de PASE (fase_pase, direccion, n_seleccionadas, reservado)
DIM_V12: int = 228
# v13: + memoria del pase (2 planos de 52): cartas que DI al receptor (aún en juego)
# y cartas que RECIBÍ del dador (aún en mano). Por perspectiva de cada jugador.
DIM_V13: int = 332
# + 40 features enriquecidas (v3): cartas_restantes, peligro_qs, liderazgo, etc.
DIM_V3: int = 265

# --- Defaults ---
DIM_ENTRENAMIENTO: int = DIM_V11  # usado por scripts/train_rllib.py
DIM_ENTORNO: int = DIM_V11        # usado por CorazonesEnvRLlib (estándar actual)

# --- Constantes del dominio del juego ---
NUM_CARTAS: int = 52      # cartas de la baraja (one-hot de mano/mesa/acción)
BAZAS_POR_MANO: int = 13  # bazas en una mano completa (52 cartas / 4 jugadores)

# --- Dimensiones aceptadas ---
DIMS_VALIDAS: Tuple[int, ...] = (DIM_V6, DIM_V10, DIM_V11, DIM_V12, DIM_V13, DIM_V3)


def con_pase_de_obs(obs_dim: int) -> bool:
    """True si la obs incluye las features de la fase de pase (obs_dim >= DIM_V12)."""
    return obs_dim >= DIM_V12

__all__ = [
    "DIM_V5",
    "DIM_V6",
    "DIM_V10",
    "DIM_V11",
    "DIM_V12",
    "DIM_V13",
    "DIM_V3",
    "DIM_ENTRENAMIENTO",
    "DIM_ENTORNO",
    "NUM_CARTAS",
    "BAZAS_POR_MANO",
    "DIMS_VALIDAS",
    "con_pase_de_obs",
]
