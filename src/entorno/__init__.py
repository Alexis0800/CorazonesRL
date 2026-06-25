"""
Entorno — Componentes del entorno RL para Corazones.

Separa la construcción de observación (observacion.py), el cálculo
de recompensas (recompensas.py) y las constantes de dimensionalidad
(dimensiones.py).
"""
from src.entorno.observacion import ObservacionBuilder
from src.entorno.recompensas import RewardConfig, CalculadoraRecompensas
from src.entorno.dimensiones import (
    DIM_V5, DIM_V6, DIM_V10, DIM_ENTRENAMIENTO, DIM_ENTORNO, DIMS_VALIDAS,
)

__all__ = [
    "ObservacionBuilder", "RewardConfig", "CalculadoraRecompensas",
    "DIM_V5", "DIM_V6", "DIM_V10",
    "DIM_ENTRENAMIENTO", "DIM_ENTORNO", "DIMS_VALIDAS",
]
