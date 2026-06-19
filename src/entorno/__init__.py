"""
Entorno — Gymnasium/PettingZoo RL environments para Corazones.

Separa la construcción de observación (observacion.py), el cálculo
de recompensas (recompensas.py) y las constantes de dimensionalidad
(dimensiones.py) del ciclo de vida del entorno.
"""
from src.entorno.observacion import ObservacionBuilder
from src.entorno.recompensas import RewardConfig, CalculadoraRecompensas
from src.entorno.dimensiones import (
    DIM_V5, DIM_V6, DIM_V10, DIM_ENTRENAMIENTO, DIM_ENTORNO, DIMS_VALIDAS,
)
from src.entorno.single_agent import CorazonesEnv
from src.entorno.multi_agent import CorazonesAEC

__all__ = [
    "ObservacionBuilder", "RewardConfig", "CalculadoraRecompensas",
    "DIM_V5", "DIM_V6", "DIM_V10",
    "DIM_ENTRENAMIENTO", "DIM_ENTORNO", "DIMS_VALIDAS",
    "CorazonesEnv", "CorazonesAEC",
]
