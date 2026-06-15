"""
Torneo — Sistema de evaluación y rating Elo.

Los módulos completos están en src/elo_torneo.py y src/evaluacion.py.
Las versiones en src/corazones/torneo/ son wrappers parciales.
"""
from src.torneo.normalizacion import (
    normalizar_obs_desde_archivo, detectar_vecnorm, cargar_vecnorm_stats,
)

__all__ = [
    "normalizar_obs_desde_archivo", "detectar_vecnorm", "cargar_vecnorm_stats",
]
