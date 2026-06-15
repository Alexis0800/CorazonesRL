"""
Entrenamiento — Pipeline de self-play y auto-entrenamiento.
"""
from src.entrenamiento.config import (
    Hiperparametros, HP_DEFAULT,
    directorio_modelos_version, directorio_snapshots_version,
    directorio_vecnorm_version, directorio_elite_version,
    directorio_torneos_version,
)
from src.entrenamiento.self_play import crear_entorno_self_play

__all__ = [
    "Hiperparametros", "HP_DEFAULT",
    "directorio_modelos_version", "directorio_snapshots_version",
    "directorio_vecnorm_version", "directorio_elite_version",
    "directorio_torneos_version",
    "crear_entorno_self_play",
]
