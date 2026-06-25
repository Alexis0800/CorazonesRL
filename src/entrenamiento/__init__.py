"""
Entrenamiento — Configuración del pipeline de entrenamiento.
"""
from src.entrenamiento.config import (
    Hiperparametros, HP_DEFAULT,
    directorio_modelos_version, directorio_snapshots_version,
    directorio_vecnorm_version, directorio_elite_version,
    directorio_torneos_version,
)

__all__ = [
    "Hiperparametros", "HP_DEFAULT",
    "directorio_modelos_version", "directorio_snapshots_version",
    "directorio_vecnorm_version", "directorio_elite_version",
    "directorio_torneos_version",
]
