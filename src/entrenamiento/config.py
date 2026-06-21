"""
Configuración de entrenamiento — hiperparámetros y paths centralizados.

Single Source of Truth (SSOT) para todos los paths y parámetros
del pipeline de entrenamiento. Elimina constantes duplicadas entre
train_auto_v6.py y train_self_play.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.entorno.dimensiones import DIM_ENTRENAMIENTO


# ------------------------------------------------------------------
# Paths base
# ------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))

DIRECTORIO_MODELOS = os.path.join(_PROJECT_DIR, "models")
DIRECTORIO_LOGS = os.path.join(_PROJECT_DIR, "logs")


def directorio_modelos_version(version: str) -> str:
    """Retorna el path estandarizado para una versión de modelos."""
    return os.path.join(DIRECTORIO_MODELOS, version)


def directorio_snapshots_version(version: str) -> str:
    """Retorna el path de snapshots para una versión."""
    return os.path.join(DIRECTORIO_MODELOS, version, "snapshots")


def directorio_vecnorm_version(version: str) -> str:
    """Retorna el path de VecNormalize para una versión."""
    return os.path.join(DIRECTORIO_MODELOS, version, "vecnorm")


def directorio_elite_version(version: str) -> str:
    """Retorna el path de elite (best) para una versión."""
    return os.path.join(DIRECTORIO_MODELOS, version, "elite")


def directorio_torneos_version(version: str) -> str:
    """Retorna el path de torneos para una versión."""
    return os.path.join(DIRECTORIO_MODELOS, version, "torneos")


# ------------------------------------------------------------------
# Hiperparámetros de entrenamiento
# ------------------------------------------------------------------

@dataclass
class Hiperparametros:
    """Hiperparámetros del pipeline de entrenamiento (immutable)."""
    # Self-play
    prob_bot_start: float = 0.50
    # v19b: compromiso entre 0.30 (v18) y 0.15 (v19)
    prob_bot_end: float = 0.20
    min_snapshot_steps: int = 500_000
    max_snapshots_pool: int = 50

    # PPO
    learning_rate: float = 3e-5
    n_steps: int = 4096
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.12
    vf_coef: float = 0.25        # v12: 0.5→0.25 (estabilizar value head)
    # v12: 0.5→0.3 (prevenir gradientes explosivos)
    max_grad_norm: float = 0.3

    # Evaluación
    eval_partidas: int = 100
    elo_partidas: int = 30
    elo_max_snapshots: int = 12
    best_top: int = 2

    # Arquitectura
    net_arch: List[int] = field(default_factory=lambda: [512, 512, 256])
    features_dim: int = 256

    # Observación — SSOT en src/entorno/dimensiones.py
    dim_observacion: int = DIM_ENTRENAMIENTO


# Instancia por defecto
HP_DEFAULT = Hiperparametros()
