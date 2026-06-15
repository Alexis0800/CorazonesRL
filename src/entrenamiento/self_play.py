"""
Entorno self-play para entrenamiento del agente RL.

Crea entornos CorazonesEnv con oponentes mixtos (bots + snapshots históricos)
para entrenamiento multi-agente con Fictitious Self-Play.
"""

from __future__ import annotations

import os
import glob
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agentes.heuristicos import BOTS_DISPONIBLES
from src.agentes.politica_rl import PoliticaSB3
from src.entrenamiento.config import (
    directorio_snapshots_version, HP_DEFAULT, Hiperparametros,
)
from src.dominio.carta import Carta


def crear_entorno_self_play(
    version: str,
    prob_bot: float = 0.30,
    min_steps: int = 500_000,
    max_snapshots: int = 50,
    agente_idx: int = 0,
    modelo_actual: Optional[Any] = None,
    vecnorm_actual: Optional[str] = None,
) -> Any:
    """Crea un entorno CorazonesEnv con oponentes mixtos.

    Los oponentes son una mezcla de bots heurísticos (prob_bot) y
    snapshots históricos de versiones anteriores (1 - prob_bot).
    Esto implementa Fictitious Self-Play con curriculum learning.

    Args:
        version: Versión del modelo (ej: 'v8').
        prob_bot: Probabilidad de usar bot heurístico vs snapshot.
        min_steps: Pasos mínimos para un snapshot (madurez).
        max_snapshots: Máximo de snapshots en el pool.
        agente_idx: Índice del agente en el entorno.
        modelo_actual: Instancia MaskablePPO actual (para el agente).
        vecnorm_actual: Ruta al VecNormalize del agente.

    Returns:
        Instancia de CorazonesEnv configurada para self-play.
    """
    # Importación diferida (evita dependencia circular)
    from src.entorno import CorazonesEnv

    snaps_dir = directorio_snapshots_version(version)
    vecnorm_dir = os.path.join(os.path.dirname(snaps_dir), "vecnorm")

    # Construir pool de oponentes
    politicas: Dict[int, object] = {}

    # Cargar snapshots históricos
    snapshots = _cargar_snapshots_pool(snaps_dir, min_steps, max_snapshots)

    for rival_idx in range(4):
        if rival_idx == agente_idx:
            continue

        if random.random() < prob_bot:
            # Usar bot heurístico (elegir uno al azar)
            politicas[rival_idx] = random.choice(BOTS_DISPONIBLES)
        elif snapshots:
            # Usar snapshot histórico
            snap = random.choice(snapshots)
            vecnorm_path = _buscar_vecnorm(snap, vecnorm_dir)
            politicas[rival_idx] = PoliticaSB3(snap, rival_idx, vecnorm_path)
        else:
            # Fallback: bot aleatorio
            politicas[rival_idx] = BOTS_DISPONIBLES[0]

    return CorazonesEnv(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
    )


def _cargar_snapshots_pool(
    snaps_dir: str, min_steps: int, max_snapshots: int
) -> List[Any]:
    """Carga snapshots históricos desde el directorio estandarizado."""
    from stable_baselines3 import MaskablePPO

    if not os.path.exists(snaps_dir):
        return []

    archivos = glob.glob(os.path.join(snaps_dir, "snapshot_*.zip"))
    archivos_validos = []
    for a in archivos:
        nombre = os.path.basename(a).replace(".zip", "")
        try:
            paso = int(nombre.replace("snapshot_", ""))
            if paso >= min_steps:
                archivos_validos.append((paso, a))
        except ValueError:
            continue

    # Ordenar por paso, tomar los max_snapshots más recientes
    archivos_validos.sort(key=lambda x: x[0])
    seleccionados = archivos_validos[-max_snapshots:] if len(
        archivos_validos) > max_snapshots else archivos_validos

    snapshots = []
    for _, path in seleccionados:
        try:
            model = MaskablePPO.load(path, device="cpu")
            snapshots.append(model)
        except Exception:
            continue

    return snapshots


def _buscar_vecnorm(snap_path: str, vecnorm_dir: str) -> Optional[str]:
    """Busca el VecNormalize asociado a un snapshot."""
    # Primero: per-snapshot
    candidato = snap_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(candidato):
        return candidato
    # Segundo: directorio vecnorm de la versión
    candidato = os.path.join(vecnorm_dir, "vecnorm.pkl")
    if os.path.exists(candidato):
        return candidato
    return None
