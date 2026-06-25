"""
Utilidades de snapshots para el pipeline de entrenamiento RLlib.

Funciones para guardar y listar checkpoints de políticas, gestionando
la rotación y el pool de snapshots históricos.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple


def guardar_snapshot(
    algo,
    paso: int,
    snapshot_dir: str,
) -> str:
    """Guarda un checkpoint de la política 'default_policy' (o 'main').

    Args:
        algo:         Instancia del algoritmo RLlib en entrenamiento.
        paso:         Número de paso global (timesteps totales).
        snapshot_dir: Directorio donde guardar los snapshots.

    Returns:
        Ruta al checkpoint guardado.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    nombre = f"snapshot_{paso:012d}"
    ruta = os.path.join(snapshot_dir, nombre)
    algo.save(checkpoint_dir=ruta)
    return ruta


def listar_snapshots(snapshot_dir: str) -> List[Tuple[int, str]]:
    """Lista snapshots disponibles en el directorio, ordenados por paso.

    Returns:
        Lista de (paso, ruta) ordenada de más antiguo a más reciente.
    """
    if not os.path.isdir(snapshot_dir):
        return []

    result: List[Tuple[int, str]] = []
    for nombre in os.listdir(snapshot_dir):
        m = re.match(r"snapshot_(\d+)$", nombre)
        if m:
            paso = int(m.group(1))
            result.append((paso, os.path.join(snapshot_dir, nombre)))

    result.sort(key=lambda x: x[0])
    return result


def cargar_policy_desde_checkpoint(checkpoint_path: str, policy_id: str = "default_policy"):
    """Carga una política RLlib desde un checkpoint.

    Args:
        checkpoint_path: Ruta al directorio del checkpoint.
        policy_id:       ID de la política a cargar.

    Returns:
        Objeto Policy de RLlib listo para compute_single_action().
    """
    from ray.rllib.algorithms.algorithm import Algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    return algo.get_policy(policy_id)


def podar_snapshots(
    snapshot_dir: str,
    mantener: int = 50,
) -> List[str]:
    """Elimina snapshots más antiguos conservando solo los `mantener` más recientes.

    Returns:
        Lista de rutas eliminadas.
    """
    import shutil
    snapshots = listar_snapshots(snapshot_dir)
    eliminados: List[str] = []
    exceso = len(snapshots) - mantener
    for _, ruta in snapshots[:exceso]:
        shutil.rmtree(ruta, ignore_errors=True)
        eliminados.append(ruta)
    return eliminados
