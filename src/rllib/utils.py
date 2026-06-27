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


def es_checkpoint_valido(checkpoint_path: str, policy_id: str = "default_policy") -> bool:
    """Devuelve True si el checkpoint tiene policy_state.pkl legible."""
    abs_path = os.path.abspath(checkpoint_path)
    policy_pkl = os.path.join(abs_path, "policies", policy_id, "policy_state.pkl")
    if not os.path.isfile(policy_pkl) or os.path.getsize(policy_pkl) == 0:
        return False
    try:
        import pickle
        with open(policy_pkl, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def cargar_policy_desde_checkpoint(checkpoint_path: str, policy_id: str = "default_policy"):
    """Carga pesos de política desde un checkpoint y devuelve un SnapshotPolicy.

    Lee directamente policy_state.pkl (sin recrear el algo ni Ray workers),
    así que es rápido y no requiere que Ray esté inicializado.

    Args:
        checkpoint_path: Ruta al directorio del checkpoint (absoluta o relativa).
        policy_id:       ID de la política a cargar.

    Returns:
        SnapshotPolicy listo para usar como callable (motor, idx, legales) -> Carta.

    Raises:
        FileNotFoundError: si el checkpoint no tiene policy_state.pkl válido.
    """
    import pickle
    from src.rllib.opponent_pool import SnapshotPolicy
    from src.entorno.dimensiones import DIM_ENTORNO

    abs_path = os.path.abspath(checkpoint_path)
    policy_pkl = os.path.join(abs_path, "policies", policy_id, "policy_state.pkl")

    if not os.path.isfile(policy_pkl):
        raise FileNotFoundError(
            f"Checkpoint sin policy_state.pkl (puede estar corrupto): {checkpoint_path}"
        )

    with open(policy_pkl, "rb") as f:
        state = pickle.load(f)

    weights: dict = state["weights"]

    # Detectar obs_dim a partir de la forma del primer layer del encoder
    try:
        obs_dim = weights["_encoder.0.weight"].shape[1]
    except Exception:
        obs_dim = DIM_ENTORNO

    return SnapshotPolicy.from_weights(weights, obs_dim=obs_dim)


def preservar_elite(
    snapshot_path: str,
    elite_dir: str,
    score: float,
    paso: int,
    max_elite: int = 5,
) -> bool:
    """Copia un snapshot al directorio `elite/` si su score entra en el top-K.

    El directorio elite NUNCA se poda por el ciclo normal, así que garantiza
    conservar los mejores modelos aunque `podar_snapshots` borre los snapshots
    antiguos. Mantiene un índice JSON (`elite_index.json`) con score y paso.

    Args:
        snapshot_path: Ruta del snapshot recién guardado a considerar.
        elite_dir:     Directorio elite destino.
        score:         Métrica de calidad (mayor = mejor).
        paso:          Paso global del snapshot.
        max_elite:     Cuántos elites conservar (top-K).

    Returns:
        True si el snapshot fue preservado, False si no calificó.
    """
    import json
    import shutil

    os.makedirs(elite_dir, exist_ok=True)
    index_path = os.path.join(elite_dir, "elite_index.json")
    index: List[dict] = []
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = []

    # ¿Califica? (hay hueco, o supera al peor del top-K)
    if len(index) >= max_elite and score <= min(e["score"] for e in index):
        return False

    nombre = f"elite_{paso:012d}"
    destino = os.path.join(elite_dir, nombre)
    if not os.path.exists(destino):
        try:
            shutil.copytree(snapshot_path, destino)
        except Exception:
            return False

    index.append({"score": round(float(score), 4), "paso": paso, "nombre": nombre})
    index.sort(key=lambda e: e["score"], reverse=True)

    # Podar elites que sobran (conservar solo top-K)
    for e in index[max_elite:]:
        shutil.rmtree(os.path.join(elite_dir, e["nombre"]), ignore_errors=True)
    index = index[:max_elite]

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return True


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
