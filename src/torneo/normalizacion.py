"""
Utilidades de normalización VecNormalize compartidas entre módulos.

Centraliza la lógica de carga y aplicación de stats de normalización,
eliminando duplicación entre evaluacion.py, entorno.py, y PoliticaSB3.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np


def cargar_vecnorm_stats(vecnorm_path: str) -> Optional[object]:
    """Carga las estadísticas obs_rms desde un archivo VecNormalize .pkl.

    Args:
        vecnorm_path: Ruta al archivo .pkl.

    Returns:
        Objeto obs_rms o None si no se pudo cargar.
    """
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return None
    try:
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        return vn.obs_rms
    except Exception:
        return None


def normalizar_obs(
    obs: np.ndarray,
    obs_rms: object,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> np.ndarray:
    """Normaliza una observación con estadísticas de VecNormalize.

    Args:
        obs: Vector de observación crudo.
        obs_rms: Objeto obs_rms con atributos mean y var.
        clip_min: Valor mínimo del clipping.
        clip_max: Valor máximo del clipping.

    Returns:
        Observación normalizada y clipeada.
    """
    if obs_rms is None or obs_rms.count < 1:
        return obs

    mean = np.array(obs_rms.mean)
    var = np.array(obs_rms.var)

    # Padding automático: soporte 190→194
    if len(mean) == 190 and len(obs) == 194:
        mean = np.concatenate([mean, np.zeros(4, dtype=np.float32)])
        var = np.concatenate([var, np.ones(4, dtype=np.float32)])

    return np.clip(
        (obs - mean) / np.sqrt(var + 1e-8), clip_min, clip_max
    ).astype(np.float32)


def normalizar_obs_desde_archivo(
    obs: np.ndarray, vecnorm_path: Optional[str]
) -> np.ndarray:
    """Normaliza observación cargando stats desde archivo.

    Conveniencia que combina cargar_vecnorm_stats + normalizar_obs.

    Args:
        obs: Vector de observación crudo.
        vecnorm_path: Ruta al archivo .pkl de VecNormalize.

    Returns:
        Observación normalizada (o cruda si no hay stats).
    """
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return obs
    try:
        obs_rms = cargar_vecnorm_stats(vecnorm_path)
        if obs_rms is None:
            return obs
        return normalizar_obs(obs, obs_rms)
    except Exception:
        return obs


def detectar_vecnorm(ruta_snapshot: str, base_dir: Optional[str] = None) -> Optional[str]:
    """Detecta el archivo VecNormalize asociado a un snapshot.

    Busca en orden:
        1. Per-snapshot: <ruta_snapshot>_vecnorm.pkl
        2. Versión global: modelos/<version>/vecnorm/vecnorm.pkl

    Args:
        ruta_snapshot: Ruta al snapshot (.zip opcional).
        base_dir: Directorio base del proyecto.

    Returns:
        Ruta al .pkl o None.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))

    ruta_limpia = ruta_snapshot.replace(".zip", "")
    candidatos = [ruta_limpia + "_vecnorm.pkl"]

    for c in candidatos:
        if os.path.exists(c):
            return c
    return None
