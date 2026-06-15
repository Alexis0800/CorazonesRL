"""
Interfaz de línea de comandos — Evaluación standalone de modelo.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from src.torneo.evaluacion import evaluar_contra_bots
from src.torneo.normalizacion import detectar_vecnorm


def ejecutar_evaluacion(
    ruta_modelo: str,
    num_partidas: int = 100,
    vecnorm_path: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Ejecuta evaluación de un modelo contra bots heurísticos."""
    from stable_baselines3 import MaskablePPO

    if not os.path.exists(ruta_modelo):
        print(f"ERROR: Modelo no encontrado: {ruta_modelo}")
        print("¿Olvidaste el .zip? Prueba: {ruta_modelo}.zip")
        sys.exit(1)

    if vecnorm_path is None:
        vecnorm_path = detectar_vecnorm(ruta_modelo)

    print(f"Cargando modelo: {ruta_modelo}")
    if vecnorm_path:
        print(f"VecNormalize: {vecnorm_path}")

    model = MaskablePPO.load(ruta_modelo)
    metricas = evaluar_contra_bots(
        model, num_partidas, vecnorm_path, verbose=verbose
    )

    print("\n" + "=" * 60)
    print(f"RESULTADOS ({metricas['total_partidas']} partidas)")
    print("=" * 60)
    print(
        f"  1º lugar:  {metricas['pct_primero']:.1%}  ({metricas['victorias']} victorias)")
    print(f"  2º lugar:  {metricas['pct_segundo']:.1%}")
    print(f"  Top-2:     {metricas['pct_top2']:.1%}")
    print(f"  3º lugar:  {metricas['pct_tercero']:.1%}")
    print(f"  4º lugar:  {metricas['pct_cuarto']:.1%}")
    print(f"  Punt. promedio: {metricas['punt_promedio']:.1f}")
    print(f"  Punt. mediana:  {metricas['punt_mediana']:.1f}")
    print("=" * 60)
