"""
Script de evaluación para el modelo entrenado de Corazones.
Mide el win rate y métricas multi-nivel contra 3 bots heurísticos en N partidas.

Métricas reportadas:
    - Porcentaje en 1º, 2º, 3º, 4º lugar
    - Top-2 (1º + 2º)
    - Puntuación promedio, mediana, mínima y máxima

Normalización:
    Si el modelo fue entrenado con VecNormalize (norm_obs=True),
    se aplica normalización manual usando las estadísticas del archivo .pkl.
    Corrección: VecNormalize se carga como objeto (no dict), se usa vn.obs_rms.

Uso:
    python evaluar_modelo.py [--ruta RUTA] [--partidas N]
"""

import argparse
import os
import sys
import pickle
from typing import Dict, List, Optional

import numpy as np

from src.entorno.single_agent import CorazonesEnv
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo


def normalizar_obs_si_hay_stats(obs: np.ndarray, vecnorm_path: str) -> np.ndarray:
    """Normaliza observación con stats de VecNormalize de SB3.

    Args:
        obs: Vector de observación crudo de shape (190,).
        vecnorm_path: Ruta al archivo .pkl de VecNormalize.

    Returns:
        Observación normalizada (o cruda si no hay stats disponibles).

    Nota:
        El archivo .pkl contiene un objeto VecNormalize (NO un dict).
        Se accede a las estadísticas vía ``vn.obs_rms`` directamente.
    """
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return obs
    try:
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        # Corrección: VecNormalize es un objeto, no un dict.
        # vn.obs_rms contiene RunningMeanStd con .mean y .var.
        obs_rms = vn.obs_rms
        if obs_rms is None or obs_rms.count < 1:
            return obs
        mean = np.array(obs_rms.mean)
        var = np.array(obs_rms.var)
        return np.clip(
            (obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0
        ).astype(np.float32)
    except Exception:
        return obs


def _calcular_posicion(punt_agente: int, punt_rivales: list) -> int:
    """Determina la posición del agente (0 = 1º lugar, 3 = 4º lugar).

    En Corazones, gana quien tiene MENOS puntos acumulados.
    En caso de empate, se asigna la mejor posición compartida.

    Args:
        punt_agente: Puntuación acumulada del agente.
        punt_rivales: Lista con las puntuaciones de los 3 rivales.

    Returns:
        Índice de posición: 0 (1º), 1 (2º), 2 (3º), 3 (4º).
    """
    todas = [punt_agente] + list(punt_rivales)
    # Ordenar de menor a mayor (menos puntos = mejor)
    ranking = sorted(range(4), key=lambda i: todas[i])
    posicion = ranking.index(0)  # 0 es el índice del agente
    return posicion


def _construir_metricas(
    posiciones: list, puntuaciones: list, total: int
) -> dict:
    """Construye el diccionario de métricas a partir de resultados individuales.

    Args:
        posiciones: Lista de posiciones del agente (0=1º, 1=2º, 2=3º, 3=4º).
        puntuaciones: Lista de puntuaciones del agente en cada partida.
        total: Número total de partidas.

    Returns:
        Diccionario con métricas agregadas.
    """
    if total == 0:
        return {
            "total_partidas": 0,
            "victorias": 0,
            "pct_primero": 0.0,
            "pct_segundo": 0.0,
            "pct_tercero": 0.0,
            "pct_cuarto": 0.0,
            "pct_top2": 0.0,
            "punt_promedio": 0.0,
            "punt_mediana": 0.0,
            "punt_min": 0.0,
            "punt_max": 0.0,
        }

    victorias = sum(1 for p in posiciones if p == 0)
    arr = np.array(puntuaciones, dtype=np.float64)

    return {
        "total_partidas": total,
        "victorias": victorias,
        "pct_primero": sum(1 for p in posiciones if p == 0) / total,
        "pct_segundo": sum(1 for p in posiciones if p == 1) / total,
        "pct_tercero": sum(1 for p in posiciones if p == 2) / total,
        "pct_cuarto": sum(1 for p in posiciones if p == 3) / total,
        "pct_top2": sum(1 for p in posiciones if p in (0, 1)) / total,
        "punt_promedio": float(np.mean(arr)),
        "punt_mediana": float(np.median(arr)),
        "punt_min": float(np.min(arr)),
        "punt_max": float(np.max(arr)),
    }


def evaluar_con_metricas(
    ruta_modelo: str,
    num_partidas: int,
    vecnorm_path: Optional[str] = None,
    oponentes: Optional[List] = None,
) -> Dict[str, float]:
    """Evalúa el modelo contra oponentes especificados con métricas multi-nivel.

    Args:
        ruta_modelo: Ruta al modelo .zip (con o sin extensión).
        num_partidas: Número de partidas de evaluación.
        vecnorm_path: Ruta al .pkl de VecNormalize (opcional).
        oponentes: Lista de 3 funciones de bot (default: conservador, agresivo, evasivo).

    Returns:
        Diccionario con métricas: pct_primero, pct_segundo, pct_tercero,
        pct_cuarto, pct_top2, punt_promedio, punt_mediana, punt_min,
        punt_max, total_partidas, victorias.
    """
    if not ruta_modelo.endswith(".zip"):
        ruta_modelo += ".zip"

    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No se encuentra {ruta_modelo}")
        sys.exit(1)

    from sb3_contrib import MaskablePPO

    try:
        modelo = MaskablePPO.load(ruta_modelo, device="cpu")
    except Exception as e:
        print(f"ERROR al cargar modelo: {e}")
        print("Intentando con device=None...")
        modelo = MaskablePPO.load(ruta_modelo, device=None, env=None)

    if oponentes is None:
        oponentes = [bot_conservador, bot_agresivo, bot_evasivo]

    posiciones: List[int] = []
    puntuaciones: List[float] = []

    for seed in range(num_partidas):
        politicas_oponentes = {
            1: oponentes[0], 2: oponentes[1], 3: oponentes[2]}
        # Rotar oponentes para variabilidad entre partidas
        if num_partidas > 1:
            politicas_oponentes = {
                1: oponentes[seed % len(oponentes)],
                2: oponentes[(seed + 1) % len(oponentes)],
                3: oponentes[(seed + 2) % len(oponentes)],
            }
        env = CorazonesEnv(
            agente_idx=0,
            politicas_oponentes=politicas_oponentes,
        )
        obs_raw, _ = env.reset(seed=seed)
        obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
        done = False

        while not done:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True)
            obs_raw, _reward, terminated, truncated, _ = env.step(int(action))
            obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
            done = terminated or truncated

        punt_agente = env._puntuacion_historica[0]
        punt_rivales = [env._puntuacion_historica[i] for i in range(1, 4)]

        posicion = _calcular_posicion(punt_agente, punt_rivales)
        posiciones.append(posicion)
        puntuaciones.append(float(punt_agente))

        env.close()

        if (seed + 1) % 25 == 0:
            victorias_parcial = sum(1 for p in posiciones if p == 0)
            top2_parcial = sum(1 for p in posiciones if p in (0, 1))
            print(
                f"  {seed + 1}/{num_partidas} partidas... "
                f"({victorias_parcial} 1º, {top2_parcial} top-2)"
            )

    return _construir_metricas(posiciones, puntuaciones, num_partidas)


def evaluar(
    ruta_modelo: str,
    num_partidas: int,
    vecnorm_path: Optional[str] = None,
    oponentes: Optional[List] = None,
) -> float:
    """Evalúa el modelo contra bots heurísticos (compatibilidad hacia atrás).

    Args:
        ruta_modelo: Ruta al modelo .zip (con o sin extensión).
        num_partidas: Número de partidas.
        vecnorm_path: Ruta al .pkl de VecNormalize (opcional).
        oponentes: Lista de funciones de bot (opcional).

    Returns:
        Win rate (0.0 a 1.0).
    """
    metricas = evaluar_con_metricas(ruta_modelo, num_partidas, vecnorm_path, oponentes)
    return metricas["pct_primero"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluar modelo de Corazones RL")
    parser.add_argument(
        "--ruta", type=str, default="modelos_historicos/modelo_final",
        help="Ruta al modelo (con o sin .zip)"
    )
    parser.add_argument(
        "--partidas", type=int, default=100,
        help="Número de partidas"
    )
    parser.add_argument(
        "--oponentes", type=str, nargs="+", default=None,
        help="Bots oponentes: conservador, agresivo, evasivo (default: los 3)"
    )
    args = parser.parse_args()

    # Resolver oponentes
    MAPA_BOTS = {
        "conservador": bot_conservador,
        "agresivo": bot_agresivo,
        "evasivo": bot_evasivo,
    }
    if args.oponentes:
        oponentes = []
        for nombre in args.oponentes:
            n = nombre.lower()
            if n in MAPA_BOTS:
                oponentes.append(MAPA_BOTS[n])
            else:
                print(f"ADVERTENCIA: Bot desconocido '{nombre}'. Opciones: conservador, agresivo, evasivo")
        if len(oponentes) < 3:
            # Rellenar con los bots por defecto
            defaults = [bot_conservador, bot_agresivo, bot_evasivo]
            for d in defaults:
                if d not in oponentes:
                    oponentes.append(d)
                if len(oponentes) >= 3:
                    break
        oponentes = oponentes[:3]  # Solo 3 oponentes
    else:
        oponentes = None  # Usar defaults en la función

    # Detectar VecNormalize asociado
    # Orden de búsqueda:
    # 1. Per-snapshot: modelos_historicos/v5/snapshot_*_vecnorm.pkl
    # 2. VecNormalize global v5: vecnormalize/v5/v5_vecnorm.pkl
    # 3. VecNormalize global v5 final: vecnormalize/v5/v5_vecnorm_final.pkl
    # 4. VecNormalize v2 (compatibilidad): vecnormalize/v2_vecnorm_final.pkl
    # 5. VecNormalize legacy: vecnormalize/vecnorm.pkl
    ruta_base = args.ruta.replace(".zip", "")
    vecnorm_path: Optional[str] = None
    for candidato in [
        ruta_base + "_vecnorm.pkl",
        "vecnormalize/v5/v5_vecnorm.pkl",
        "vecnormalize/v5/v5_vecnorm_final.pkl",
        "vecnormalize/v2_vecnorm_final.pkl",
        "vecnormalize/v2_vecnorm.pkl",
        "vecnormalize/vecnorm.pkl",
    ]:
        if os.path.exists(candidato):
            vecnorm_path = candidato
            break

    if oponentes:
        nombres_ops = [b.__name__.replace("bot_", "") for b in oponentes]
    else:
        nombres_ops = ["conservador", "agresivo", "evasivo"]

    print("=" * 55)
    print(f"Evaluando: {ruta_base}.zip")
    print(f"Partidas: {args.partidas}")
    print(f"Oponentes: {', '.join(nombres_ops)}")
    print(f"VecNormalize: {vecnorm_path or 'NO (usando obs crudas)'}")
    print("-" * 55)

    metricas = evaluar_con_metricas(ruta_base, args.partidas, vecnorm_path, oponentes)

    print("-" * 55)
    print(f"Resultados ({metricas['total_partidas']} partidas):")
    print(
        f"  🥇 1º lugar: {metricas['pct_primero']:.1%} ({metricas['victorias']} victorias)")
    print(f"  🥈 2º lugar: {metricas['pct_segundo']:.1%}")
    print(f"  📊 Top-2:    {metricas['pct_top2']:.1%}")
    print(f"  🥉 3º lugar: {metricas['pct_tercero']:.1%}")
    print(f"  4º lugar:    {metricas['pct_cuarto']:.1%}")
    print(f"  ─────────────────────────")
    print(f"  Punt. promedio: {metricas['punt_promedio']:.1f}")
    print(f"  Punt. mediana:  {metricas['punt_mediana']:.1f}")
    print(f"  Punt. mínima:   {metricas['punt_min']:.1f}")
    print(f"  Punt. máxima:   {metricas['punt_max']:.1f}")
    print("=" * 55)
