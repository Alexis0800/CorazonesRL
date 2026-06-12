"""
Sistema de rating Elo para torneos entre snapshots históricos.

Mide la mejora real del agente enfrentándolo contra sus versiones anteriores
en un torneo round-robin. Cada par de snapshots juega N partidas; el ganador
de cada partida es quien termina con menor puntuación.

Uso:
    python -m src.elo_torneo --partidas 50

Output:
    Ranking de snapshots ordenados por Elo, mostrando la progresión del
    aprendizaje en el tiempo.

Fórmula Elo:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    R_A' = R_A + K * (S_A - E_A)
"""

from __future__ import annotations

import os
import sys
import glob
import pickle
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_MODELOS_V5_DIR = os.path.join(_PROJECT_DIR, "modelos_historicos", "v5")

K_FACTOR: int = 32
ELO_INICIAL: int = 1500


# ------------------------------------------------------------------
# Fórmulas Elo
# ------------------------------------------------------------------

def _expected_score(ra: float, rb: float) -> float:
    """Probabilidad esperada de que A le gane a B según ratings Elo.

    Args:
        ra: Rating de A.
        rb: Rating de B.

    Returns:
        Probabilidad esperada (0.0 a 1.0) de que A gane.
    """
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _update_elo(
    ra: float, rb: float, score_a: float, k: int = K_FACTOR
) -> Tuple[float, float]:
    """Actualiza ratings Elo después de un enfrentamiento.

    Args:
        ra: Rating actual de A.
        rb: Rating actual de B.
        score_a: Resultado real de A (1.0 = victoria, 0.0 = derrota, 0.5 = empate).
        k: Factor K (default 32).

    Returns:
        Tuple (nuevo_ra, nuevo_rb) con ratings actualizados.
    """
    ea = _expected_score(ra, rb)
    delta = k * (score_a - ea)
    return ra + delta, rb - delta


# ------------------------------------------------------------------
# Utilidades de snapshots
# ------------------------------------------------------------------

def _extraer_paso_snapshot(ruta: str) -> int:
    """Extrae el número de paso de una ruta de snapshot.

    Args:
        ruta: Ruta como 'snapshot_0005000000.zip' o path completo.

    Returns:
        Número de paso (int).
    """
    nombre = os.path.basename(ruta).replace(".zip", "")
    return int(nombre.replace("snapshot_", ""))


def _listar_snapshots_torneo(
    directorios: Optional[List[str]] = None,
    min_paso: int = 0,
    max_snapshots: int = 20,
) -> List[Tuple[str, str]]:
    """Lista snapshots disponibles para torneo, ordenados por paso.

    Con múltiples directorios, aplica muestreo estratificado:
    cada directorio aporta ~max_snapshots/N snapshots, priorizando
    los más recientes de cada uno. Así v5 y v6 compiten en igualdad.

    Args:
        directorios: Lista de directorios de snapshots (default: [v5]).
        min_paso: Paso mínimo para incluir.
        max_snapshots: Máximo total de snapshots.

    Returns:
        Lista de tuplas (ruta_absoluta, label_directorio).
    """
    if directorios is None:
        directorios = [_MODELOS_V5_DIR]

    # Recolectar snaps por directorio
    snaps_por_dir: Dict[str, List[str]] = {}
    for d in directorios:
        if not os.path.isdir(d):
            continue
        label = os.path.basename(d.rstrip("/\\"))
        snaps = glob.glob(os.path.join(d, "snapshot_*.zip"))
        snaps = [s for s in snaps if _extraer_paso_snapshot(s) >= min_paso]
        snaps.sort(key=_extraer_paso_snapshot)
        if snaps:
            snaps_por_dir[label] = snaps

    if not snaps_por_dir:
        return []

    num_dirs = len(snaps_por_dir)
    por_dir = max(2, max_snapshots // num_dirs)

    resultado: List[Tuple[str, str]] = []
    for label, snaps in snaps_por_dir.items():
        if len(snaps) <= por_dir:
            tomados = snaps
        else:
            # Tomar 1-2 tempranos + el resto de los más recientes
            primeros = snaps[:min(2, len(snaps))]
            ultimos = snaps[-(por_dir - len(primeros)):]
            tomados = primeros + [s for s in ultimos if s not in primeros]
        resultado.extend((s, label) for s in tomados)

    # Deducir duplicados (mismo paso en distinto directorio)
    seen = set()
    unique: List[Tuple[str, str]] = []
    for s, label in resultado:
        paso = _extraer_paso_snapshot(s)
        if paso not in seen:
            seen.add(paso)
            unique.append((s, label))

    unique.sort(key=lambda x: _extraer_paso_snapshot(x[0]))
    return unique


def _encontrar_adyacentes(
    ruta_snapshot: str, pool: List[str], n: int = 2
) -> List[str]:
    """Encuentra los n snapshots más cercanos en pasos de entrenamiento.

    Útil para Elo puro: en vez de bots, los otros asientos se llenan
    con snapshots de skill similar al rival B.

    Args:
        ruta_snapshot: Snapshot de referencia.
        pool: Lista de todos los snapshots disponibles.
        n: Número de adyacentes a retornar.

    Returns:
        Lista de hasta n rutas de snapshots cercanos (excluye el mismo).
    """
    paso_ref = _extraer_paso_snapshot(ruta_snapshot)
    otros = [(abs(_extraer_paso_snapshot(s) - paso_ref), s)
             for s in pool if s != ruta_snapshot]
    otros.sort(key=lambda x: x[0])
    return [s for _, s in otros[:n]]


def _simular_resultado_torneo(
    snaps: List[str],
    ratings_iniciales: Dict[str, float],
    resultados: Dict[Tuple[str, str], Tuple[int, int]],
) -> Dict[str, float]:
    """Calcula ratings finales dados los resultados de matches.

    Args:
        snaps: Lista de identificadores de snapshots.
        ratings_iniciales: Diccionario {snap_id: rating_inicial}.
        resultados: Diccionario {(snap_a, snap_b): (victorias_a, victorias_b)}.

    Returns:
        Diccionario {snap_id: rating_final}.
    """
    ratings = dict(ratings_iniciales)

    for (a, b), (wins_a, wins_b) in resultados.items():
        total = wins_a + wins_b
        if total == 0:
            continue

        # Procesar cada partida individualmente para actualización incremental
        for _ in range(wins_a):
            ratings[a], ratings[b] = _update_elo(
                ratings[a], ratings[b], score_a=1.0, k=K_FACTOR)
        for _ in range(wins_b):
            ratings[a], ratings[b] = _update_elo(
                ratings[a], ratings[b], score_a=0.0, k=K_FACTOR)

    return ratings


# ------------------------------------------------------------------
# Match entre dos snapshots
# ------------------------------------------------------------------

def _jugar_match_snapshots(
    ruta_a: str,
    ruta_b: str,
    num_partidas: int = 30,
    vecnorm_a: Optional[str] = None,
    vecnorm_b: Optional[str] = None,
    elo_puro: bool = False,
    adyacentes_b: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """Enfrenta dos snapshots A vs B.

    Modos:
        - Normal: A (seat 0) vs B (seat 1) + 2 bots (seats 2, 3).
        - Elo puro: A (seat 0) vs B (seat 1) + 2 snapshots adyacentes a B.
          Esto elimina el ruido de bots y hace la comparación más precisa.

    Args:
        ruta_a: Ruta al snapshot A (.zip).
        ruta_b: Ruta al snapshot B (.zip).
        num_partidas: Número de partidas.
        vecnorm_a: VecNormalize para A (auto-detecta si None).
        vecnorm_b: VecNormalize para B (auto-detecta si None).
        elo_puro: Si True, usar snapshots adyacentes en vez de bots.
        adyacentes_b: Lista de rutas de snapshots adyacentes a B (2 elementos).

    Returns:
        Tuple (victorias_a, victorias_b).
    """
    from sb3_contrib import MaskablePPO
    from src.entorno import CorazonesEnv
    from src.bots import bot_conservador, bot_agresivo, bot_evasivo
    from src.evaluacion import (
        normalizar_obs_si_hay_stats, _detectar_vecnorm, _PoliticaSnapshot,
    )

    # Cargar modelos (SB3 a veces agrega .zip, normalizamos primero)
    ruta_a_clean = ruta_a[:-4] if ruta_a.endswith(".zip") else ruta_a
    ruta_b_clean = ruta_b[:-4] if ruta_b.endswith(".zip") else ruta_b
    modelo_a = MaskablePPO.load(ruta_a_clean, device="cpu")
    modelo_b_raw = MaskablePPO.load(ruta_b_clean, device="cpu")

    # Detectar VecNormalize
    if vecnorm_a is None:
        vecnorm_a = _detectar_vecnorm(ruta_a)
    if vecnorm_b is None:
        vecnorm_b = _detectar_vecnorm(ruta_b)

    # Envolver B en adaptador
    snap_b = _PoliticaSnapshot(modelo_b_raw, vecnorm_b)

    # Preparar oponentes para seats 2 y 3
    if elo_puro and adyacentes_b and len(adyacentes_b) >= 1:
        # Modo puro: snapshots adyacentes como "equipo B"
        seats_extra: List[Any] = []
        for adj_path in adyacentes_b[:2]:
            try:
                adj_model = MaskablePPO.load(adj_path, device="cpu")
                adj_vecnorm = _detectar_vecnorm(adj_path) or vecnorm_b
                seats_extra.append(_PoliticaSnapshot(adj_model, adj_vecnorm))
            except Exception:
                pass
        # Completar con bots si faltan adyacentes
        bots = [bot_conservador, bot_agresivo, bot_evasivo]
        while len(seats_extra) < 2:
            seats_extra.append(bots[len(seats_extra) % 3])
    else:
        # Modo normal: bots
        bots = [bot_conservador, bot_agresivo, bot_evasivo]
        seats_extra = [bots[0], bots[1]]

    wins_a = 0
    wins_b = 0

    for seed in range(num_partidas):
        # Rotar seats extra para variabilidad
        if not elo_puro:
            seats_extra = [bots[seed % 3], bots[(seed + 1) % 3]]

        politicas = {
            1: snap_b,
            2: seats_extra[0],
            3: seats_extra[1],
        }

        env = CorazonesEnv(agente_idx=0, politicas_oponentes=politicas)
        obs_raw, _ = env.reset(seed=seed)
        obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_a)
        done = False

        while not done:
            mask = env.action_masks()
            action, _ = modelo_a.predict(
                obs, action_masks=mask, deterministic=True)
            obs_raw, _reward, terminated, truncated, _ = env.step(int(action))
            obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_a)
            done = terminated or truncated

        score_a = env._puntuacion_historica[0]
        score_b = env._puntuacion_historica[1]

        if score_a < score_b:
            wins_a += 1
        elif score_b < score_a:
            wins_b += 1

        env.close()

    return wins_a, wins_b


# ------------------------------------------------------------------
# Torneo round-robin
# ------------------------------------------------------------------

def torneo_elo(
    directorios: Optional[List[str]] = None,
    num_partidas: int = 30,
    min_paso: int = 0,
    max_snapshots: int = 15,
    verbose: bool = True,
    elo_puro: bool = False,
) -> Dict[str, Any]:
    """Ejecuta un torneo round-robin entre snapshots históricos.

    Cada par de snapshots juega num_partidas partidas. Los resultados
    se usan para calcular ratings Elo.

    Args:
        directorio: Directorio de snapshots (default: v5).
        num_partidas: Partidas por enfrentamiento.
        min_paso: Paso mínimo de entrenamiento para considerar un snapshot.
        max_snapshots: Máximo de snapshots en el torneo.
        verbose: Imprimir progreso.

    Returns:
        Diccionario con:
            - ratings: {ruta: rating_elo}
            - ranking: [(ruta, rating, paso)] ordenado por rating
            - resultados: {(a, b): (wins_a, wins_b)}
    """
    snaps = _listar_snapshots_torneo(directorios, min_paso, max_snapshots)
    # snaps: List[(ruta, label)]

    if len(snaps) < 2:
        print("Se necesitan al menos 2 snapshots para un torneo.")
        return {"ratings": {}, "ranking": [], "resultados": {}}

    # Extraer rutas para cargar modelos
    rutas = [s[0] for s in snaps]
    origenes = {s[0]: s[1] for s in snaps}

    if verbose:
        modo = "Puro (adyacentes)" if elo_puro else "Normal (bots)"
        dirs_list = ", ".join(sorted(set(s[1] for s in snaps)))
        print("=" * 60)
        print(f"🏆 Torneo Elo — {len(snaps)} snapshots ({dirs_list}) | Modo: {modo}")
        print(f"   Partidas por enfrentamiento: {num_partidas}")
        print(f"   Total de partidas: {num_partidas * len(snaps) * (len(snaps) - 1) // 2}")
        print("=" * 60)

    # Inicializar ratings (key = ruta)
    ratings: Dict[str, float] = {
        r: float(ELO_INICIAL) for r in rutas
    }
    resultados: Dict[Tuple[str, str], Tuple[int, int]] = {}

    nombres_cortos = {r: os.path.basename(r).replace(".zip", "")
                      for r in rutas}

    # Round-robin
    total_pares = len(rutas) * (len(rutas) - 1) // 2
    par_idx = 0

    for i in range(len(rutas)):
        for j in range(i + 1, len(rutas)):
            par_idx += 1
            a, b = rutas[i], rutas[j]
            na, nb = nombres_cortos[a], nombres_cortos[b]

            if verbose:
                print(f"  [{par_idx}/{total_pares}] {na} vs {nb} ... ",
                      end="", flush=True)

            # Encontrar adyacentes si estamos en modo puro (usar rutas como pool)
            adyacentes = None
            if elo_puro:
                adyacentes = _encontrar_adyacentes(b, rutas)

            wins_a, wins_b = _jugar_match_snapshots(
                a, b, num_partidas,
                elo_puro=elo_puro,
                adyacentes_b=adyacentes,
            )
            resultados[(a, b)] = (wins_a, wins_b)

            # Actualizar ratings incrementalmente
            for _ in range(wins_a):
                ratings[a], ratings[b] = _update_elo(
                    ratings[a], ratings[b], score_a=1.0)
            for _ in range(wins_b):
                ratings[a], ratings[b] = _update_elo(
                    ratings[a], ratings[b], score_a=0.0)

            if verbose:
                print(f"{wins_a}-{wins_b}")

    # Construir ranking
    ranking = sorted(
        [(r, ratings[r], _extraer_paso_snapshot(r), origenes.get(r, "?"))
         for r in rutas],
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "ratings": ratings,
        "ranking": ranking,
        "resultados": resultados,
        "nombres": nombres_cortos,
        "origenes": origenes,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Torneo Elo entre snapshots históricos de Corazones RL")
    parser.add_argument("--directorio", type=str, default=None,
                        help="Directorio único de snapshots (atajo, equivale a --directorios)")
    parser.add_argument("--directorios", type=str, nargs="+", default=None,
                        help="Directorios de snapshots (múltiples)")
    parser.add_argument("--partidas", type=int, default=30,
                        help="Partidas por enfrentamiento (default=30)")
    parser.add_argument("--min-paso", type=int, default=0,
                        help="Paso mínimo (default=0 = todos)")
    parser.add_argument("--max-snapshots", type=int, default=15,
                        help="Máximo de snapshots en el torneo (default=15)")
    parser.add_argument("--elo-puro", action="store_true", default=False,
                        help="Usar snapshots adyacentes en vez de bots (más preciso)")
    args = parser.parse_args()

    # Resolver directorios
    if args.directorios:
        directorios = args.directorios
    elif args.directorio:
        directorios = [args.directorio]
    else:
        directorios = [_MODELOS_V5_DIR]

    resultado = torneo_elo(
        directorios=directorios,
        num_partidas=args.partidas,
        min_paso=args.min_paso,
        max_snapshots=args.max_snapshots,
        verbose=True,
        elo_puro=args.elo_puro,
    )

    if not resultado["ranking"]:
        print("No se encontraron snapshots para el torneo.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🏆 CLASIFICACIÓN FINAL")
    print("=" * 60)
    print(f"{'Pos':<5} {'Snapshot':<30} {'Paso':<12} {'Elo':<8} {'Δ desde anterior'}")
    print("-" * 60)

    for pos, (snap, rating, paso, origen) in enumerate(resultado["ranking"]):
        nombre = resultado["nombres"].get(snap, os.path.basename(snap))
        delta = ""
        if pos > 0:
            _, prev_rating, _, _ = resultado["ranking"][pos - 1]
            diff = rating - prev_rating
            delta = f"{diff:+.0f}"
        print(f"{pos+1:<5} {nombre:<30} {paso:>10,}  {rating:<8.0f} {delta}")
