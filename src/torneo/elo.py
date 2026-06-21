"""
Sistema de rating Elo para torneos entre snapshots históricos.

Mide la mejora real del agente enfrentándolo contra sus versiones anteriores
en un torneo round-robin. Cada par de snapshots juega N partidas; el ganador
de cada partida es quien termina con menor puntuación.

Uso:
    python -m src.torneo.elo --partidas 50

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

from src.entorno.dimensiones import DIM_ENTORNO

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_MODELOS_V5_DIR = os.path.join(_PROJECT_DIR, "models", "v5", "snapshots")
_MODELOS_V6_DIR = os.path.join(_PROJECT_DIR, "models", "v6", "snapshots")

K_FACTOR: int = 32
ELO_INICIAL: int = 1500

# Prefijo para identificar participantes bot en el torneo
BOT_PREFIX = "__BOT__"
_BOT_NAMES = ["conservador", "agresivo", "evasivo"]
_BOT_NAMES_EXPERTO = ["conservador", "agresivo", "evasivo", "experto"]
__all__ = [
    "K_FACTOR", "ELO_INICIAL", "BOT_PREFIX",
    "_expected_score", "_update_elo", "_calcular_elo_convergente",
    "_Modelo190Wrapper",
    "_extraer_paso_snapshot", "_listar_snapshots_torneo",
    "_encontrar_adyacentes", "_simular_resultado_torneo",
    "_get_bot_func", "_es_bot", "_nombre_bot", "_jugar_match_snapshots",
    "torneo_elo",
]
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


def _calcular_elo_convergente(
    snaps: List[str],
    resultados: Dict[Tuple[str, str], Tuple[int, int]],
    k: int = K_FACTOR,
    elo_inicial: int = ELO_INICIAL,
    max_iters: int = 200,
    epsilon: float = 0.01,
    seed: int = 42,
) -> Tuple[Dict[str, float], int]:
    """Calcula ratings Elo resolviendo el sistema de log-odds directamente.

    Para cada par (A,B) con wins_a/wins_b, la diferencia Elo teórica es:
        R_A - R_B ≈ 400 / ln(10) * ln(wins_a / wins_b)

    Se añade pseudo-count 0.5 para manejar scores extremos (10-0).
    El sistema sobredeterminado se resuelve por mínimos cuadrados,
    fijando la media de ratings en ELO_INICIAL.

    Args:
        snaps: Lista de nombres/etiquetas de los participantes.
        resultados: Dict {(a, b): (wins_a, wins_b)} con resultados H2H.
        k: Factor K (ignorado; se usa fórmula log-odds directa).
        elo_inicial: Rating medio del sistema (default 1500).
        max_iters: Ignorado (solución directa, no iterativa).
        epsilon: Ignorado (solución directa).
        seed: Ignorado.

    Returns:
        Tuple (ratings, 1) — siempre 1 iteración (solución directa).
    """
    if not resultados:
        return {s: float(elo_inicial) for s in snaps}, 0

    # Construir índice de snap → posición
    idx = {s: i for i, s in enumerate(snaps)}
    n = len(snaps)

    # Pseudo-count bayesiano para evitar log(0) en scores extremos
    PSEUDO = 0.5
    SCALE = 400.0 / np.log(10)  # ≈ 173.717

    # Sistema sobredeterminado: A * ratings = b
    # Cada par aporta una ecuación: R_a - R_b = SCALE * ln((wa+0.5)/(wb+0.5))
    rows = []
    b_vals = []
    weights = []

    for (a, b), (wins_a, wins_b) in resultados.items():
        if a not in idx or b not in idx:
            continue
        total = wins_a + wins_b
        if total == 0:
            continue

        wa = wins_a + PSEUDO
        wb = wins_b + PSEUDO
        diff = SCALE * np.log(wa / wb)
        weight = np.sqrt(total)  # Más partidas = más peso

        row = np.zeros(n)
        row[idx[a]] = 1.0
        row[idx[b]] = -1.0
        rows.append(row)
        b_vals.append(diff)
        weights.append(weight)

    if not rows:
        return {s: float(elo_inicial) for s in snaps}, 0

    A = np.array(rows)  # m × n
    b = np.array(b_vals)  # m
    W = np.diag(weights)  # m × m

    # Restricción: media de ratings = elo_inicial
    # Añadir como ecuación adicional con peso alto
    A_rest = np.ones((1, n))  # suma de ratings
    b_rest = np.array([n * elo_inicial])
    W_rest = np.diag([float(n)])  # peso proporcional al número de snaps

    A_aug = np.vstack([A, A_rest])
    b_aug = np.hstack([b, b_rest])
    W_aug = np.zeros((len(rows) + 1, len(rows) + 1))
    W_aug[:len(rows), :len(rows)] = W
    W_aug[len(rows), len(rows)] = float(n)

    # Resolver mínimos cuadrados ponderados: (A^T W A) x = A^T W b
    AtWA = A_aug.T @ W_aug @ A_aug
    AtWb = A_aug.T @ W_aug @ b_aug
    ratings_vec = np.linalg.solve(AtWA, AtWb)

    ratings = {s: float(ratings_vec[idx[s]]) for s in snaps}
    return ratings, 1


# ------------------------------------------------------------------
# Utilidades de snapshots
# ------------------------------------------------------------------

class _Modelo190Wrapper:
    """Envuelve un modelo 190-dim para aceptar observaciones 194-dim.

    Recorta las 4 dimensiones extra (all_void) antes de pasarlas
    al modelo subyacente. Esto permite que snapshots v5 compitan
    en entornos v6 sin modificar sus pesos.
    """

    def __init__(self, modelo_190: Any):
        self._model = modelo_190
        self.policy = modelo_190.policy
        self._total_timesteps = getattr(modelo_190, '_total_timesteps', 0)

        # Parchear observation_space para compatibilidad con predict()
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(DIM_ENTORNO,), dtype=np.float32
        )

    def predict(self, observation, **kwargs):
        """Recorta observacion de 194 a 190 y delega al modelo original."""
        import numpy as np
        obs = np.asarray(observation)
        if obs.ndim == 1 and obs.shape[0] == 194:
            obs = obs[:190]
        elif obs.ndim == 2 and obs.shape[1] == 194:
            obs = obs[:, :190]
        return self._model.predict(obs, **kwargs)

    def save(self, *args, **kwargs):
        return self._model.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self._model.load(*args, **kwargs)


def _extraer_paso_snapshot(ruta: str) -> int:
    """Extrae el número de paso de una ruta de snapshot.

    Args:
        ruta: Ruta como 'snapshot_0005000000.zip' o path completo.

    Returns:
        Número de paso (int). Retorna 0 para entradas de bot.
    """
    if ruta.startswith(BOT_PREFIX):
        return 0  # Bots no tienen paso
    nombre = os.path.basename(ruta).replace(".zip", "")
    return int(nombre.replace("snapshot_", ""))


def _listar_snapshots_torneo(
    directorios: Optional[List[str]] = None,
    min_paso: int = 0,
    max_snapshots: int = 20,
    incluir_bots: bool = False,
    nombres_bots: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Lista snapshots disponibles para torneo, ordenados por paso.

    Con múltiples directorios, aplica muestreo estratificado:
    cada directorio aporta ~max_snapshots/N snapshots, priorizando
    los más recientes de cada uno. Así v5 y v6 compiten en igualdad.

    Si incluir_bots=True, agrega bots heurísticos como participantes
    virtuales (sin paso, colocados al inicio).

    Args:
        directorios: Lista de directorios de snapshots (default: [v5]).
        min_paso: Paso mínimo para incluir.
        max_snapshots: Máximo total de snapshots (modelos + bots).
        incluir_bots: Si True, agrega los 3 bots heurísticos al torneo.

    Returns:
        Lista de tuplas (ruta_absoluta, label_directorio).
        Las entradas de bot usan el prefijo __BOT__.
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

    _lista_bots = nombres_bots if nombres_bots is not None else _BOT_NAMES
    if not snaps_por_dir:
        if incluir_bots:
            return [(f"{BOT_PREFIX}{name}", "bots") for name in _lista_bots]
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
            restantes = por_dir - len(primeros)
            if restantes > 0:
                ultimos = snaps[-restantes:]
                tomados = primeros + [s for s in ultimos if s not in primeros]
            else:
                tomados = primeros
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

    # Agregar bots como participantes virtuales (no sujetos a dedup por paso)
    if incluir_bots:
        for bot_name in _lista_bots:
            bot_id = f"{BOT_PREFIX}{bot_name}"
            unique.append((bot_id, "bots"))

    return unique


def _encontrar_adyacentes(
    ruta_snapshot: str, pool: List[str], n: int = 2
) -> List[str]:
    """Encuentra los n snapshots más cercanos en pasos de entrenamiento.

    Útil para Elo puro: en vez de bots, los otros asientos se llenan
    con snapshots de skill similar al rival B.

    Ignora entradas de bot (prefijo __BOT__).

    Args:
        ruta_snapshot: Snapshot de referencia.
        pool: Lista de todos los snapshots disponibles.
        n: Número de adyacentes a retornar.

    Returns:
        Lista de hasta n rutas de snapshots cercanos (excluye el mismo).
    """
    paso_ref = _extraer_paso_snapshot(ruta_snapshot)
    otros = [(abs(_extraer_paso_snapshot(s) - paso_ref), s)
             for s in pool if s != ruta_snapshot and not _es_bot(s)]
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

def _get_bot_func(bot_name: str):
    """Obtiene la función de bot por nombre.

    Args:
        bot_name: Nombre del bot ('conservador', 'agresivo', 'evasivo', 'experto').

    Returns:
        Callable del bot (motor, jugador_idx, legales) -> Carta.
        BotExperto devuelve una instancia fresca (tiene estado interno).
    """
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
    if bot_name == "experto":
        from src.agentes.bot_experto import BotExperto
        return BotExperto()
    mapa = {
        "conservador": bot_conservador,
        "agresivo": bot_agresivo,
        "evasivo": bot_evasivo,
    }
    return mapa[bot_name]


def _es_bot(ruta: str) -> bool:
    """Determina si una ruta corresponde a un bot virtual."""
    return ruta.startswith(BOT_PREFIX)


def _nombre_bot(ruta: str) -> str:
    """Extrae el nombre del bot de una ruta virtual."""
    return ruta.replace(BOT_PREFIX, "")


def _jugar_match_snapshots(
    ruta_a: str,
    ruta_b: str,
    num_partidas: int = 30,
    vecnorm_a: Optional[str] = None,
    vecnorm_b: Optional[str] = None,
    elo_puro: bool = False,
    adyacentes_b: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """Enfrenta dos participantes A vs B (modelos o bots).

    Modos:
        - Normal: A (seat 0) vs B (seat 1) + 2 bots (seats 2, 3).
        - Elo puro: A (seat 0) vs B (seat 1) + 2 snapshots adyacentes a B.
          Esto elimina el ruido de bots y hace la comparación más precisa.

    Args:
        ruta_a: Ruta al snapshot A (.zip) o ID de bot (__BOT__nombre).
        ruta_b: Ruta al snapshot B (.zip) o ID de bot (__BOT__nombre).
        num_partidas: Número de partidas.
        vecnorm_a: VecNormalize para A (auto-detecta si None). Ignorado si A es bot.
        vecnorm_b: VecNormalize para B (auto-detecta si None). Ignorado si B es bot.
        elo_puro: Si True, usar snapshots adyacentes en vez de bots.
        adyacentes_b: Lista de rutas de snapshots adyacentes a B (2 elementos).
                      Ignorado si B es bot.

    Returns:
        Tuple (victorias_a, victorias_b).
    """
    from sb3_contrib import MaskablePPO
    from src.entorno.single_agent import CorazonesEnv
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
    from src.torneo.evaluacion import (
        normalizar_obs_si_hay_stats, _detectar_vecnorm, _PoliticaSnapshot,
    )

    es_bot_a = _es_bot(ruta_a)
    es_bot_b = _es_bot(ruta_b)

    # --- Cargar participante A ---
    if es_bot_a:
        modelo_a = None
        vecnorm_a = None
        bot_a_func = _get_bot_func(_nombre_bot(ruta_a))
    else:
        ruta_a_clean = ruta_a[:-4] if ruta_a.endswith(".zip") else ruta_a
        modelo_a_raw = MaskablePPO.load(ruta_a_clean, device="cpu")
        modelo_a = (_Modelo190Wrapper(modelo_a_raw)
                    if modelo_a_raw.observation_space.shape[0] == 190
                    else modelo_a_raw)
        if vecnorm_a is None:
            vecnorm_a = _detectar_vecnorm(ruta_a)

    # --- Cargar participante B ---
    if es_bot_b:
        snap_b = _get_bot_func(_nombre_bot(ruta_b))
    else:
        ruta_b_clean = ruta_b[:-4] if ruta_b.endswith(".zip") else ruta_b
        modelo_b_raw = MaskablePPO.load(ruta_b_clean, device="cpu")
        modelo_b = (_Modelo190Wrapper(modelo_b_raw)
                    if modelo_b_raw.observation_space.shape[0] == 190
                    else modelo_b_raw)
        if vecnorm_b is None:
            vecnorm_b = _detectar_vecnorm(ruta_b)
        snap_b = _PoliticaSnapshot(modelo_b, vecnorm_b)

    # --- Preparar oponentes para seats 2 y 3 ---
    todos_bots = [bot_conservador, bot_agresivo, bot_evasivo]

    if elo_puro and not es_bot_b and adyacentes_b:
        # Modo puro con snapshots adyacentes (solo si B es modelo)
        seats_extra: List[Any] = []
        for adj_path in adyacentes_b[:2]:
            if _es_bot(adj_path):
                seats_extra.append(_get_bot_func(_nombre_bot(adj_path)))
                continue
            try:
                adj_model_raw = MaskablePPO.load(adj_path, device="cpu")
                adj_model = (_Modelo190Wrapper(adj_model_raw)
                             if adj_model_raw.observation_space.shape[0] == 190
                             else adj_model_raw)
                adj_vecnorm = _detectar_vecnorm(adj_path) or vecnorm_b
                seats_extra.append(_PoliticaSnapshot(adj_model, adj_vecnorm))
            except Exception:
                pass
        # Completar con bots si faltan, evitando duplicar bots participantes
        while len(seats_extra) < 2:
            cand = todos_bots[len(seats_extra) % 3]
            if es_bot_a and cand is bot_a_func:
                cand = todos_bots[(len(seats_extra) + 1) % 3]
            if es_bot_b and cand is snap_b:
                cand = todos_bots[(len(seats_extra) + 2) % 3]
            seats_extra.append(cand)
    else:
        # Modo normal: bots para seats 2 y 3
        # Evitar que el mismo bot esté en seat 1 y seat 2/3
        idx_offset = 0
        seats_extra = []
        for i in range(2):
            while True:
                cand = todos_bots[(i + idx_offset) % 3]
                if (es_bot_a and cand is bot_a_func) or (es_bot_b and cand is snap_b):
                    idx_offset += 1
                else:
                    break
            seats_extra.append(cand)
            idx_offset += 1

    # --- Detectar obs_dim de los modelos para crear el entorno correcto ---
    obs_dim: int = DIM_ENTORNO  # fallback
    if not es_bot_a:
        try:
            obs_dim = modelo_a.observation_space.shape[0]
        except Exception:
            pass
    elif not es_bot_b:
        try:
            obs_dim = modelo_b.observation_space.shape[0]
        except Exception:
            pass

    # --- Jugar partidas ---
    wins_a = 0
    wins_b = 0

    for seed in range(num_partidas):
        # Rotar seats extra para variabilidad
        if not elo_puro:
            r_idx = seed % 3
            seats_extra = [todos_bots[r_idx], todos_bots[(r_idx + 1) % 3]]
            # Evitar duplicados con participantes
            if es_bot_a and seats_extra[0] is bot_a_func:
                seats_extra[0] = todos_bots[(r_idx + 2) % 3]
            if es_bot_b and seats_extra[0] is snap_b:
                seats_extra[0] = todos_bots[(r_idx + 2) % 3]
            if es_bot_a and seats_extra[1] is bot_a_func:
                seats_extra[1] = todos_bots[(r_idx + 2) % 3]
            if es_bot_b and seats_extra[1] is snap_b:
                seats_extra[1] = todos_bots[(r_idx + 2) % 3]
            # Si ambos seats 2 y 3 serían el mismo, rotar
            if seats_extra[0] is seats_extra[1]:
                seats_extra[1] = todos_bots[(r_idx + 2) % 3]

        politicas = {
            1: snap_b,
            2: seats_extra[0],
            3: seats_extra[1],
        }

        env = CorazonesEnv(
            agente_idx=0, politicas_oponentes=politicas, obs_dim=obs_dim)
        obs_raw, _ = env.reset(seed=seed)
        obs = normalizar_obs_si_hay_stats(
            obs_raw, vecnorm_a) if not es_bot_a else obs_raw
        done = False

        while not done:
            if es_bot_a:
                # Bot A: usar la función de bot directamente
                legales = env.motor.obtener_jugadas_legales(0)
                carta = bot_a_func(env.motor, 0, legales)
                action = carta.id
            else:
                mask = env.action_masks()
                action, _ = modelo_a.predict(
                    obs, action_masks=mask, deterministic=True)
                action = int(action)

            obs_raw, _reward, terminated, truncated, _ = env.step(action)
            obs = normalizar_obs_si_hay_stats(
                obs_raw, vecnorm_a) if not es_bot_a else obs_raw
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
    incluir_bots: bool = False,
    nombres_bots: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Ejecuta un torneo round-robin entre snapshots históricos.

    Cada par de snapshots juega num_partidas partidas. Los resultados
    se usan para calcular ratings Elo.

    Args:
        directorios: Lista de directorios de snapshots (default: [v5]).
        num_partidas: Partidas por enfrentamiento.
        min_paso: Paso mínimo de entrenamiento para considerar un snapshot.
        max_snapshots: Máximo de snapshots en el torneo (modelos + bots).
        verbose: Imprimir progreso.
        elo_puro: Si True, usar snapshots adyacentes en vez de bots.
        incluir_bots: Si True, incluir bots heurísticos como participantes.
        nombres_bots: Lista de nombres de bots a incluir. Default: los 3 heurísticos.
                      Puede incluir 'experto' para añadir BotExperto.

    Returns:
        Diccionario con:
            - ratings: {ruta: rating_elo}
            - ranking: [(ruta, rating, paso, origen)] ordenado por rating
            - resultados: {(a, b): (wins_a, wins_b)}
    """
    snaps = _listar_snapshots_torneo(
        directorios, min_paso, max_snapshots,
        incluir_bots=incluir_bots, nombres_bots=nombres_bots)
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
        print(
            f"=== Torneo Elo === {len(snaps)} snapshots ({dirs_list}) | Modo: {modo}")
        print(f"   Partidas por enfrentamiento: {num_partidas}")
        print(
            f"   Total de partidas: {num_partidas * len(snaps) * (len(snaps) - 1) // 2}")
        print("=" * 60)

    # Inicializar ratings (key = ruta)
    ratings: Dict[str, float] = {
        r: float(ELO_INICIAL) for r in rutas
    }
    resultados: Dict[Tuple[str, str], Tuple[int, int]] = {}

    nombres_cortos = {}
    for r in rutas:
        if _es_bot(r):
            nombres_cortos[r] = f"[BOT] {_nombre_bot(r)}"
        else:
            nombres_cortos[r] = os.path.basename(r).replace(".zip", "")

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

            if verbose:
                print(f"{wins_a}-{wins_b}")

    # Recalcular ratings con convergencia (sin order bias)
    ratings, iters = _calcular_elo_convergente(rutas, resultados)
    if verbose:
        print(f"   Elo convergente: {iters} iteraciones")

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
    parser.add_argument("--incluir-v6", action="store_true", default=False,
                        help="Incluir snapshots de v6 en el torneo")
    parser.add_argument("--incluir-bots", action="store_true", default=False,
                        help="Incluir bots heurísticos (conservador/agresivo/evasivo) como participantes")
    parser.add_argument("--incluir-experto", action="store_true", default=False,
                        help="Incluir BotExperto como participante adicional")
    args = parser.parse_args()

    # Resolver directorios
    if args.directorios:
        directorios = args.directorios
    elif args.directorio:
        directorios = [args.directorio]
    else:
        directorios = [_MODELOS_V5_DIR]

    # Agregar v6 si se solicita
    if args.incluir_v6:
        if _MODELOS_V6_DIR not in directorios:
            directorios = list(directorios) + [_MODELOS_V6_DIR]

    # Determinar qué bots participan
    nombres_bots: Optional[List[str]] = None
    if args.incluir_bots and args.incluir_experto:
        nombres_bots = _BOT_NAMES_EXPERTO
    elif args.incluir_experto:
        nombres_bots = ["experto"]
    elif args.incluir_bots:
        nombres_bots = _BOT_NAMES

    resultado = torneo_elo(
        directorios=directorios,
        num_partidas=args.partidas,
        min_paso=args.min_paso,
        max_snapshots=args.max_snapshots,
        verbose=True,
        elo_puro=args.elo_puro,
        incluir_bots=(args.incluir_bots or args.incluir_experto),
        nombres_bots=nombres_bots,
    )

    if not resultado["ranking"]:
        print("No se encontraron snapshots para el torneo.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("=== CLASIFICACION FINAL ===")
    print("=" * 60)
    print(f"{'Pos':<5} {'Snapshot':<30} {'Paso':<12} {'Elo':<8} {'Diff'}")
    print("-" * 60)

    for pos, (snap, rating, paso, origen) in enumerate(resultado["ranking"]):
        nombre = resultado["nombres"].get(snap, os.path.basename(snap))
        delta = ""
        if pos > 0:
            _, prev_rating, _, _ = resultado["ranking"][pos - 1]
            diff = rating - prev_rating
            delta = f"{diff:+.0f}"
        paso_str = f"{paso:>10,}" if paso > 0 else "     (bot)"
        print(f"{pos+1:<5} {nombre:<30} {paso_str}  {rating:<8.0f} {delta}")
