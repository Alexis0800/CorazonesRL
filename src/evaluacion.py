"""
Módulo de evaluación multi-nivel para el agente RL de Corazones.

Provee tres niveles de evaluación:
    1. Contra bots heurísticos (rápido, ~30s para 100 partidas).
    2. Contra snapshots históricos — Self-Play eval (medio, ~2min).
    3. Escenarios estratégicos (quirúrgico, ~5s).

Y funciones de soporte para integración con el bucle de entrenamiento
(auto-evaluación periódica con early stopping opcional).

Uso desde train_self_play.py:
    from src.evaluacion import evaluar_snapshot_callback, guardar_log_evaluacion

Uso standalone:
    python -m src.evaluacion --ruta modelos_historicos/v5/snapshot_0002000000
"""

from __future__ import annotations

import json
import os
import sys
import pickle
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Constantes de rutas (relativas al proyecto)
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_VECNORM_V5_DIR = os.path.join(_PROJECT_DIR, "vecnormalize", "v5")
_MODELOS_V5_DIR = os.path.join(_PROJECT_DIR, "modelos_historicos", "v5")


# ------------------------------------------------------------------
# Funciones de métricas (compartidas con evaluar_modelo.py)
# ------------------------------------------------------------------

def _calcular_posicion(punt_agente: int, punt_rivales: list) -> int:
    """Determina la posición del agente (0 = 1º lugar, 3 = 4º lugar)."""
    todas = [punt_agente] + list(punt_rivales)
    ranking = sorted(range(4), key=lambda i: todas[i])
    return ranking.index(0)


def _construir_metricas(
    posiciones: list, puntuaciones: list, total: int
) -> Dict[str, float]:
    """Construye diccionario de métricas a partir de resultados individuales."""
    if total == 0:
        return {
            "total_partidas": 0, "victorias": 0,
            "pct_primero": 0.0, "pct_segundo": 0.0,
            "pct_tercero": 0.0, "pct_cuarto": 0.0,
            "pct_top2": 0.0,
            "punt_promedio": 0.0, "punt_mediana": 0.0,
            "punt_min": 0.0, "punt_max": 0.0,
        }
    arr = np.array(puntuaciones, dtype=np.float64)
    return {
        "total_partidas": total,
        "victorias": sum(1 for p in posiciones if p == 0),
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


# ------------------------------------------------------------------
# Normalización
# ------------------------------------------------------------------

def normalizar_obs_si_hay_stats(
    obs: np.ndarray, vecnorm_path: Optional[str]
) -> np.ndarray:
    """Normaliza observación con stats de VecNormalize de SB3.

    Soporta padding automático de 190→194 dimensiones para
    compatibilidad con snapshots v5 cargados en entornos v6.

    Args:
        obs: Vector de observación crudo de shape (190,) o (194,).
        vecnorm_path: Ruta al archivo .pkl de VecNormalize.

    Returns:
        Observación normalizada (o cruda si no hay stats).
    """
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return obs
    try:
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        obs_rms = vn.obs_rms
        if obs_rms is None or obs_rms.count < 1:
            return obs
        mean = np.array(obs_rms.mean)
        var = np.array(obs_rms.var)
        # Padding automático: stats 190-dim → 194-dim
        if len(mean) == 190 and len(obs) == 194:
            mean = np.concatenate([mean, np.zeros(4, dtype=np.float32)])
            var = np.concatenate([var, np.ones(4, dtype=np.float32)])
        return np.clip(
            (obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0
        ).astype(np.float32)
    except Exception:
        return obs


def _detectar_vecnorm(
    ruta_snapshot: str, base_dir: Optional[str] = None
) -> Optional[str]:
    """Detecta el archivo VecNormalize asociado a un snapshot.

    Busca en orden:
        1. Per-snapshot: <ruta_snapshot>_vecnorm.pkl
        2. Global v5: vecnormalize/v5/v5_vecnorm.pkl
        3. Global v5 final: vecnormalize/v5/v5_vecnorm_final.pkl

    Args:
        ruta_snapshot: Ruta al snapshot (con o sin .zip).
        base_dir: Directorio base del proyecto (default: detectado).

    Returns:
        Ruta al .pkl o None.
    """
    if base_dir is None:
        base_dir = _PROJECT_DIR

    ruta_limpia = ruta_snapshot.replace(".zip", "")
    candidatos = [
        ruta_limpia + "_vecnorm.pkl",
    ]
    # Agregar paths globales solo si base_dir coincide con el proyecto real
    v5_global = os.path.join(base_dir, "vecnormalize", "v5", "v5_vecnorm.pkl")
    v5_final = os.path.join(base_dir, "vecnormalize",
                            "v5", "v5_vecnorm_final.pkl")
    candidatos.extend([v5_global, v5_final])
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None


# ------------------------------------------------------------------
# Nivel 1: Evaluación contra bots
# ------------------------------------------------------------------

def evaluar_contra_bots(
    modelo: Any,
    num_partidas: int = 100,
    vecnorm_path: Optional[str] = None,
    verbose: bool = False,
    shuffle_bots: bool = True,
) -> Dict[str, float]:
    """Evalúa el modelo contra 3 bots heurísticos.

    Args:
        modelo: Instancia de MaskablePPO.
        num_partidas: Número de partidas a jugar.
        vecnorm_path: Ruta al VecNormalize .pkl (opcional).
        verbose: Si es True, imprime progreso cada 25 partidas.
        shuffle_bots: Si es True, reordena los bots en cada partida.

    Returns:
        Diccionario de métricas.
    """
    from src.bots import bot_conservador, bot_agresivo, bot_evasivo
    from src.entorno import CorazonesEnv

    bots_pool = [bot_conservador, bot_agresivo, bot_evasivo]
    posiciones: List[int] = []
    puntuaciones: List[float] = []

    for seed in range(num_partidas):
        if shuffle_bots:
            orden = np.random.permutation(3)
            bots = [bots_pool[orden[0]],
                    bots_pool[orden[1]], bots_pool[orden[2]]]
        else:
            bots = list(bots_pool)

        env = CorazonesEnv(
            agente_idx=0,
            politicas_oponentes={1: bots[0], 2: bots[1], 3: bots[2]},
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
        posiciones.append(_calcular_posicion(punt_agente, punt_rivales))
        puntuaciones.append(float(punt_agente))
        env.close()

        if verbose and (seed + 1) % 25 == 0:
            w = sum(1 for p in posiciones if p == 0)
            t2 = sum(1 for p in posiciones if p in (0, 1))
            print(f"  {seed + 1}/{num_partidas} partidas... ({w} 1º, {t2} top-2)")

    return _construir_metricas(posiciones, puntuaciones, num_partidas)


# ------------------------------------------------------------------
# Nivel 2: Evaluación contra snapshots (Self-Play eval)
# ------------------------------------------------------------------

class _PoliticaSnapshot:
    """Adaptador que envuelve un MaskablePPO para usarlo como oponente en CorazonesEnv.

    El entorno llama ``politica(motor, jugador_idx, legales) -> Carta``.
    Este adaptador construye la observación, normaliza con VecNormalize,
    llama a model.predict() con action masks, y devuelve la Carta elegida.
    """

    def __init__(self, model: Any, vecnorm_path: Optional[str] = None):
        self.model = model
        self._obs_rms = None
        if vecnorm_path and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    vn = pickle.load(f)
                self._obs_rms = vn.obs_rms
            except Exception:
                self._obs_rms = None

    def _construir_obs(self, motor: Any, jugador_idx: int) -> np.ndarray:
        """Construye vector 194-dim (v6) desde la perspectiva del jugador."""
        from src.carta import Carta
        obs = np.zeros(194, dtype=np.float32)
        a = jugador_idx

        for c in motor.jugadores[a].mano:
            obs[c.id] = 1.0
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0
        for i in range(4):
            for c in motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        # Vacíos conocidos (simplificado: no trackeamos voids precisos del oponente)
        # Puntajes históricos
        for i in range(4):
            rel = (i - a) % 4
            score = motor.jugadores[i].puntuacion_historica
            obs[172 + rel] = min(score / 100.0, 1.0)

        # Puntos mano actual
        for i in range(4):
            rel = (i - a) % 4
            pts = motor.jugadores[i].contar_puntos_bazas()
            obs[176 + rel] = min(pts / 26.0, 1.0)

        obs[180] = 1.0 if motor.corazones_rotos else 0.0
        posiciones = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
        obs[181] = posiciones.get(len(motor.mesa), 0.0)

        # Features v5+v6 se dejan en 0 (no disponibles desde motor para oponentes)

        return obs

    def __call__(self, motor: Any, jugador_idx: int, legales: List[Any]) -> Any:
        from src.carta import Carta
        obs = self._construir_obs(motor, jugador_idx)

        if self._obs_rms is not None:
            mean = np.array(self._obs_rms.mean)
            var = np.array(self._obs_rms.var)
            # Padding: stats de 190-dim → 194-dim (nuevos features: mean=0, var=1)
            if len(mean) == 190:
                mean = np.concatenate([mean, np.zeros(4, dtype=np.float32)])
                var = np.concatenate([var, np.ones(4, dtype=np.float32)])
            obs = np.clip((obs - mean) / np.sqrt(var + 1e-8),
                          -10.0, 10.0).astype(np.float32)

        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True

        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True)
        return Carta._TODAS[int(action)]


def evaluar_contra_snapshots(
    modelo: Any,
    num_partidas: int = 50,
    vecnorm_path: Optional[str] = None,
    snapshots_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evalúa el modelo contra sus snapshots históricos (Self-Play eval).

    Toma los últimos 5 snapshots del directorio v5 y los usa como oponentes.
    Un win rate >50% contra snapshots anteriores indica mejora real.

    Args:
        modelo: Instancia de MaskablePPO actual.
        num_partidas: Partidas por snapshot oponente.
        vecnorm_path: Ruta al VecNormalize .pkl.
        snapshots_dir: Directorio de snapshots v5.
        verbose: Imprimir progreso.

    Returns:
        Diccionario con win_rate_promedio y métricas agregadas.
    """
    if snapshots_dir is None:
        snapshots_dir = _MODELOS_V5_DIR

    import glob
    snaps = sorted(glob.glob(os.path.join(snapshots_dir, "snapshot_*.zip")))
    if len(snaps) < 2:
        return {"win_rate_vs_self": 0.0, "num_oponentes": 0, "partidas": 0}

    # Usar los últimos 5 snapshots (excluyendo el más reciente si es el actual)
    oponentes = snaps[-6:-1] if len(snaps) >= 6 else snaps[:-1]
    if len(oponentes) == 0:
        oponentes = snaps[-1:]

    from sb3_contrib import MaskablePPO
    from src.entorno import CorazonesEnv
    from src.bots import bot_conservador, bot_agresivo, bot_evasivo

    bots = [bot_conservador, bot_agresivo, bot_evasivo]
    todas_posiciones: List[int] = []
    todas_puntuaciones: List[float] = []

    for snap_path in oponentes:
        snap_vecnorm = snap_path.replace(".zip", "_vecnorm.pkl")
        if not os.path.exists(snap_vecnorm):
            snap_vecnorm = vecnorm_path

        try:
            modelo_oponente = MaskablePPO.load(snap_path, device="cpu")
        except Exception:
            continue

        # Envolver snapshot en adaptador para que CorazonesEnv lo use como oponente
        snap_politica = _PoliticaSnapshot(modelo_oponente, snap_vecnorm)

        for seed in range(num_partidas):
            # Snapshots como oponentes: slot 1 es el snapshot, 2 y 3 son bots
            politicas = {
                1: snap_politica,
                2: bots[seed % 3],
                3: bots[(seed + 1) % 3],
            }
            env = CorazonesEnv(agente_idx=0, politicas_oponentes=politicas)
            obs_raw, _ = env.reset(seed=seed)
            obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
            done = False

            while not done:
                mask = env.action_masks()
                action, _ = modelo.predict(
                    obs, action_masks=mask, deterministic=True)
                obs_raw, _reward, terminated, truncated, _ = env.step(
                    int(action))
                obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
                done = terminated or truncated

            punt_agente = env._puntuacion_historica[0]
            punt_rivales = [env._puntuacion_historica[i] for i in range(1, 4)]
            todas_posiciones.append(
                _calcular_posicion(punt_agente, punt_rivales))
            todas_puntuaciones.append(float(punt_agente))
            env.close()

        if verbose:
            snap_name = os.path.basename(snap_path).replace(".zip", "")
            w = sum(1 for p in todas_posiciones[-num_partidas:] if p == 0)
            print(f"  vs {snap_name}: {w}/{num_partidas} victorias")

    total = len(todas_posiciones)
    metricas = _construir_metricas(todas_posiciones, todas_puntuaciones, total)
    metricas["num_oponentes"] = len(oponentes)
    return metricas


# ------------------------------------------------------------------
# Nivel 3: Escenarios estratégicos
# ------------------------------------------------------------------

def _crear_escenario_q_spades_sin_pozo() -> Dict[str, Any]:
    """Escenario donde el agente tiene Q♠ pero el pozo NO es viable.

    El agente debe evitar jugar Q♠ para no comerse 13 pts innecesarios.
    """
    return {
        "nombre": "Q♠ sin pozo viable",
        "mano_agente": [
            23,  # Q♠
            0, 1, 2, 3,        # 4 tréboles bajos
            13, 14, 15,         # 3 diamantes bajos
            39, 40, 41,         # 3 corazones bajos (no pozo viable)
            26, 27,             # 2 picas bajas
        ],
        "manos_rivales": [
            [4, 5, 6, 7, 16, 17, 18, 28, 29, 30, 42, 43, 44],
            [8, 9, 10, 11, 19, 20, 21, 31, 32, 33, 45, 46, 47],
            [12, 22, 24, 25, 34, 35, 36, 37, 38, 48, 49, 50, 51],
        ],
        "corazones_rotos": False,
        "palo_inicial": 0,  # Trébol
        "comportamiento_deseado": "NO jugar Q♠ en la primera oportunidad",
    }


def _crear_escenario_pozo_viable() -> Dict[str, Any]:
    """Escenario donde el agente tiene mano para shooting the moon.

    Debe intentar ganar todas las bazas (o al menos no evitar corazones).
    """
    return {
        "nombre": "Pozo viable (shooting the moon)",
        "mano_agente": [
            # 7 corazones incluyendo A♥, K♥, Q♥, J♥
            51, 50, 49, 48,     # A♥, K♥, Q♥, J♥
            45, 44, 43,         # 8♥, 7♥, 6♥
            # Cartas altas de otros palos para ganar bazas
            12, 25, 38,         # A♣, A♠, A♦
            11, 24, 37,         # K♣, K♠, K♦
        ],
        "manos_rivales": [
            [0, 1, 2, 3, 13, 14, 15, 26, 27, 28, 39, 40, 41],
            [4, 5, 6, 7, 16, 17, 18, 29, 30, 31, 42, 46, 47],
            [8, 9, 10, 19, 20, 21, 22, 32, 33, 34, 35, 36, 37],
        ],
        "corazones_rotos": False,
        "palo_inicial": 0,
        "comportamiento_deseado": "Ganar bazas con corazones, buscar pozo",
    }


def evaluar_escenario(
    modelo: Any,
    escenario: Dict[str, Any],
    num_repeticiones: int = 20,
    vecnorm_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evalúa el modelo en un escenario estratégico específico.

    Args:
        modelo: Instancia de MaskablePPO.
        escenario: Diccionario con la configuración del escenario.
        num_repeticiones: Veces que se repite el escenario.
        vecnorm_path: Ruta al VecNormalize .pkl.

    Returns:
        Diccionario con métricas del escenario.
    """
    from src.entorno import CorazonesEnv
    from src.carta import Carta
    from src.bots import bot_conservador, bot_agresivo, bot_evasivo

    nombre = escenario.get("nombre", "sin_nombre")
    bots = [bot_conservador, bot_agresivo, bot_evasivo]

    veces_juega_q_spades = 0
    veces_gana_baza_con_corazon = 0
    puntuaciones: List[float] = []
    posiciones: List[int] = []

    for rep in range(num_repeticiones):
        env = CorazonesEnv(
            agente_idx=0,
            politicas_oponentes={1: bots[0], 2: bots[1], 3: bots[2]},
        )
        env.reset(seed=rep)

        # Inyectar el escenario en el motor
        # Nota: esto es frágil; el motor reparte automáticamente.
        # Para escenarios controlados, manipulamos el estado post-reparto.
        motor = env.motor
        mano_agente = [Carta._TODAS[cid] for cid in escenario["mano_agente"]]
        motor.jugadores[0].mano = list(mano_agente)
        motor.corazones_rotos = escenario.get("corazones_rotos", False)

        obs_raw = env._construir_observacion()
        obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
        done = False

        while not done:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True)
            carta_jugada = Carta._TODAS[int(action)]

            if carta_jugada.es_dama_de_picas:
                veces_juega_q_spades += 1

            obs_raw, _reward, terminated, truncated, _ = env.step(int(action))
            obs = normalizar_obs_si_hay_stats(obs_raw, vecnorm_path)
            done = terminated or truncated

            # Detectar si la baza ganada por el agente contenía corazones
            if len(motor.mesa) == 0 and not done:
                baza_anterior = motor.jugadores[0].bazas_ganadas
                # Simplificación: verificamos al final de la mano

        punt_agente = env._puntuacion_historica[0]
        punt_rivales = [env._puntuacion_historica[i] for i in range(1, 4)]
        puntuaciones.append(float(punt_agente))
        posiciones.append(_calcular_posicion(punt_agente, punt_rivales))
        env.close()

    metricas = _construir_metricas(posiciones, puntuaciones, num_repeticiones)
    metricas["escenario"] = nombre
    metricas["pct_juega_q_spades"] = veces_juega_q_spades / num_repeticiones
    return metricas


# ------------------------------------------------------------------
# Evaluación completa (orquesta los 2 niveles rápidos)
# ------------------------------------------------------------------

def evaluacion_rapida(
    ruta_modelo: str,
    num_partidas_bots: int = 100,
    num_partidas_self: int = 30,
) -> Dict[str, Any]:
    """Ejecuta evaluación rápida de 2 niveles (bots + self-play).

    Args:
        ruta_modelo: Ruta al snapshot .zip.
        num_partidas_bots: Partidas contra bots.
        num_partidas_self: Partidas contra snapshots antiguos.

    Returns:
        Diccionario con resultados agregados.
    """
    if not ruta_modelo.endswith(".zip"):
        ruta_modelo += ".zip"

    from sb3_contrib import MaskablePPO

    modelo = MaskablePPO.load(ruta_modelo, device="cpu")
    vecnorm_path = _detectar_vecnorm(ruta_modelo)

    resultado: Dict[str, Any] = {
        "snapshot": os.path.basename(ruta_modelo).replace(".zip", ""),
        "timestamp": datetime.now().isoformat(),
    }

    # Nivel 1: Bots
    metricas_bots = evaluar_contra_bots(
        modelo, num_partidas=num_partidas_bots,
        vecnorm_path=vecnorm_path, verbose=True)
    resultado["bots"] = metricas_bots
    resultado["win_rate_bots"] = metricas_bots["pct_primero"]
    resultado["top2_bots"] = metricas_bots["pct_top2"]
    resultado["punt_promedio_bots"] = metricas_bots["punt_promedio"]

    # Nivel 2: Self-Play (solo si hay ≥2 snapshots)
    metricas_self = evaluar_contra_snapshots(
        modelo, num_partidas=num_partidas_self,
        vecnorm_path=vecnorm_path, verbose=True)
    resultado["self_play"] = metricas_self
    resultado["win_rate_self"] = metricas_self.get("pct_primero", 0.0)

    return resultado


# ------------------------------------------------------------------
# Funciones para integración con train_self_play.py
# ------------------------------------------------------------------

def _guardar_log_evaluacion(log_path: str, resultado: Dict[str, Any]) -> None:
    """Agrega una línea JSON al archivo de log de evaluaciones.

    Args:
        log_path: Ruta al archivo .jsonl (si está vacío, no escribe).
        resultado: Diccionario con métricas de la evaluación.
    """
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")


def evaluar_snapshot_callback(
    ruta_snapshot: str,
    paso: int,
    log_path: str,
    num_partidas: int = 100,
    min_win_rate: float = 0.0,
) -> Dict[str, Any]:
    """Callback de evaluación para el bucle de entrenamiento.

    Evalúa el snapshot recién guardado y registra el resultado.
    Si el win rate cae por debajo de min_win_rate, lo reporta
    (la decisión de parar la toma el llamador).

    Args:
        ruta_snapshot: Ruta al snapshot .zip recién guardado.
        paso: Paso global de entrenamiento.
        log_path: Ruta al archivo .jsonl de log.
        num_partidas: Partidas de evaluación.
        min_win_rate: Umbral mínimo de win rate (0.0 = sin early stopping).

    Returns:
        Diccionario con el resultado de la evaluación.
    """
    from sb3_contrib import MaskablePPO

    if not ruta_snapshot.endswith(".zip"):
        ruta_snapshot += ".zip"

    modelo = MaskablePPO.load(ruta_snapshot, device="cpu")
    vecnorm_path = _detectar_vecnorm(ruta_snapshot)

    print(f"\n  📊 [Eval] Evaluando snapshot paso {paso:,}...")
    metricas = evaluar_contra_bots(
        modelo, num_partidas=num_partidas,
        vecnorm_path=vecnorm_path, verbose=False)

    resultado = {
        "paso": paso,
        "snapshot": os.path.basename(ruta_snapshot).replace(".zip", ""),
        "timestamp": datetime.now().isoformat(),
        "win_rate_bots": metricas["pct_primero"],
        "top2_bots": metricas["pct_top2"],
        "punt_promedio": metricas["punt_promedio"],
        "punt_mediana": metricas["punt_mediana"],
        "pct_cuarto": metricas["pct_cuarto"],
    }

    _guardar_log_evaluacion(log_path, resultado)

    wr = resultado["win_rate_bots"]
    print(f"  📊 [Eval] WR={wr:.1%} | Top-2={resultado['top2_bots']:.1%} "
          f"| AvgPts={resultado['punt_promedio']:.1f} | 4º={resultado['pct_cuarto']:.1%}")

    if min_win_rate > 0 and wr < min_win_rate:
        print(
            f"  ⚠️  [Eval] Win rate {wr:.1%} < umbral {min_win_rate:.1%} — early stopping")

    return resultado


# ------------------------------------------------------------------
# CLI standalone
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluación multi-nivel de Corazones RL")
    parser.add_argument("--ruta", type=str, default=None,
                        help="Ruta al snapshot .zip (requerido excepto con --elo)")
    parser.add_argument("--partidas", type=int, default=100,
                        help="Partidas contra bots")
    parser.add_argument("--self-play", action="store_true",
                        help="Incluir evaluación Self-Play")
    parser.add_argument("--elo", action="store_true",
                        help="Ejecutar torneo Elo entre todos los snapshots v5")
    parser.add_argument("--elo-partidas", type=int, default=30,
                        help="Partidas por enfrentamiento en torneo Elo (default=30)")
    parser.add_argument("--elo-max-snaps", type=int, default=15,
                        help="Máximo de snapshots en torneo Elo (default=15)")
    parser.add_argument("--elo-puro", action="store_true",
                        help="Modo Elo puro: snapshots adyacentes en vez de bots")
    parser.add_argument("--elo-min-paso", type=int, default=0,
                        help="Paso mínimo para snapshots en torneo (default=0)")
    parser.add_argument("--elo-dir", type=str, default=None,
                        help="Directorio de snapshots para torneo (default: v5)")
    parser.add_argument("--elo-dirs", type=str, default=None, nargs="+",
                        help="Directorios múltiples (v5+v6 juntos). Ej: --elo-dirs modelos_historicos/v5 modelos_historicos/v6")
    args = parser.parse_args()

    # Modo torneo Elo (no requiere --ruta)
    if args.elo:
        from src.elo_torneo import torneo_elo
        # Determinar directorios para el torneo
        if args.elo_dirs:
            dirs = args.elo_dirs
        elif args.elo_dir:
            dirs = [args.elo_dir]
        else:
            dirs = None  # default: solo v5

        resultado = torneo_elo(
            directorios=dirs,
            num_partidas=args.elo_partidas,
            max_snapshots=args.elo_max_snaps,
            min_paso=args.elo_min_paso,
            verbose=True,
            elo_puro=args.elo_puro,
        )
        if not resultado["ranking"]:
            print("No se encontraron snapshots para el torneo.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("🏆 CLASIFICACIÓN FINAL ELO")
        print("=" * 60)
        print(f"{'Pos':<5} {'Snapshot':<30} {'Origen':<8} {'Paso':<12} {'Elo':<8} {'Δ'}")
        print("-" * 60)
        for pos, (snap, rating, paso, origen) in enumerate(resultado["ranking"]):
            nombre = resultado["nombres"].get(
                snap, os.path.basename(snap))
            delta = ""
            if pos > 0:
                _, prev_r, _, _ = resultado["ranking"][pos - 1]
                delta = f"{rating - prev_r:+.0f}"
            print(
                f"{pos+1:<5} {nombre:<30} {origen:<8} {paso:>10,}  {rating:<8.0f} {delta}")
        print("-" * 60)
        sys.exit(0)

    # Modo normal: requiere --ruta
    if not args.ruta:
        parser.error("Se requiere --ruta o --elo")

    print("=" * 55)
    print(f"Evaluación multi-nivel: {args.ruta}")
    print("=" * 55)

    resultado = evaluacion_rapida(
        args.ruta,
        num_partidas_bots=args.partidas,
        num_partidas_self=30 if args.self_play else 0,
    )

    print("\n--- Resultados contra bots ---")
    b = resultado["bots"]
    print(f"  🥇 1º: {b['pct_primero']:.1%} | 🥈 2º: {b['pct_segundo']:.1%}")
    print(f"  📊 Top-2: {b['pct_top2']:.1%} | 💀 4º: {b['pct_cuarto']:.1%}")
    print(f"  Punt: μ={b['punt_promedio']:.1f} med={b['punt_mediana']:.1f}")

    if args.self_play and resultado.get("self_play", {}).get("num_oponentes", 0) > 0:
        s = resultado["self_play"]
        print(
            f"\n--- Resultados Self-Play ({s.get('num_oponentes', '?')} oponentes) ---")
        print(
            f"  🥇 1º: {s.get('pct_primero', 0):.1%} | Top-2: {s.get('pct_top2', 0):.1%}")
