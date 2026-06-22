"""Evaluacion estandarizada para modelos v3.

Proporciona un formato canonico para evaluar UN modelo individual
contra un campo fijo y realista de oponentes:

    Mesa: [Modelo, BotExperto, BotExperto, BotRotativo]

Donde BotRotativo alterna entre {conservador, agresivo, evasivo}
cada mano para garantizar diversidad tactica.

Este modulo es el Single Source of Truth (SSOT) para evaluaciones
individuales. Tanto train.py como los scripts de analisis deben
importar desde aqui.

Uso:
    from src.v3.evaluacion import evaluar_estandar, EvaluacionEstandar

    res = evaluar_estandar(
        nombre="snapshot_0004300000",
        modelo_path="models/v3/snapshots/snapshot_0004300000",
        num_manos=200,
        seed_base=42,
    )
    print(f"WR estandar: {res.wr:.1%} | Score: {res.avg_score:.1f}")
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


# ------------------------------------------------------------------
# Dataclass de resultado
# ------------------------------------------------------------------

@dataclass
class EvaluacionEstandar:
    """Resultado de una evaluacion estandarizada.

    Attributes:
        wr: Win rate — fraccion de manos con score <= 8.
        avg_score: Puntuacion promedio del agente.
        num_manos: Total de manos jugadas.
        nombre: Identificador del jugador evaluado.
        scores: Lista de puntuaciones por mano.
        es_bot: Si el jugador es un bot heuristico.
        es_experto: Si el jugador es BotExperto.
    """
    wr: float = 0.0
    avg_score: float = 0.0
    num_manos: int = 0
    nombre: str = ""
    scores: List[float] = field(default_factory=list)
    es_bot: bool = False
    es_experto: bool = False


# ------------------------------------------------------------------
# Construccion del campo estandar
# ------------------------------------------------------------------


def _pad_vecnorm_stats(stats: np.ndarray, target_dim: int, pad_value: float = 0.0) -> np.ndarray:
    """Asegura que las stats de VecNormalize tengan target_dim elementos.

    Si las stats son mas cortas (modelo antiguo con menos dims),
    rellena con pad_value las nuevas dimensiones.
    - mean: pad_value=0 (sin desplazamiento en nuevas dims)
    - var: pad_value=1 (sin escalado en nuevas dims)
    """
    if stats.shape[0] >= target_dim:
        return stats[:target_dim].copy()
    padded = np.full(target_dim, pad_value, dtype=stats.dtype)
    padded[:stats.shape[0]] = stats
    return padded


def _construir_oponentes_estandar(
    agente_idx: int,
    seed: int,
) -> Dict[int, Callable]:
    """Construye el campo [Experto, Experto, BotRotativo] dejando
    libre el asiento del agente.

    Args:
        agente_idx: Posicion del agente evaluado (0-3).
        seed: Semilla para elegir el bot rotativo.

    Returns:
        Dict {seat_idx: politica_callable} para los 3 oponentes.
    """
    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import (
        bot_conservador, bot_agresivo, bot_evasivo,
    )

    bots_heuristicos = [bot_conservador, bot_agresivo, bot_evasivo]
    bot_rotativo = bots_heuristicos[seed % 3]

    # Asientos ocupados por los oponentes (todos menos el agente)
    seats_oponentes = [s for s in range(4) if s != agente_idx]

    # Asignar: 2 Expertos + 1 bot rotativo
    politicas: Dict[int, Callable] = {}
    politicas[seats_oponentes[0]] = BotExperto()
    politicas[seats_oponentes[1]] = BotExperto()
    politicas[seats_oponentes[2]] = bot_rotativo

    return politicas


# ------------------------------------------------------------------
# Evaluacion principal
# ------------------------------------------------------------------

def evaluar_estandar(
    nombre: str,
    num_manos: int = 200,
    seed_base: int = 42,
    es_bot: bool = False,
    es_experto: bool = False,
    bot_idx: int = 0,
    modelo_path: Optional[str] = None,
    device: str = "cpu",
) -> EvaluacionEstandar:
    """Evalua un jugador contra el campo estandar [Experto, Experto, Bot].

    Soporta tres tipos de jugador:
      - Bot heuristico (es_bot=True, bot_idx=0/1/2)
      - BotExperto (es_experto=True)
      - Modelo MaskablePPO v3 (modelo_path=...)

    Args:
        nombre: Identificador del jugador.
        num_manos: Manos independientes a jugar.
        seed_base: Semilla base.
        es_bot: True si es un bot heuristico.
        es_experto: True si es BotExperto.
        bot_idx: Indice del bot (0=conservador, 1=agresivo, 2=evasivo).
        modelo_path: Ruta al snapshot .zip del modelo v3.
        device: Dispositivo para cargar el modelo.

    Returns:
        EvaluacionEstandar con wr, avg_score, scores, etc.
    """
    from src.agentes.heuristicos import (
        bot_conservador, bot_agresivo, bot_evasivo,
    )
    from src.agentes.bot_experto import BotExperto
    from src.dominio.motor import MotorCorazones
    from src.dominio.carta import Carta

    # ── Preparar politica del agente ──
    modelo = None
    obs_mean = None
    obs_var = None

    if es_experto:
        politica_agente: Any = BotExperto()
    elif es_bot:
        bots_pool = [bot_conservador, bot_agresivo, bot_evasivo]
        politica_agente = bots_pool[bot_idx % 3]
    else:
        politica_agente = None  # se usara _predecir_modelo()
        if modelo_path:
            from sb3_contrib import MaskablePPO
            ruta = modelo_path
            if not ruta.endswith(".zip"):
                ruta += ".zip"
            modelo = MaskablePPO.load(ruta, device=device)
            # Cargar VecNormalize stats
            vn_path = ruta.replace(".zip", "_vecnorm.pkl")
            if os.path.exists(vn_path):
                with open(vn_path, "rb") as f:
                    vn = pickle.load(f)
                obs_mean = vn.obs_rms.mean
                obs_var = vn.obs_rms.var
                # Compatibilidad: pad a DIM_V3 si son de version anterior
                from src.entorno.dimensiones import DIM_V3
                obs_mean = _pad_vecnorm_stats(obs_mean, DIM_V3, pad_value=0.0)
                obs_var = _pad_vecnorm_stats(obs_var, DIM_V3, pad_value=1.0)

    # ── Jugar manos ──
    victorias = 0
    scores: List[float] = []

    for h in range(num_manos):
        seed = seed_base + h
        motor = MotorCorazones()
        motor.repartir()

        # Asignar asientos: agente en posicion rotativa
        agente_idx = h % 4

        # Construir campo de oponentes
        oponentes = _construir_oponentes_estandar(agente_idx, seed)

        # Mapa completo de politicas: agente + oponentes
        politicas: Dict[int, Any] = dict(oponentes)
        politicas[agente_idx] = politica_agente

        # Jugar las 13 bazas
        for _ in range(13):
            for _ in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)

                if idx == agente_idx and modelo is not None:
                    carta = _predecir_modelo(
                        modelo, motor, idx, legales,
                        obs_mean, obs_var, seed)
                else:
                    pol = politicas.get(idx)
                    if pol is not None:
                        carta = pol(motor, idx, legales)
                    else:
                        carta = legales[0]

                motor.jugar_carta(idx, carta)
            motor.resolver_baza()

        # Resultado de esta mano
        score_agente = motor.jugadores[agente_idx].contar_puntos_bazas()
        scores.append(float(score_agente))
        if score_agente <= 8:
            victorias += 1

    return EvaluacionEstandar(
        wr=victorias / num_manos,
        avg_score=float(np.mean(scores)),
        num_manos=num_manos,
        nombre=nombre,
        scores=scores,
        es_bot=es_bot,
        es_experto=es_experto,
    )


# ------------------------------------------------------------------
# Prediccion del modelo (helper interno)
# ------------------------------------------------------------------

def _predecir_modelo(
    modelo: Any,
    motor: Any,
    idx: int,
    legales: List[Any],
    obs_mean: Optional[np.ndarray],
    obs_var: Optional[np.ndarray],
    seed: int,
) -> Any:
    """Predice una carta usando el modelo MaskablePPO con VecNormalize.

    Construye la observacion v3 (250-dim) con puntuacion_historica=[0,0,0,0]
    porque cada mano es independiente (single-hand).
    """
    from src.v3.observacion import ObservacionBuilderV3, DIM_V3
    from src.dominio.carta import Carta

    builder = ObservacionBuilderV3(dim=DIM_V3)
    obs_raw = builder.construir(
        motor, idx,
        vacios=[set() for _ in range(4)],
        puntuacion_historica=[0, 0, 0, 0],
        puntos_mano_actual=[j.contar_puntos_bazas() for j in motor.jugadores],
        dama_picas_en=None,
    )

    if obs_mean is not None and obs_var is not None:
        obs = np.clip(
            (obs_raw - obs_mean) /
            (np.sqrt(obs_var) + 1e-8), -10, 10,
        ).astype(np.float32)
    else:
        obs = obs_raw.astype(np.float32)

    mask = np.zeros(52, dtype=np.bool_)
    for c in legales:
        mask[c.id] = True

    # Truncar obs al espacio del modelo si es mas pequeno (compatibilidad cross-gen)
    model_dim = modelo.observation_space.shape[0]
    if obs.shape[0] > model_dim:
        obs = obs[:model_dim]

    action, _ = modelo.predict(
        obs, action_masks=mask, deterministic=True,
    )
    return Carta._TODAS[int(action)]
