#!/usr/bin/env python
"""
Análisis comparativo exhaustivo: v3_mcts 400K vs v3_base 3.4M vs BotExperto.

Usa MotorCorazones directamente (sin Gym env) para tener acceso al estado
en cada decisión y poder hacer análisis PIMC on-the-fly.

Evalúa:
  - Win rate, score medio, distribución
  - Q♠ capture rate
  - Moon shooting (pozo logrado / bloqueado)
  - PIMC error analysis (bazas ≥ 8, mundos ≤ 100K)
  - Errores por situación (liderar/seguir/descartar) y etapa (early/mid/late)

Uso:
    python scripts/analisis_comparativo.py
"""

from __future__ import annotations
from src.mcts.analisis import analizar_decision
from src.mcts.pimc import _clonar_motor
from src.entorno.dimensiones import DIM_V3
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────

MODELOS = [
    ("v3_mcts_400K", "models/v3_mcts/snapshots/snapshot_0000400000"),
    ("v3_base_3.4M", "models/v3/snapshots/snapshot_0003400000"),
    ("BotExperto", None),
]

NUM_MANOS: int = 100
SEED_BASE: int = 12345
BAZA_MIN_PIMC: int = 10
MAX_MUNDOS_PIMC: int = 100_000
DEVICE: str = "cpu"


# ──────────────────────────────────────────────────────────────
# Carga de modelos
# ──────────────────────────────────────────────────────────────

def cargar_modelo_y_vecnorm(ruta: str, device: str = "cpu"):
    """Carga MaskablePPO + VecNormalize stats desde una ruta de snapshot.

    Retorna (modelo, obs_mean, obs_var, obs_dim).
    obs_dim = la dimensión real que el modelo espera (detectada del VecNorm o del modelo).
    Las stats se paddean a obs_dim si son más cortas.
    """
    from sb3_contrib import MaskablePPO

    ruta_clean = ruta.replace(".zip", "")
    modelo = MaskablePPO.load(ruta_clean, device=device)

    # Detectar dim real del modelo
    obs_dim = modelo.observation_space.shape[0]

    obs_mean = None
    obs_var = None
    vn_path = ruta_clean + "_vecnorm.pkl"
    if os.path.exists(vn_path):
        with open(vn_path, "rb") as f:
            vn = pickle.load(f)
        obs_mean = vn.obs_rms.mean
        obs_var = vn.obs_rms.var
        orig_dim = obs_mean.shape[0]
        if orig_dim < obs_dim:
            obs_mean = _pad_vecnorm_stats(obs_mean, obs_dim, pad_value=0.0)
            obs_var = _pad_vecnorm_stats(obs_var, obs_dim, pad_value=1.0)
            print(f"  VecNorm: {orig_dim}d→{obs_dim}d (padded)")
        else:
            print(f"  VecNorm: {obs_dim}d")
    else:
        print("  WARN: VecNorm no encontrado")

    return modelo, obs_mean, obs_var, obs_dim


def _pad_vecnorm_stats(
    stats: np.ndarray, target: int, pad_value: float
) -> np.ndarray:
    if stats.shape[0] >= target:
        return stats[:target].copy()
    out = np.full(target, pad_value, dtype=stats.dtype)
    out[:stats.shape[0]] = stats
    return out


def _predecir(
    modelo, obs: np.ndarray, mask: np.ndarray,
    obs_mean: Optional[np.ndarray], obs_var: Optional[np.ndarray],
) -> int:
    """Predice acción con normalización VecNorm."""
    o = obs.copy()
    if obs_mean is not None and obs_var is not None:
        o = np.clip(
            (o - obs_mean) / np.sqrt(obs_var + 1e-8), -10.0, 10.0
        ).astype(np.float32)
    action, _ = modelo.predict(o, action_masks=mask, deterministic=True)
    return int(action)


# ──────────────────────────────────────────────────────────────
# Oponentes
# ──────────────────────────────────────────────────────────────

def _construir_oponentes_estandar(
    agente_idx: int, seed: int,
) -> Dict[int, Any]:
    """[Experto, Experto, Bot rotativo] en los 3 asientos sin el agente."""
    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo

    bots = [bot_conservador, bot_agresivo, bot_evasivo]
    bot_rot = bots[seed % 3]

    seats = [s for s in range(4) if s != agente_idx]
    return {
        seats[0]: BotExperto(),
        seats[1]: BotExperto(),
        seats[2]: bot_rot,
    }


# ──────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────

@dataclass
class Agregado:
    nombre: str
    total: int = 0
    victorias: int = 0
    score_total: float = 0.0
    scores: List[float] = field(default_factory=list)
    manos_cero: int = 0
    q_capturadas: int = 0
    pozos_logrados: int = 0
    pozos_bloqueados: int = 0
    por_seat: Dict[int, List[float]] = field(
        default_factory=lambda: {s: [] for s in range(4)})

    # PIMC
    n_pimc: int = 0
    n_errores: int = 0
    coste_total: float = 0.0
    err_liderar: int = 0
    err_seguir: int = 0
    err_descartar: int = 0
    err_early: int = 0
    err_mid: int = 0
    err_late: int = 0
    graves: List[Dict] = field(default_factory=list)

    @property
    def wr(self) -> float:
        return self.victorias / self.total * 100 if self.total else 0

    @property
    def avg(self) -> float:
        return self.score_total / self.total if self.total else 0

    @property
    def tasa_q(self) -> float:
        return self.q_capturadas / self.total * 100 if self.total else 0

    @property
    def tasa_err(self) -> float:
        return self.n_errores / self.n_pimc * 100 if self.n_pimc else 0

    @property
    def coste_medio(self) -> float:
        return self.coste_total / self.n_errores if self.n_errores else 0

    @property
    def coste_por_decision(self) -> float:
        return self.coste_total / self.n_pimc if self.n_pimc else 0


# ──────────────────────────────────────────────────────────────
# Evaluación con PIMC
# ──────────────────────────────────────────────────────────────

def evaluar_modelo(
    nombre: str, modelo, obs_mean, obs_var, obs_dim: int,
    num_manos: int, seed_base: int,
) -> Agregado:
    """Evalúa modelo RL haciendo PIMC en cada decisión de bazas ≥ 8."""
    agg = Agregado(nombre=nombre)

    # Elegir builder según dimensión
    if obs_dim >= 260:
        from src.v3.observacion import ObservacionBuilderV3
        obs_builder = ObservacionBuilderV3(dim=obs_dim)
    else:
        from src.entorno.observacion import ObservacionBuilder
        obs_builder = ObservacionBuilder(dim=obs_dim)

    for h in range(num_manos):
        seed = seed_base + h
        rng = np.random.default_rng(seed)
        agente_idx = h % 4

        motor = MotorCorazones()
        motor.repartir()

        oponentes = _construir_oponentes_estandar(agente_idx, seed)

        # Debug counters
        n_pimc_attempts = 0
        n_pimc_skipped_mundos = 0
        n_pimc_exceptions = 0

        # ── Jugar hasta que termine la mano ──
        while motor._mano_activa:
            if len(motor.mesa) == 4:
                motor.resolver_baza()
                if not motor._mano_activa:
                    break

            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)
            if not legales:
                break  # safety

            if idx == agente_idx:
                obs = obs_builder.construir(
                    motor, agente_idx,
                    [set() for _ in range(4)],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    None,
                )
                mask = np.zeros(52, dtype=np.bool_)
                for c in legales:
                    mask[c.id] = True

                # ── PIMC hook ──
                hacer_pimc = (
                    motor.numero_baza >= BAZA_MIN_PIMC
                    and len(legales) > 1
                )
                pimc_done = False
                if hacer_pimc:
                    n_pimc_attempts += 1
                    try:
                        mc = _clonar_motor(motor)
                        action = _predecir(
                            modelo, obs, mask, obs_mean, obs_var)
                        carta_elegida = Carta._TODAS[action]

                        res = analizar_decision(
                            mc, idx, legales,
                            carta_elegida=carta_elegida,
                            rollout_tipo="bots",
                            rng=rng,
                        )

                        agg.n_pimc += 1
                        if res.coste > 0:
                            agg.n_errores += 1
                            agg.coste_total += res.coste
                            baza = motor.numero_baza
                            if baza <= 4:
                                agg.err_early += 1
                            elif baza <= 8:
                                agg.err_mid += 1
                            else:
                                agg.err_late += 1

                            if not motor.mesa:
                                agg.err_liderar += 1
                            elif any(
                                c.palo == motor.palo_de_salida
                                for c in legales
                            ):
                                agg.err_seguir += 1
                            else:
                                agg.err_descartar += 1

                            if res.coste >= 3:
                                agg.graves.append({
                                    "mano": h,
                                    "baza": baza,
                                    "elegida": carta_elegida.id,
                                    "optima": (
                                        res.carta_optima.id
                                        if res.carta_optima else None
                                    ),
                                    "coste": float(res.coste),
                                    "sit": (
                                        "liderar" if not motor.mesa
                                        else "seguir"
                                        if any(
                                            c.palo == motor.palo_de_salida
                                            for c in legales
                                        )
                                        else "descartar"
                                    ),
                                })
                        pimc_done = True
                    except Exception as e:
                        n_pimc_exceptions += 1

                if not pimc_done:
                    action = _predecir(
                        modelo, obs, mask, obs_mean, obs_var)
                carta = Carta._TODAS[action]
            else:
                carta = oponentes[idx](motor, idx, legales)

            motor.jugar_carta(idx, carta)

        # ── Métricas post-mano ──
        score_agente = motor.jugadores[agente_idx].contar_puntos_bazas()
        scores_mano = [
            motor.jugadores[i].contar_puntos_bazas() for i in range(4)
        ]
        q_capturada = any(
            c.es_dama_de_picas
            for c in motor.jugadores[agente_idx].bazas_ganadas
        )
        pozo_logrado = score_agente == 0 and any(s == 26 for s in scores_mano)
        pozo_bloqueado = (
            score_agente > 0
            and score_agente < 26
            and any(s == 26 for s in scores_mano)
        )

        agg.total += 1
        agg.score_total += score_agente
        agg.scores.append(float(score_agente))
        if score_agente <= 8:
            agg.victorias += 1
        if score_agente == 0:
            agg.manos_cero += 1
        if q_capturada:
            agg.q_capturadas += 1
        if pozo_logrado:
            agg.pozos_logrados += 1
        if pozo_bloqueado:
            agg.pozos_bloqueados += 1
        agg.por_seat[agente_idx].append(float(score_agente))

        print(f"  [{nombre}] {h+1}/{num_manos} | "
              f"score={score_agente} Q♠={q_capturada} "
              f"PIMC={agg.n_errores}/{agg.n_pimc} "
              f"(try={n_pimc_attempts} exc={n_pimc_exceptions})", end="\r")

    print()
    return agg


def evaluar_bot(
    nombre: str, politica_bot, num_manos: int, seed_base: int,
) -> Agregado:
    """Evalúa bot sin PIMC."""
    agg = Agregado(nombre=nombre)

    for h in range(num_manos):
        seed = seed_base + h
        agente_idx = h % 4

        motor = MotorCorazones()
        motor.repartir()

        oponentes = _construir_oponentes_estandar(agente_idx, seed)
        politicas: Dict[int, Any] = dict(oponentes)
        politicas[agente_idx] = politica_bot

        while motor._mano_activa:
            if len(motor.mesa) == 4:
                motor.resolver_baza()
                if not motor._mano_activa:
                    break
            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)
            if not legales:
                break
            carta = politicas[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)

        score_agente = motor.jugadores[agente_idx].contar_puntos_bazas()
        scores_mano = [
            motor.jugadores[i].contar_puntos_bazas() for i in range(4)
        ]
        q_capturada = any(
            c.es_dama_de_picas
            for c in motor.jugadores[agente_idx].bazas_ganadas
        )
        pozo_logrado = score_agente == 0 and any(s == 26 for s in scores_mano)
        pozo_bloqueado = (
            score_agente > 0
            and score_agente < 26
            and any(s == 26 for s in scores_mano)
        )

        agg.total += 1
        agg.score_total += score_agente
        agg.scores.append(float(score_agente))
        if score_agente <= 8:
            agg.victorias += 1
        if score_agente == 0:
            agg.manos_cero += 1
        if q_capturada:
            agg.q_capturadas += 1
        if pozo_logrado:
            agg.pozos_logrados += 1
        if pozo_bloqueado:
            agg.pozos_bloqueados += 1
        agg.por_seat[agente_idx].append(float(score_agente))

    return agg


# ──────────────────────────────────────────────────────────────
# Reporte
# ──────────────────────────────────────────────────────────────

def reporte(resultados: List[Agregado]) -> None:
    print("\n" + "=" * 72)
    print("  ANÁLISIS COMPARATIVO")
    print("=" * 72)

    for agg in resultados:
        print(f"\n{'─'*72}")
        print(f"  {agg.nombre}")
        print(f"{'─'*72}")
        print(f"  Manos: {agg.total}  WR(≤8): {agg.wr:.1f}%  "
              f"μ={agg.avg:.2f}  σ={np.std(agg.scores):.2f}  "
              f"med={np.median(agg.scores):.1f}")
        print(f"  0pts: {agg.manos_cero} ({agg.manos_cero/agg.total*100:.1f}%)  "
              f"Q♠: {agg.q_capturadas} ({agg.tasa_q:.1f}%)  "
              f"Pozo: +{agg.pozos_logrados} -{agg.pozos_bloqueados}")

        if agg.n_pimc > 0:
            print(f"  PIMC({agg.n_pimc}): err={agg.n_errores}({agg.tasa_err:.1f}%) "
                  f"coste={agg.coste_total:.2f} "
                  f"μ/err={agg.coste_medio:.2f} μ/dec={agg.coste_por_decision:.4f}")
            print(f"  Lid:{agg.err_liderar} Seg:{agg.err_seguir} "
                  f"Des:{agg.err_descartar} | "
                  f"E:{agg.err_early} M:{agg.err_mid} L:{agg.err_late}")
            if agg.graves:
                print(f"  Graves(c≥3): {len(agg.graves)}")
                for g in agg.graves[:3]:
                    print(f"    m={g['mano']} b={g['baza']} "
                          f"e={g['elegida']} o={g['optima']} "
                          f"c={g['coste']:.1f} [{g['sit']}]")

    if len(resultados) >= 2:
        print(f"\n{'─'*72}")
        print("  COMPARATIVA")
        print(f"{'─'*72}")
        for i, a in enumerate(resultados):
            for j, b in enumerate(resultados):
                if i >= j:
                    continue
                dw = a.wr - b.wr
                ds = a.avg - b.avg
                dq = a.tasa_q - b.tasa_q
                w = a.nombre if dw > 0 else b.nombre if dw < 0 else "empate"
                print(f"  {a.nombre} vs {b.nombre}: "
                      f"ΔWR={dw:+.1f}% Δμ={ds:+.2f} ΔQ♠={dq:+.1f}% → {w}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  ANÁLISIS: v3_mcts 400K vs v3_base 3.4M vs BotExperto")
    print(
        f"  {NUM_MANOS} manos | PIMC≥b{BAZA_MIN_PIMC} | ≤{MAX_MUNDOS_PIMC//1000}K mundos")
    print("=" * 72)

    resultados: List[Agregado] = []

    for nombre, ruta in MODELOS:
        print(f"\n{'='*72}")
        if nombre == "BotExperto":
            print("  Evaluando: BotExperto")
            from src.agentes.bot_experto import BotExperto
            t0 = time.time()
            agg = evaluar_bot(nombre, BotExperto(), NUM_MANOS, SEED_BASE)
        else:
            print(f"  Cargando {nombre}...")
            modelo, om, ov, obs_dim = cargar_modelo_y_vecnorm(ruta, DEVICE)
            print(f"  Evaluando...")
            t0 = time.time()
            agg = evaluar_modelo(nombre, modelo, om, ov,
                                 obs_dim, NUM_MANOS, SEED_BASE)
        print(f"  {time.time()-t0:.0f}s")
        resultados.append(agg)

    # JSON
    output = {
        "config": {
            "num_manos": NUM_MANOS,
            "baza_min_pimc": BAZA_MIN_PIMC,
            "max_mundos_pimc": MAX_MUNDOS_PIMC,
        },
        "resultados": [],
    }
    for agg in resultados:
        output["resultados"].append({
            "nombre": agg.nombre,
            "wr": agg.wr,
            "avg_score": agg.avg,
            "score_median": float(np.median(agg.scores)),
            "score_std": float(np.std(agg.scores)),
            "tasa_q_s": agg.tasa_q,
            "manos_cero": agg.manos_cero,
            "pozos_logrados": agg.pozos_logrados,
            "pozos_bloqueados": agg.pozos_bloqueados,
            "n_pimc": agg.n_pimc,
            "n_errores": agg.n_errores,
            "tasa_err_pimc": agg.tasa_err,
            "coste_total": agg.coste_total,
            "coste_medio_err": agg.coste_medio,
            "coste_por_dec": agg.coste_por_decision,
            "err_liderar": agg.err_liderar,
            "err_seguir": agg.err_seguir,
            "err_descartar": agg.err_descartar,
            "err_early": agg.err_early,
            "err_mid": agg.err_mid,
            "err_late": agg.err_late,
            "graves": agg.graves[:10],
        })
    os.makedirs("torneos", exist_ok=True)
    json_path = "torneos/20260621_analisis_comparativo.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {json_path}")

    reporte(resultados)


if __name__ == "__main__":
    main()
