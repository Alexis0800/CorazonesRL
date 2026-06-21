#!/usr/bin/env python
"""
Diagnóstico exhaustivo de decisiones del modelo usando PIMC como oráculo.

Para cada decisión del modelo (en bazas donde PIMC puede calcular el ground truth),
compara la carta elegida con la óptima según enumeración completa de mundos.
Categoriza los errores por situación, baza, tipo de carta y magnitud.

También ejecuta el mismo análisis sobre BotExperto como baseline, para distinguir
errores "inevitables" (PIMC ve el futuro) de errores "evitables" (mala heurística).

Uso:
    python scripts/diagnosticar_modelo.py \
        --modelo models/v2_1/snapshots/snapshot_0002600000.zip \
        --manos 100 --baza-min 8
"""

from __future__ import annotations
from src.mcts.analisis import (
    analizar_decision,
    pimc_exacto,
    ResultadoDecision,
    _num_mundos_posibles,
)
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
import numpy as np

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Asegurar path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# Asegurar que src/ está en el path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ──────────────────────────────────────────────────────────────
# Tipos para agregación de errores
# ──────────────────────────────────────────────────────────────

@dataclass
class ErrorCategoria:
    """Agregación de errores por categoría."""
    nombre: str
    count: int = 0
    coste_total: float = 0.0
    coste_max: float = 0.0


@dataclass
class Diagnostico:
    """Resultado completo del diagnóstico."""
    nombre_politica: str
    num_manos: int
    num_decisiones_analizadas: int = 0
    num_decisiones_exactas: int = 0
    num_errores: int = 0  # decisiones con coste > 0
    coste_total: float = 0.0
    coste_medio_por_error: float = 0.0
    coste_medio_por_decision: float = 0.0
    puntuacion_media: float = 0.0
    puntuacion_std: float = 0.0
    tasa_cero: float = 0.0
    tasa_q: float = 0.0

    # Desglose por situación
    errores_liderar: ErrorCategoria = field(
        default_factory=lambda: ErrorCategoria("liderar"))
    errores_seguir: ErrorCategoria = field(
        default_factory=lambda: ErrorCategoria("seguir"))
    errores_descartar: ErrorCategoria = field(
        default_factory=lambda: ErrorCategoria("descartar"))

    # Desglose por baza (early 1-4, mid 5-8, late 9-13)
    errores_early: ErrorCategoria = field(
        default_factory=lambda: ErrorCategoria("bazas 1-4"))
    errores_mid: ErrorCategoria = field(
        default_factory=lambda: ErrorCategoria("bazas 5-8"))
    errores_late: ErrorCategoria = field(
        default_factory=lambda: ErrorCategoria("bazas 9-13"))

    # Errores grandes (coste ≥ 3 pts)
    errores_graves: List[ResultadoDecision] = field(default_factory=list)

    # Puntuaciones por mano
    puntuaciones: List[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Carga de políticas
# ──────────────────────────────────────────────────────────────

def _cargar_modelo_v2(ruta: str) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray]]:
    """Carga modelo MaskablePPO + VecNormalize desde ruta .zip."""
    import pickle
    from sb3_contrib import MaskablePPO

    # Quitar extensión .zip o _vecnorm.pkl
    ruta_base = ruta
    if ruta_base.endswith("_vecnorm.pkl"):
        ruta_base = ruta_base.replace("_vecnorm.pkl", "")
    if ruta_base.endswith(".zip"):
        ruta_base = ruta_base[:-4]

    modelo = MaskablePPO.load(ruta_base, device="cpu")

    obs_mean = None
    obs_var = None
    vn_path = ruta_base + "_vecnorm.pkl"
    if os.path.exists(vn_path):
        with open(vn_path, "rb") as f:
            vn = pickle.load(f)
        obs_mean = vn.obs_rms.mean.copy()
        obs_var = vn.obs_rms.var.copy()

    return modelo, obs_mean, obs_var


def _elegir_con_modelo(
    modelo, obs_mean, obs_var,
    motor: MotorCorazones,
    idx: int,
    legales: List[Carta],
) -> Carta:
    """Elige carta usando el modelo v2_1."""
    from src.entorno.observacion import ObservacionBuilder
    from src.entorno.dimensiones import DIM_ENTRENAMIENTO

    builder = ObservacionBuilder(dim=DIM_ENTRENAMIENTO)
    obs_raw = builder.construir(
        motor, idx,
        vacios=[set() for _ in range(4)],
        puntuacion_historica=[0, 0, 0, 0],
        puntos_mano_actual=[j.contar_puntos_bazas() for j in motor.jugadores],
        dama_picas_en=None,
    )

    if obs_mean is not None:
        obs = np.clip(
            (obs_raw - obs_mean) / (np.sqrt(obs_var) + 1e-8), -10, 10,
        )
    else:
        obs = obs_raw

    mask = np.zeros(52, dtype=bool)
    for c in legales:
        mask[c.id] = True

    action, _ = modelo.predict(obs, action_masks=mask, deterministic=True)
    return Carta._TODAS[int(action)]


# ──────────────────────────────────────────────────────────────
# Diagnóstico de una política
# ──────────────────────────────────────────────────────────────

def diagnosticar_politica(
    politica: Callable[[MotorCorazones, int, List[Carta]], Carta],
    nombre: str,
    num_manos: int = 100,
    baza_min: int = 8,
    seed_inicial: int = 42,
    rollout_tipo: str = "experto",
    verbose: bool = True,
) -> Diagnostico:
    """Evalúa una política decisión por decisión usando PIMC como oráculo.

    Args:
        politica: Callable(motor, idx, legales) -> Carta
        nombre: Etiqueta para reportes.
        num_manos: Manos a evaluar.
        baza_min: Baza mínima para análisis exacto (default 8).
        seed_inicial: Semilla base.
        rollout_tipo: Política de rollout para PIMC.
        verbose: Mostrar progreso.

    Returns:
        Diagnostico con todas las métricas agregadas.
    """
    from src.agentes.bot_experto import BotExperto

    diag = Diagnostico(nombre_politica=nombre, num_manos=num_manos)
    rng = np.random.default_rng(seed_inicial)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  DIAGNÓSTICO: {nombre}")
        print(
            f"  Manos: {num_manos} | Baza mín: {baza_min} | Rollout: {rollout_tipo}")
        print(f"{'='*60}")

    for mano_idx in range(num_manos):
        seed = seed_inicial + mano_idx
        np.random.seed(seed)
        import random
        random.seed(seed)

        # Oponentes: BotExperto
        oponentes = [BotExperto() for _ in range(4)]

        motor = MotorCorazones()
        motor.repartir()

        while motor._mano_activa:
            if len(motor.mesa) == 4:
                motor.resolver_baza()
                if not motor._mano_activa:
                    break

            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)
            if not legales:
                break

            if idx == 0:
                # ── Turno del agente ──
                carta_elegida = politica(motor, 0, legales)

                # Analizar si hay ≥2 opciones y baza ≥ baza_min
                if len(legales) >= 2 and motor.numero_baza >= baza_min:
                    n_mundos = _num_mundos_posibles(motor, 0)
                    if n_mundos <= 100_000:  # viable para exacto
                        resultado = analizar_decision(
                            motor, 0, legales, carta_elegida,
                            vacios={},
                            rollout_tipo=rollout_tipo,
                            rng=rng,
                        )
                        diag.num_decisiones_analizadas += 1
                        if resultado.exacto:
                            diag.num_decisiones_exactas += 1

                        if resultado.coste > 0:
                            diag.num_errores += 1
                            diag.coste_total += resultado.coste

                            # Categorizar por situación
                            if resultado.situacion == "liderar":
                                diag.errores_liderar.count += 1
                                diag.errores_liderar.coste_total += resultado.coste
                                diag.errores_liderar.coste_max = max(
                                    diag.errores_liderar.coste_max, resultado.coste)
                            elif resultado.situacion == "seguir":
                                diag.errores_seguir.count += 1
                                diag.errores_seguir.coste_total += resultado.coste
                                diag.errores_seguir.coste_max = max(
                                    diag.errores_seguir.coste_max, resultado.coste)
                            else:
                                diag.errores_descartar.count += 1
                                diag.errores_descartar.coste_total += resultado.coste
                                diag.errores_descartar.coste_max = max(
                                    diag.errores_descartar.coste_max, resultado.coste)

                            # Categorizar por fase
                            baza = resultado.baza
                            if baza <= 4:
                                diag.errores_early.count += 1
                                diag.errores_early.coste_total += resultado.coste
                            elif baza <= 8:
                                diag.errores_mid.count += 1
                                diag.errores_mid.coste_total += resultado.coste
                            else:
                                diag.errores_late.count += 1
                                diag.errores_late.coste_total += resultado.coste

                            # Errores graves (≥3 pts)
                            if resultado.coste >= 3.0:
                                diag.errores_graves.append(resultado)

                motor.jugar_carta(0, carta_elegida)
            else:
                # ── Turno de oponente ──
                carta = oponentes[idx](motor, idx, legales)
                motor.jugar_carta(idx, carta)

        # Puntuación final de la mano
        mi_score = motor.jugadores[0].contar_puntos_bazas()
        diag.puntuaciones.append(mi_score)

        if verbose and (mano_idx + 1) % max(1, num_manos // 10) == 0:
            n_err = diag.num_errores
            coste = diag.coste_total
            media = np.mean(diag.puntuaciones)
            print(f"  Mano {mano_idx+1:4d}/{num_manos} | "
                  f"Score avg: {media:.1f} | Errores: {n_err} | "
                  f"Coste: {coste:.1f} pts")

    # Calcular métricas finales
    arr = np.array(diag.puntuaciones, dtype=np.float64)
    diag.puntuacion_media = float(np.mean(arr))
    diag.puntuacion_std = float(np.std(arr))
    diag.tasa_cero = float(np.mean(arr == 0))
    diag.tasa_q = float(
        np.mean([1 for s in diag.puntuaciones if s >= 13 and s < 26]))

    if diag.num_errores > 0:
        diag.coste_medio_por_error = diag.coste_total / diag.num_errores
    if diag.num_decisiones_analizadas > 0:
        diag.coste_medio_por_decision = diag.coste_total / diag.num_decisiones_analizadas

    return diag


# ──────────────────────────────────────────────────────────────
# Reporte
# ──────────────────────────────────────────────────────────────

def imprimir_reporte(diag: Diagnostico) -> None:
    """Imprime reporte formateado del diagnóstico."""
    print(f"\n{'='*60}")
    print(f"  REPORTE: {diag.nombre_politica}")
    print(f"{'='*60}")

    print(f"\n  📊 Métricas globales ({diag.num_manos} manos):")
    print(
        f"    Puntuación media : {diag.puntuacion_media:.2f} ± {diag.puntuacion_std:.2f}")
    print(f"    Tasa 0 puntos    : {diag.tasa_cero*100:.1f}%")
    print(f"    Tasa captura Q♠  : {diag.tasa_q*100:.1f}%")

    print(f"\n  🔍 Decisiones analizadas (baza ≥ 8, enumeración exacta):")
    print(f"    Total analizadas : {diag.num_decisiones_analizadas}")
    print(f"    Exactas (enum)   : {diag.num_decisiones_exactas}")
    print(f"    Con error         : {diag.num_errores} "
          f"({diag.num_errores/max(1, diag.num_decisiones_analizadas)*100:.1f}%)")
    print(f"    Coste total      : {diag.coste_total:.2f} pts")
    print(f"    Coste medio/error: {diag.coste_medio_por_error:.2f} pts")
    print(f"    Coste medio/dec  : {diag.coste_medio_por_decision:.4f} pts")

    print(f"\n  🎯 Errores por situación:")
    for cat in [diag.errores_liderar, diag.errores_seguir, diag.errores_descartar]:
        if cat.count > 0:
            print(f"    {cat.nombre:12s}: {cat.count:3d} errores | "
                  f"coste total={cat.coste_total:.1f} | "
                  f"coste max={cat.coste_max:.1f} | "
                  f"media={cat.coste_total/cat.count:.2f}")

    print(f"\n  📅 Errores por fase:")
    for cat in [diag.errores_early, diag.errores_mid, diag.errores_late]:
        if cat.count > 0:
            print(f"    {cat.nombre:12s}: {cat.count:3d} errores | "
                  f"coste total={cat.coste_total:.1f} | "
                  f"media={cat.coste_total/max(1, cat.count):.2f}")

    if diag.errores_graves:
        print(
            f"\n  💀 Errores graves (coste ≥ 3 pts): {len(diag.errores_graves)}")
        # Mostrar los 5 peores
        peores = sorted(diag.errores_graves,
                        key=lambda d: d.coste, reverse=True)[:5]
        for d in peores:
            print(f"    Baza {d.baza} | {d.situacion} | "
                  f"Eligió {d.carta_elegida} ({d.score_elegido} pts) | "
                  f"Óptima {d.carta_optima} ({d.score_optimo} pts) | "
                  f"Coste: {d.coste:+.1f} | Modo: {d.modo_bot}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnóstico de modelo usando PIMC como oráculo",
    )
    parser.add_argument("--modelo", type=str, required=True,
                        help="Ruta al snapshot .zip del modelo v2_1.")
    parser.add_argument("--manos", type=int, default=100,
                        help="Número de manos a evaluar (default 100).")
    parser.add_argument("--baza-min", type=int, default=8,
                        help="Baza mínima para análisis exacto (default 8).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout", type=str, default="experto",
                        choices=["evasivo", "experto", "mixto"])
    parser.add_argument("--baseline", action="store_true",
                        help="También diagnosticar BotExperto como baseline.")
    parser.add_argument("--output", type=str, default=None,
                        help="Guardar resultados como JSON.")

    args = parser.parse_args()

    # ── Cargar modelo ──
    print(f"Cargando modelo: {args.modelo}")
    modelo, obs_mean, obs_var = _cargar_modelo_v2(args.modelo)

    def politica_modelo(motor, idx, legales):
        return _elegir_con_modelo(modelo, obs_mean, obs_var, motor, idx, legales)

    # ── Diagnosticar modelo ──
    t0 = time.time()
    diag_modelo = diagnosticar_politica(
        politica=politica_modelo,
        nombre=f"v2_1 ({os.path.basename(args.modelo)})",
        num_manos=args.manos,
        baza_min=args.baza_min,
        seed_inicial=args.seed,
        rollout_tipo=args.rollout,
    )
    elapsed = time.time() - t0
    print(f"\n  ⏱ Tiempo: {elapsed:.1f}s")

    imprimir_reporte(diag_modelo)

    # ── Baseline BotExperto (opcional) ──
    diag_experto = None
    if args.baseline:
        from src.agentes.bot_experto import BotExperto
        experto = BotExperto()

        def politica_experto(motor, idx, legales):
            return experto(motor, idx, legales)

        print(f"\n{'─'*60}")
        print("  Ejecutando baseline BotExperto...")
        t0 = time.time()
        diag_experto = diagnosticar_politica(
            politica=politica_experto,
            nombre="BotExperto (baseline)",
            num_manos=args.manos,
            baza_min=args.baza_min,
            seed_inicial=args.seed,
            rollout_tipo=args.rollout,
        )
        elapsed = time.time() - t0
        print(f"\n  ⏱ Tiempo: {elapsed:.1f}s")
        imprimir_reporte(diag_experto)

    # ── Comparativa ──
    if diag_experto:
        print(f"\n{'='*60}")
        print(f"  COMPARATIVA MODELO vs BOTEXPERTO")
        print(f"{'='*60}")
        print(f"  {'Métrica':30s} {'v2_1':>10s} {'Experto':>10s} {'Δ':>10s}")
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'Puntuación media':30s} {diag_modelo.puntuacion_media:10.2f} "
              f"{diag_experto.puntuacion_media:10.2f} "
              f"{diag_modelo.puntuacion_media - diag_experto.puntuacion_media:+10.2f}")
        print(f"  {'Tasa 0 puntos':30s} {diag_modelo.tasa_cero*100:9.1f}% "
              f"{diag_experto.tasa_cero*100:9.1f}% "
              f"{(diag_modelo.tasa_cero - diag_experto.tasa_cero)*100:+10.1f}%")
        print(f"  {'Errores/decision':30s} {diag_modelo.num_errores/max(1, diag_modelo.num_decisiones_analizadas)*100:9.1f}% "
              f"{diag_experto.num_errores/max(1, diag_experto.num_decisiones_analizadas)*100:9.1f}% "
              f"{(diag_modelo.num_errores/max(1, diag_modelo.num_decisiones_analizadas) - diag_experto.num_errores/max(1, diag_experto.num_decisiones_analizadas))*100:+10.1f}%")
        print(f"  {'Coste medio/dec':30s} {diag_modelo.coste_medio_por_decision:10.4f} "
              f"{diag_experto.coste_medio_por_decision:10.4f} "
              f"{diag_modelo.coste_medio_por_decision - diag_experto.coste_medio_por_decision:+10.4f}")

    # ── Guardar ──
    if args.output:
        resultados = {
            "modelo": {
                "nombre": diag_modelo.nombre_politica,
                "num_manos": diag_modelo.num_manos,
                "puntuacion_media": diag_modelo.puntuacion_media,
                "puntuacion_std": diag_modelo.puntuacion_std,
                "tasa_cero": diag_modelo.tasa_cero,
                "num_decisiones": diag_modelo.num_decisiones_analizadas,
                "num_exactas": diag_modelo.num_decisiones_exactas,
                "num_errores": diag_modelo.num_errores,
                "coste_total": diag_modelo.coste_total,
                "coste_medio_error": diag_modelo.coste_medio_por_error,
                "errores_liderar": diag_modelo.errores_liderar.count,
                "errores_seguir": diag_modelo.errores_seguir.count,
                "errores_descartar": diag_modelo.errores_descartar.count,
                "errores_early": diag_modelo.errores_early.count,
                "errores_mid": diag_modelo.errores_mid.count,
                "errores_late": diag_modelo.errores_late.count,
                "errores_graves": len(diag_modelo.errores_graves),
            },
        }
        if diag_experto:
            resultados["experto"] = {
                "nombre": diag_experto.nombre_politica,
                "puntuacion_media": diag_experto.puntuacion_media,
                "tasa_cero": diag_experto.tasa_cero,
                "num_errores": diag_experto.num_errores,
                "coste_total": diag_experto.coste_total,
            }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
        print(f"\n  Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
