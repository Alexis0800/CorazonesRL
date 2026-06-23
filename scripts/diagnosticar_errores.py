#!/usr/bin/env python
"""
Diagnóstico de errores: compara decisiones del modelo vs oráculo MCTS (PIMC sampling).

Por cada mano:
  1. El modelo decide cada jugada (determinista).
  2. PIMC con sampling evalúa todas las cartas legales.
  3. Se registra si el modelo eligió subóptimo y por cuánto margen.
  4. Se clasifica el error en patrones.

Outputs:
  - Tabla de errores por categoría y baza
  - Score delta promedio (cuántos puntos extra toma el modelo vs oráculo ideal)
  - Distribución de posiciones del modelo vs oráculo

Uso:
  python scripts/diagnosticar_errores.py --modelo models/v3_mcts/snapshots/snapshot_0000800000 --manos 200
  python scripts/diagnosticar_errores.py --modelo models/v3_mcts/snapshots/snapshot_0000800000 --manos 200 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root in path
_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proyecto not in sys.path:
    sys.path.insert(0, _proyecto)


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class ErrorInfo:
    """Información de una decisión subóptima."""
    baza: int
    carta_elegida_id: int
    carta_optima_id: int
    score_elegida: float       # score PIMC de la carta elegida
    score_optima: float         # score PIMC de la carta óptima
    delta: float                # score_elegida - score_optima (siempre >= 0)
    palo_elegido: int           # 0=♠, 1=♥, 2=♦, 3=♣
    palo_optimo: int
    es_corazon_elegido: bool
    es_corazon_optimo: bool
    es_qs_elegido: bool
    es_qs_optimo: bool
    lidera: bool                # agente lidera la baza
    num_legales: int
    riesgo_baza: float          # puntos ya acumulados en la mesa


@dataclass
class ManoResultado:
    """Resultado de una mano completa."""
    mano_id: int
    score_modelo_mano: float
    score_oraculo_mano: float   # score simulado si el oráculo decidiera
    # score acumulado de cartas óptimas (cota inferior)
    score_oraculo_teorico: float
    errores: List[ErrorInfo] = field(default_factory=list)
    decisiones: int = 0
    decisiones_suboptimas: int = 0


# ──────────────────────────────────────────────────────────────
# Categorías de error
# ──────────────────────────────────────────────────────────────

def clasificar_error(err: ErrorInfo) -> str:
    """Clasifica un error en una categoría semántica.

    Returns:
        Categoría: 'qs_castigo', 'corazon_evitable', 'descarte_malo',
        'liderazgo_malo', 'no_vaciarse', 'alimentar_qs', 'otro'
    """
    # Q♠ involucrada
    if err.es_qs_elegido:
        return "qs_castigo"
    if err.es_qs_optimo:
        return "qs_oportunidad_perdida"

    # Elegir corazón cuando había alternativa no-corazón
    if err.es_corazon_elegido and not err.es_corazon_optimo:
        return "corazon_evitable"

    # Liderar mal
    if err.lidera:
        return "liderazgo_malo"

    # No descartarse / vaciarse
    if err.num_legales >= 3 and err.delta >= 2.0:
        return "descarte_malo"

    # Errores pequeños
    if err.delta < 1.0:
        return "ruido"

    return "otro"


# ──────────────────────────────────────────────────────────────
# Diagnóstico principal
# ──────────────────────────────────────────────────────────────

def diagnosticar(
    modelo_path: str,
    num_manos: int = 200,
    seed: int = 42,
    num_mundos_pimc: int = 30,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Ejecuta diagnóstico completo comparando modelo vs oráculo PIMC.

    Args:
        modelo_path: Ruta al snapshot .zip del modelo.
        num_manos: Manos a jugar.
        seed: Semilla.
        num_mundos_pimc: Mundos de sampling PIMC por decisión.
        verbose: Si True, imprime detalles de cada error.

    Returns:
        Dict con estadísticas agregadas.
    """
    from sb3_contrib import MaskablePPO
    from src.dominio.motor import MotorCorazones
    from src.dominio.baraja import Baraja
    from src.agentes.bot_experto import BotExperto
    from src.agentes.heuristicos import BOTS_DISPONIBLES
    from src.mcts.pimc import _puntaje_esperado_por_carta
    from src.v3.entorno import CorazonesEnvV3
    from src.v3.observacion import ObservacionBuilderV3, DIM_V3

    # ── Cargar modelo ──
    if not modelo_path.endswith(".zip"):
        modelo_path += ".zip"
    print(f"Cargando modelo: {modelo_path}")
    modelo = MaskablePPO.load(modelo_path, device="cpu")
    print(f"  Política: {type(modelo.policy).__name__}")

    # ── Cargar VecNormalize si existe ──
    vecnorm = None
    vn_path = modelo_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vn_path):
        import pickle
        with open(vn_path, "rb") as f:
            vecnorm = pickle.load(f)
        print(f"  VecNormalize: cargado ({vn_path})")

    builder = ObservacionBuilderV3(dim=DIM_V3)
    rng = np.random.default_rng(seed)

    # ── Estadísticas ──
    resultados: List[ManoResultado] = []
    errores_por_categoria: Dict[str, int] = defaultdict(int)
    errores_por_baza: Dict[int, int] = defaultdict(int)
    delta_por_baza: Dict[int, List[float]] = defaultdict(list)
    delta_por_categoria: Dict[str, List[float]] = defaultdict(list)

    for mano_id in range(num_manos):
        # Crear juego nuevo con semilla determinista
        import random as _random
        _random.seed(seed + mano_id * 1000)
        np.random.seed(seed + mano_id * 1000 + 1)

        agente_idx = mano_id % 4

        # Oponentes: mezcla de bots
        politicas: Dict[int, object] = {}
        bots_pool = [BotExperto()] + [b for b in BOTS_DISPONIBLES]
        for offset in (1, 2, 3):
            rival_idx = (agente_idx + offset) % 4
            politicas[rival_idx] = bots_pool[(
                mano_id * 3 + offset) % len(bots_pool)]

        env = CorazonesEnvV3(
            agente_idx=agente_idx,
            politicas_oponentes=politicas,
        )
        obs, _ = env.reset(seed=seed + mano_id * 1000)

        resultado = ManoResultado(mano_id=mano_id, score_modelo_mano=0.0,
                                  score_oraculo_mano=0.0, score_oraculo_teorico=0.0)
        score_oraculo_acum = 0.0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            # ── Decisión del modelo ──
            mask = env.action_masks()
            legales_ids = [i for i, m in enumerate(mask) if m]
            legales = [c for c in env.motor.jugadores[agente_idx].mano
                       if c.id in legales_ids]

            if len(legales) <= 1:
                # Sin decisión que analizar
                action = legales[0].id if legales else 0
                obs, reward, terminated, truncated, info = env.step(action)
                resultado.decisiones += 1
                continue

            # Decisión del modelo (determinista)
            obs_norm = obs  # env ya normaliza si tiene VecNormalize interno
            action, _ = modelo.predict(
                obs_norm, action_masks=mask, deterministic=True)
            carta_elegida = next(c for c in legales if c.id == int(action))

            # ── PIMC oracle: evaluar todas las cartas legales ──
            try:
                scores_pimc = _puntaje_esperado_por_carta(
                    env.motor, agente_idx, legales,
                    num_mundos=num_mundos_pimc,
                    rng=np.random.default_rng(
                        seed + mano_id * 10000 + env.motor.numero_baza),
                )
                carta_optima = min(legales, key=lambda c: scores_pimc[c.id])
                score_optima = scores_pimc[carta_optima.id]
                score_elegida = scores_pimc[carta_elegida.id]
                delta = score_elegida - score_optima
            except Exception:
                carta_optima = carta_elegida
                score_optima = score_elegida = 0.0
                delta = 0.0

            score_oraculo_acum += score_optima

            # ── Registrar error si hay ──
            if carta_elegida.id != carta_optima.id:
                riesgo = sum(
                    c.puntos for _, c in env.motor.mesa
                    if c is not None and c.puntos > 0
                )
                err = ErrorInfo(
                    baza=env.motor.numero_baza,
                    carta_elegida_id=carta_elegida.id,
                    carta_optima_id=carta_optima.id,
                    score_elegida=score_elegida,
                    score_optima=score_optima,
                    delta=delta,
                    palo_elegido=carta_elegida.palo,
                    palo_optimo=carta_optima.palo,
                    es_corazon_elegido=carta_elegida.es_corazon,
                    es_corazon_optimo=carta_optima.es_corazon,
                    es_qs_elegido=carta_elegida.es_dama_de_picas,
                    es_qs_optimo=carta_optima.es_dama_de_picas,
                    lidera=(len(env.motor.mesa) == 0 or all(
                        c is None for _, c in env.motor.mesa)),
                    num_legales=len(legales),
                    riesgo_baza=riesgo,
                )
                resultado.errores.append(err)
                resultado.decisiones_suboptimas += 1

                cat = clasificar_error(err)
                errores_por_categoria[cat] += 1
                errores_por_baza[err.baza] += 1
                delta_por_baza[err.baza].append(delta)
                delta_por_categoria[cat].append(delta)

                if verbose:
                    print(
                        f"  M{mano_id:03d} B{err.baza:02d} | "
                        f"Eligió={carta_elegida} ({score_elegida:.1f}pts) | "
                        f"Óptimo={carta_optima} ({score_optima:.1f}pts) | "
                        f"Δ={delta:+.1f} | {cat} | legales={err.num_legales}"
                    )

            # Jugar la carta del modelo (NO la óptima)
            obs, reward, terminated, truncated, info = env.step(int(action))
            resultado.decisiones += 1

        # ── Fin de la mano ──
        puntos = env.motor.calcular_puntuacion_mano()
        resultado.score_modelo_mano = puntos[agente_idx]
        resultados.append(resultado)
        env.close()

        if (mano_id + 1) % 50 == 0:
            print(f"  Progreso: {mano_id + 1}/{num_manos} manos...")

    # ── Agregar estadísticas ──
    return _generar_reporte(resultados, errores_por_categoria, errores_por_baza,
                            delta_por_baza, delta_por_categoria)


def _generar_reporte(
    resultados: List[ManoResultado],
    errores_por_categoria: Dict[str, int],
    errores_por_baza: Dict[int, int],
    delta_por_baza: Dict[int, List[float]],
    delta_por_categoria: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Genera reporte detallado."""
    total_decisiones = sum(r.decisiones for r in resultados)
    total_errores = sum(r.decisiones_suboptimas for r in resultados)
    tasa_error = total_errores / total_decisiones if total_decisiones > 0 else 0

    scores_modelo = [r.score_modelo_mano for r in resultados]
    score_promedio = np.mean(scores_modelo)
    score_std = np.std(scores_modelo)

    # Posiciones
    posiciones = [0, 0, 0, 0]
    # Nota: no tenemos posiciones reales porque no simulamos con oráculo,
    # pero podemos estimar basado en score
    for s in scores_modelo:
        if s <= 2:
            posiciones[0] += 1
        elif s <= 6:
            posiciones[1] += 1
        elif s <= 10:
            posiciones[2] += 1
        else:
            posiciones[3] += 1

    # Delta promedio
    all_deltas = [e.delta for r in resultados for e in r.errores]
    delta_promedio = np.mean(all_deltas) if all_deltas else 0.0
    delta_total = sum(all_deltas)

    # ── Imprimir reporte ──
    print("\n" + "=" * 70)
    print("📊 DIAGNÓSTICO DE ERRORES — Modelo vs Oráculo PIMC")
    print("=" * 70)
    print(f"  Manos evaluadas:        {len(resultados)}")
    print(f"  Decisiones totales:     {total_decisiones}")
    print(f"  Decisiones subóptimas:  {total_errores} ({tasa_error:.1%})")
    print(f"  Score promedio modelo:  {score_promedio:.2f} ± {score_std:.2f}")
    print(f"  Delta total estimado:   {delta_total:.1f} pts "
          f"(~{delta_total / len(resultados):.1f} pts/mano)")
    print(f"  Delta promedio/error:   {delta_promedio:.2f} pts")

    # Errores por categoría
    print(f"\n{'─'*70}")
    print("📂 ERRORES POR CATEGORÍA")
    print(f"{'Categoría':<30} {'Count':>6} {'%':>6} {'Δ medio':>8}")
    print(f"{'─'*30} {'─'*6} {'─'*6} {'─'*8}")
    for cat in sorted(errores_por_categoria.keys(),
                      key=lambda c: -errores_por_categoria[c]):
        count = errores_por_categoria[cat]
        pct = count / total_errores * 100 if total_errores > 0 else 0
        avg_d = np.mean(
            delta_por_categoria[cat]) if delta_por_categoria[cat] else 0
        print(f"  {cat:<28} {count:>6} {pct:>5.1f}% {avg_d:>7.2f}")

    # Errores por baza
    print(f"\n{'─'*70}")
    print("📂 ERRORES POR BAZA")
    print(f"{'Baza':<6} {'Errores':>8} {'%':>6} {'Δ medio':>8}")
    print(f"{'─'*6} {'─'*8} {'─'*6} {'─'*8}")
    for baza in sorted(errores_por_baza.keys()):
        count = errores_por_baza[baza]
        pct = count / total_errores * 100 if total_errores > 0 else 0
        avg_d = np.mean(delta_por_baza[baza]) if delta_por_baza[baza] else 0
        bar = "█" * int(count / max(errores_por_baza.values())
                        * 20) if errores_por_baza else ""
        print(f"  {baza:<5} {count:>8} {pct:>5.1f}% {avg_d:>7.2f}  {bar}")

    # Posiciones estimadas
    print(f"\n{'─'*70}")
    print("📂 DISTRIBUCIÓN DE SCORES (estimado)")
    total_m = len(scores_modelo)
    print(f"  0-2 pts  (1°): {posiciones[0]:>4} ({posiciones[0]/total_m:.0%})")
    print(f"  3-6 pts  (2°): {posiciones[1]:>4} ({posiciones[1]/total_m:.0%})")
    print(f"  7-10 pts (3°): {posiciones[2]:>4} ({posiciones[2]/total_m:.0%})")
    print(f"  11+ pts  (4°): {posiciones[3]:>4} ({posiciones[3]/total_m:.0%})")

    # Recomendaciones
    print(f"\n{'─'*70}")
    print("💡 RECOMENDACIONES")
    print(f"{'─'*70}")

    if errores_por_categoria.get("qs_castigo", 0) > total_errores * 0.05:
        print("  ⚠️  Alta tasa de errores con Q♠ → AUMENTAR penalización Q♠")
        print("      Sugerencia: REWARD_QS_PREVENTIVO -8 → -15")

    if errores_por_categoria.get("corazon_evitable", 0) > total_errores * 0.10:
        print("  ⚠️  Muchos corazones evitables → Reforzar evitación de corazones")
        print("      Sugerencia: Añadir REWARD_CORAZON_EVITADO +2")

    if errores_por_categoria.get("liderazgo_malo", 0) > total_errores * 0.05:
        print("  ⚠️  Mal liderazgo → El modelo no sabe iniciar bazas")
        print("      Sugerencia: Añadir feature 'mejor_carta_liderazgo' al observation")

    if errores_por_categoria.get("descarte_malo", 0) > total_errores * 0.10:
        print("  ⚠️  Mal descarte → No se vacía correctamente de palos")
        print("      Sugerencia: Añadir REWARD_VACIARSE_PALO +3")

    if tasa_error > 0.40:
        print(
            f"  ⚠️  Tasa de error alta ({tasa_error:.0%}) → El modelo diverge del oráculo")
        print("      Sugerencia: Aumentar MCTS_FREQUENCY (0.40 → 0.60)")
        print("      Sugerencia: Reducir BC reg lr (1e-5 → 1e-6) o eliminar BC reg")

    if score_promedio > 7.5:
        print(
            f"  ⚠️  Score promedio alto ({score_promedio:.1f}) → El modelo toma muchos puntos")
        print("      Sugerencia: Revisar sistema de recompensas (terminal reward)")
        print("      Sugerencia: Escalar terminal reward por score (no lineal)")

    print(f"{'─'*70}")

    return {
        "manos": len(resultados),
        "total_decisiones": total_decisiones,
        "total_errores": total_errores,
        "tasa_error": tasa_error,
        "score_promedio": score_promedio,
        "score_std": score_std,
        "delta_total": delta_total,
        "delta_promedio": delta_promedio,
        "errores_por_categoria": dict(errores_por_categoria),
        "errores_por_baza": dict(errores_por_baza),
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnóstico de errores: modelo vs oráculo PIMC")
    parser.add_argument("--modelo", type=str, required=True,
                        help="Ruta al snapshot .zip del modelo")
    parser.add_argument("--manos", type=int, default=200,
                        help="Número de manos a evaluar (default: 200)")
    parser.add_argument("--mundos", type=int, default=30,
                        help="Mundos de sampling PIMC por decisión (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar detalles de cada error")

    args = parser.parse_args()

    diagnosticar(
        modelo_path=args.modelo,
        num_manos=args.manos,
        seed=args.seed,
        num_mundos_pimc=args.mundos,
        verbose=args.verbose,
    )
