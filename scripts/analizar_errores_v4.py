#!/usr/bin/env python
"""
Análisis de errores del modelo v4 usando PIMC con enumeración completa.

Para cada decisión no trivial (≥2 opciones legales, baza ≥ 10 para PIMC exacto),
compara la elección del modelo RL contra la jugada óptima según PIMC.
Genera un reporte Markdown con:
  - Tipos de error y su frecuencia
  - Coste acumulado por tipo de error
  - Etapas de la mano donde ocurren más errores
  - Si los errores eran evitables (existía mejor jugada legal distinta)

Uso:
    python scripts/analizar_errores_v4.py --modelo models/v4/snapshots/snapshot_0008100000 --manos 100
"""

from __future__ import annotations
from sb3_contrib import MaskablePPO
from src.mcts.pimc import _puntaje_esperado_por_carta, crear_bots_rollout
from src.mcts.analisis import pimc_exacto, _num_mundos_posibles
from src.agentes.heuristicos import BOTS_DISPONIBLES, bot_evasivo
from src.agentes.politica_rl import PoliticaSB3
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ──────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────

_PALO = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
_VALOR = {12: "A", 13: "2", 14: "3", 15: "4", 16: "5", 17: "6",
          18: "7", 19: "8", 20: "9", 21: "10", 22: "J", 23: "Q", 24: "K"}
_BAZA_MINIMA_EXACTA = 10  # baza mínima para PIMC con enumeración completa


def _nombre_carta(c: Carta) -> str:
    v = _VALOR.get(c.valor, str(c.valor))
    return f"{v}{_PALO[c.palo]}"


@dataclass
class ErrorDetectado:
    baza: int
    situacion: str         # liderar / seguir / descartar
    cartas_legales: int
    carta_modelo: str
    carta_optima: str
    score_modelo: float    # puntaje esperado de la jugada del modelo
    score_optimo: float    # puntaje esperado de la jugada óptima
    coste: float           # score_modelo - score_optimo (>0 = modelo peor)
    delta_score: float     # diferencia entre mejor y peor opción legal
    pimc_exacto: bool      # True si se usó enumeración completa


@dataclass
class ResumenMano:
    seed: int
    puntos_agente: int
    puntos_oponentes: List[int]
    errores: List[ErrorDetectado]
    coste_total: float
    baza_qs: int           # baza donde salió Q♠ (-1 si no salió)


# ──────────────────────────────────────────────────────────────
# Análisis principal
# ──────────────────────────────────────────────────────────────


def analizar_mano(
    motor: MotorCorazones,
    agente_idx: int,
    politica,
    seed: int,
) -> ResumenMano:
    """Juega una mano completa con la política y analiza cada decisión."""
    rng = np.random.default_rng(seed)
    errores: List[ErrorDetectado] = []

    # Inicializar
    motor.repartir()
    baza_qs = -1

    for _ in range(13):  # 13 bazas
        for _ in range(4):  # 4 jugadores por baza
            jugador_actual = motor.obtener_jugador_actual()
            if jugador_actual != agente_idx:
                # ── Oponente: bot evasivo ──
                legales_op = motor.obtener_jugadas_legales(jugador_actual)
                carta_op = bot_evasivo(motor, jugador_actual, legales_op)
                motor.jugar_carta(jugador_actual, carta_op)
                continue

            # ── Turno del agente (modelo) ──
            legales = motor.obtener_jugadas_legales(agente_idx)
            if len(legales) == 0:
                break

            baza_actual = motor.numero_baza

            # Decisión del modelo
            carta_modelo = politica(
                motor, agente_idx, legales, deterministic=True)

            # ── PIMC exacto (solo si ≥2 opciones y baza suficiente) ──
            num_opciones = len(legales)

            if num_opciones >= 2 and baza_actual >= _BAZA_MINIMA_EXACTA:
                # Usar PIMC con enumeración completa
                mejor_carta, scores, exacto = pimc_exacto(
                    motor, agente_idx, legales,
                    rollout_tipo="evasivo",
                    rng=rng,
                )

                if scores and len(scores) > 0:
                    score_modelo = scores.get(carta_modelo.id, 999)
                    score_optimo = scores.get(mejor_carta.id, 0)
                    coste = score_modelo - score_optimo

                    scores_list = list(scores.values())
                    delta = max(scores_list) - \
                        min(scores_list) if scores_list else 0

                    situacion = "liderar"
                    if motor.palo_de_salida is not None:
                        if any(c.palo == motor.palo_de_salida for c in legales):
                            situacion = "seguir"
                        else:
                            situacion = "descartar"

                    if coste > 0.01:  # diferencia significativa
                        errores.append(ErrorDetectado(
                            baza=baza_actual,
                            situacion=situacion,
                            cartas_legales=num_opciones,
                            carta_modelo=_nombre_carta(carta_modelo),
                            carta_optima=_nombre_carta(mejor_carta),
                            score_modelo=round(score_modelo, 2),
                            score_optimo=round(score_optimo, 2),
                            coste=round(coste, 2),
                            delta_score=round(delta, 2),
                            pimc_exacto=exacto,
                        ))

            # Rastrear Q♠
            if carta_modelo.id == 49:  # Q♠
                baza_qs = baza_actual

            # Ejecutar jugada
            motor.jugar_carta(agente_idx, carta_modelo)

        # ── Resolver baza ──
        motor.resolver_baza()

    # Puntuación final
    puntos = motor.calcular_puntuacion_mano()
    return ResumenMano(
        seed=seed,
        puntos_agente=puntos[agente_idx],
        puntos_oponentes=[puntos[i] for i in range(4) if i != agente_idx],
        errores=errores,
        coste_total=sum(e.coste for e in errores),
        baza_qs=baza_qs,
    )


def _construir_obs_minima(motor: MotorCorazones, agente_idx: int) -> np.ndarray:
    """Construye una observación mínima (220-dims) para el modelo."""
    from src.entorno.observacion import ObservacionBuilder
    builder = ObservacionBuilder()
    return builder.construir_desde_motor(motor, agente_idx)


# ──────────────────────────────────────────────────────────────
# Reporte
# ──────────────────────────────────────────────────────────────


def generar_reporte(
    modelo_path: str,
    manos: List[ResumenMano],
    elapsed: float,
) -> str:
    """Genera reporte Markdown completo."""
    total_errores = sum(len(m.errores) for m in manos)
    coste_total = sum(m.coste_total for m in manos)
    avg_puntos = np.mean([m.puntos_agente for m in manos])
    manos_0 = sum(1 for m in manos if m.puntos_agente == 0)
    manos_qs = sum(1 for m in manos if m.baza_qs >= 0)

    # Clasificar errores
    errores_por_baza: Dict[int, List[ErrorDetectado]] = defaultdict(list)
    errores_por_situacion: Dict[str, List[ErrorDetectado]] = defaultdict(list)
    coste_por_baza: Dict[int, float] = defaultdict(float)

    for m in manos:
        for e in m.errores:
            errores_por_baza[e.baza].append(e)
            errores_por_situacion[e.situacion].append(e)
            coste_por_baza[e.baza] += e.coste

    # Errores con mayor coste
    todos_errores = []
    for m in manos:
        todos_errores.extend(m.errores)
    todos_errores.sort(key=lambda e: e.coste, reverse=True)

    # ── Construir reporte ──
    modelo_nombre = os.path.basename(modelo_path).replace(".zip", "")

    lines = []
    lines.append(f"# Análisis de Errores — v4 {modelo_nombre}")
    lines.append(f"")
    lines.append(f"**Fecha**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Modelo**: `{modelo_path}`")
    lines.append(f"**Manos analizadas**: {len(manos)}")
    lines.append(f"**Tiempo de análisis**: {elapsed:.1f}s")
    lines.append(f"**Baza mínima para PIMC exacto**: {_BAZA_MINIMA_EXACTA}")
    lines.append(f"")

    # ── Resumen ejecutivo ──
    lines.append(f"## 1. Resumen Ejecutivo")
    lines.append(f"")
    lines.append(f"| Métrica | Valor |")
    lines.append(f"|---------|-------|")
    lines.append(f"| Puntuación media | {avg_puntos:.1f} |")
    lines.append(
        f"| Manos con 0 pts | {manos_0}/{len(manos)} ({manos_0/len(manos)*100:.0f}%) |")
    lines.append(
        f"| Manos donde capturó Q♠ | {manos_qs}/{len(manos)} ({manos_qs/len(manos)*100:.0f}%) |")
    lines.append(f"| **Total errores detectados** | **{total_errores}** |")
    lines.append(f"| **Coste total acumulado** | **{coste_total:.1f} pts** |")
    lines.append(
        f"| Coste medio por mano | {coste_total/len(manos):.1f} pts |")
    lines.append(
        f"| Coste medio por error | {coste_total/total_errores if total_errores > 0 else 0:.2f} pts |")
    lines.append(f"")

    # ── Distribución por baza ──
    lines.append(f"## 2. Errores por Etapa de la Mano")
    lines.append(f"")
    lines.append(
        f"| Baza | Errores | Coste Total | Coste Medio | ¿Evitable? |")
    lines.append(
        f"|------|---------|-------------|-------------|------------|")
    for baza in sorted(errores_por_baza.keys()):
        errs = errores_por_baza[baza]
        c_total = coste_por_baza[baza]
        c_medio = c_total / len(errs)
        # En bazas 10+ usamos PIMC exacto, antes es sampling
        exacto = baza >= _BAZA_MINIMA_EXACTA
        evitable = "✅ Sí (PIMC exacto)" if exacto else "⚠️ Aprox (sampling)"
        lines.append(
            f"| {baza} | {len(errs)} | {c_total:.1f} | {c_medio:.2f} | {evitable} |")
    lines.append(f"")

    # ── Distribución por situación ──
    lines.append(f"## 3. Errores por Situación de Juego")
    lines.append(f"")
    lines.append(f"| Situación | Errores | Coste Total | Coste Medio |")
    lines.append(f"|-----------|---------|-------------|-------------|")
    for sit in ["liderar", "seguir", "descartar"]:
        errs = errores_por_situacion.get(sit, [])
        c_total = sum(e.coste for e in errs)
        c_medio = c_total / len(errs) if errs else 0
        lines.append(
            f"| {sit} | {len(errs)} | {c_total:.1f} | {c_medio:.2f} |")
    lines.append(f"")

    # ── Top 10 errores más costosos ──
    lines.append(f"## 4. Top 10 Errores Más Costosos")
    lines.append(f"")
    lines.append(
        f"| # | Baza | Situación | Modelo eligió | Óptimo PIMC | Coste | ΔScore |")
    lines.append(
        f"|---|------|-----------|---------------|-------------|-------|--------|")
    for i, e in enumerate(todos_errores[:10]):
        lines.append(
            f"| {i+1} | {e.baza} | {e.situacion} | {e.carta_modelo} | {e.carta_optima} | {e.coste:.1f} | {e.delta_score:.1f} |")
    lines.append(f"")

    # ── Distribución del coste ──
    lines.append(f"## 5. Distribución del Coste de Errores")
    lines.append(f"")
    if todos_errores:
        costes = [e.coste for e in todos_errores]
        lines.append(f"| Percentil | Coste (pts) |")
        lines.append(f"|-----------|-------------|")
        for p in [50, 75, 90, 95, 99, 100]:
            val = np.percentile(costes, p)
            lines.append(f"| P{p} | {val:.2f} |")
        lines.append(f"")

    # ── Análisis cualitativo ──
    lines.append(f"## 6. Análisis Cualitativo")
    lines.append(f"")

    # Patrones observados
    liderar_errs = errores_por_situacion.get("liderar", [])
    seguir_errs = errores_por_situacion.get("seguir", [])
    descartar_errs = errores_por_situacion.get("descartar", [])

    lines.append(f"### Errores al Liderar ({len(liderar_errs)} casos)")
    lines.append(f"")
    if liderar_errs:
        avg_cost = sum(e.coste for e in liderar_errs) / len(liderar_errs)
        lines.append(f"- Coste medio: {avg_cost:.2f} pts")
        lines.append(
            f"- Suele ocurrir cuando el modelo **no identifica la carta óptima para iniciar la baza**.")
        lines.append(
            f"- El modelo puede estar jugando cartas demasiado altas o demasiado bajas para la fase de la mano.")
    else:
        lines.append(f"- Sin errores detectados en esta situación.")
    lines.append(f"")

    lines.append(f"### Errores al Seguir Palo ({len(seguir_errs)} casos)")
    lines.append(f"")
    if seguir_errs:
        avg_cost = sum(e.coste for e in seguir_errs) / len(seguir_errs)
        lines.append(f"- Coste medio: {avg_cost:.2f} pts")
        lines.append(
            f"- El modelo **no ajusta bien la altura de la carta** al seguir el palo.")
        lines.append(
            f"- Posiblemente juega cartas muy altas cuando debería jugar bajas (o viceversa).")
    else:
        lines.append(f"- Sin errores detectados en esta situación.")
    lines.append(f"")

    lines.append(f"### Errores al Descartar ({len(descartar_errs)} casos)")
    lines.append(f"")
    if descartar_errs:
        avg_cost = sum(e.coste for e in descartar_errs) / len(descartar_errs)
        lines.append(f"- Coste medio: {avg_cost:.2f} pts")
        lines.append(
            f"- El modelo **elige mal qué palo descartar** o **tira una carta incorrecta**.")
        lines.append(
            f"- Posiblemente no está considerando correctamente los vacíos de los oponentes.")
    else:
        lines.append(f"- Sin errores detectados en esta situación.")
    lines.append(f"")

    # ── Comparación con v4 rendimiento global ──
    lines.append(f"## 7. Implicaciones para v5")
    lines.append(f"")
    lines.append(
        f"- **Coste total por mano**: {coste_total/len(manos):.1f} pts de error evitable.")
    lines.append(
        f"- Si v5 logra reducir este coste a la mitad, la puntuación media mejoraría ~{(coste_total/len(manos))/2:.1f} pts.")
    lines.append(
        f"- Las bazas críticas son: {', '.join(str(b) for b in sorted(errores_por_baza.keys(), key=lambda b: coste_por_baza[b], reverse=True)[:3])}.")
    lines.append(
        f"- v5 con self-play puro debería mostrar **menor coste en bazas tardías** (10+) donde PIMC tiene ground truth.")
    lines.append(f"")

    # ── Manos de ejemplo ──
    lines.append(f"## 8. Manos con Más Errores")
    lines.append(f"")
    manos_ordenadas = sorted(manos, key=lambda m: len(m.errores), reverse=True)
    lines.append(f"| Seed | Puntos | Q♠ | Errores | Coste Total |")
    lines.append(f"|------|--------|-----|---------|-------------|")
    for m in manos_ordenadas[:10]:
        qs_str = f"Baza {m.baza_qs}" if m.baza_qs >= 0 else "No"
        lines.append(
            f"| {m.seed} | {m.puntos_agente} | {qs_str} | {len(m.errores)} | {m.coste_total:.1f} |")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analiza errores del modelo v4 vs PIMC exacto")
    parser.add_argument("--modelo", type=str,
                        default="models/v4/snapshots/snapshot_0008100000",
                        help="Ruta al snapshot del modelo")
    parser.add_argument("--manos", type=int, default=100,
                        help="Número de manos a analizar")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla inicial")
    parser.add_argument("--output", type=str, default=None,
                        help="Archivo de salida (default: docs/analisis_v4_errores.md)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar progreso detallado")

    args = parser.parse_args()

    output_path = args.output or "docs/analisis_v4_errores.md"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ── Cargar modelo ──
    print(f"Cargando modelo: {args.modelo}")
    modelo = MaskablePPO.load(args.modelo, device="cpu")

    # Cargar VecNormalize si existe
    modelo_path_clean = args.modelo
    if modelo_path_clean.endswith('.zip'):
        modelo_path_clean = modelo_path_clean[:-4]
    vecnorm_path = f"{modelo_path_clean}_vecnorm.pkl"
    vecnorm = vecnorm_path if os.path.exists(vecnorm_path) else None
    print(f"  VecNormalize: {'Sí' if vecnorm else 'No'}")

    politica = PoliticaSB3(modelo, 0, vecnorm)

    # ── Analizar manos ──
    print(
        f"Analizando {args.manos} manos con PIMC exacto (baza ≥ {_BAZA_MINIMA_EXACTA})...")
    print()

    t0 = time.time()
    manos: List[ResumenMano] = []
    seeds_usadas: List[int] = []

    for i in range(args.manos):
        seed = args.seed + i
        seeds_usadas.append(seed)
        motor = MotorCorazones()

        res = analizar_mano(motor, i % 4, politica, seed)
        manos.append(res)

        if args.verbose or (i + 1) % 10 == 0:
            errs_so_far = sum(len(m.errores) for m in manos)
            pts_avg = np.mean([m.puntos_agente for m in manos])
            print(f"  Mano {i+1:4d}/{args.manos} | "
                  f"Media: {pts_avg:.1f} | Errores totales: {errs_so_far} | "
                  f"Seed: {seed}")

    elapsed = time.time() - t0

    # ── Generar reporte ──
    print(f"\nGenerando reporte...")
    reporte = generar_reporte(args.modelo, manos, elapsed)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(reporte)

    print(f"✅ Reporte guardado: {output_path}")
    print(f"   {len(manos)} manos analizadas en {elapsed:.1f}s")
    total_errs = sum(len(m.errores) for m in manos)
    print(f"   {total_errs} errores detectados")


if __name__ == "__main__":
    main()
