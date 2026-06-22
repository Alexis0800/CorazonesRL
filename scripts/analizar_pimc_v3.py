"""
Análisis PIMC detallado de los mejores modelos v3.

Para cada decisión del modelo, compara su elección contra la jugada
óptima calculada por PIMC exacto (enumeración cuando es viable,
sampling cuando no). Agrega resultados por situación, etapa de la
mano, y tipo de error.

Uso:
    python scripts/analizar_pimc_v3.py

El script carga los top 3 modelos v3 (4.3M, 3.4M, 3.8M) más
BotExperto como baseline, y analiza cada decisión en N manos.
"""

from __future__ import annotations
from sb3_contrib import MaskablePPO
from src.agentes.bot_experto import BotExperto
from src.v3.politica_rl import PoliticaSB3V3
from src.mcts.analisis import (
    ResultadoDecision,
    analizar_decision,
)
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Añadir src/ al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────

MODELOS_A_ANALIZAR = [
    ("v3_4.3M (#1)", "models/v3/snapshots/snapshot_0004300000"),
    ("v3_3.4M (#2)", "models/v3/snapshots/snapshot_0003400000"),
    ("v3_3.8M (#3)", "models/v3/snapshots/snapshot_0003800000"),
    ("v3_1.1M (#4)", "models/v3/snapshots/snapshot_0001100000"),
    ("v3_0.1M",     "models/v3/snapshots/snapshot_0000100000"),
]

NUM_MANOS: int = 20          # manos por modelo
SEED_BASE: int = 42
ROLLOUT_TIPO: str = "experto"  # política de rollout para PIMC
DEVICE: str = "cpu"            # "cpu", "cuda", "dml"


# ──────────────────────────────────────────────────────────────
# Carga de modelos
# ──────────────────────────────────────────────────────────────

def cargar_modelo_v3(ruta: str, device: str = "cpu"):
    """Carga MaskablePPO v3 + VecNormalize desde un snapshot."""
    ruta_clean = ruta.replace(".zip", "")

    print(f"  Cargando modelo: {os.path.basename(ruta_clean)} ...")
    modelo = MaskablePPO.load(ruta_clean, device=device)

    # VecNormalize
    vecnorm_path = ruta_clean + "_vecnorm.pkl"
    if not os.path.exists(vecnorm_path):
        alt = ruta_clean.replace(
            "snapshots/", "vecnorm/").replace("snapshot_", "v3_") + ".pkl"
        if os.path.exists(alt):
            vecnorm_path = alt
        else:
            vecnorm_path = None

    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"    VecNormalize: {os.path.basename(vecnorm_path)} ✓")
    else:
        print(f"    VecNormalize: NO ENCONTRADO ⚠️")

    return modelo, vecnorm_path


# ──────────────────────────────────────────────────────────────
# Análisis de UNA mano completa
# ──────────────────────────────────────────────────────────────

def analizar_mano(
    seed: int,
    agente_idx: int,
    politica: PoliticaSB3V3,
    rollout_tipo: str = "experto",
    verbose: bool = False,
) -> Tuple[List[ResultadoDecision], List[int], int]:
    """Juega una mano completa analizando cada decisión del agente.

    Returns:
        (decisiones, puntuacion_final, q_capturador)
    """
    import random
    random.seed(seed)
    rng = np.random.default_rng(seed)

    from src.agentes.bot_experto import BotExperto
    bots_op = [BotExperto() for _ in range(4)]

    motor = MotorCorazones()
    motor.repartir()

    decisiones: List[ResultadoDecision] = []

    while motor._mano_activa:
        if len(motor.mesa) == 4:
            motor.resolver_baza()
            if not motor._mano_activa:
                break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        if not legales:
            break

        if idx == agente_idx:
            carta_elegida = politica(motor, idx, legales, deterministic=True)

            # Analizar solo si hay al menos 2 opciones
            if len(legales) >= 2:
                try:
                    resultado = analizar_decision(
                        motor, agente_idx, legales, carta_elegida,
                        vacios=None,
                        rollout_tipo=rollout_tipo,
                        rng=rng,
                    )
                    decisiones.append(resultado)
                except Exception as e:
                    if verbose:
                        print(
                            f"  ⚠️ Error en analizar_decision (baza {motor.numero_baza}): {e}")

            motor.jugar_carta(idx, carta_elegida)
        else:
            carta = bots_op[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)

    # Resultado final
    puntuacion = motor.calcular_puntuacion_mano()
    q_capturador = -1
    for i, j in enumerate(motor.jugadores):
        for c in j.bazas_ganadas:
            if c.es_dama_de_picas:
                q_capturador = i

    return decisiones, puntuacion, q_capturador


# ──────────────────────────────────────────────────────────────
# Agregación de resultados
# ──────────────────────────────────────────────────────────────

@dataclass
class AgregadoAnalisis:
    """Estadísticas agregadas del análisis PIMC de N manos."""
    nombre: str = ""
    num_manos: int = 0
    num_decisiones: int = 0
    num_optimas: int = 0          # coste == 0
    num_suboptimas: int = 0       # coste > 0
    num_exactas: int = 0          # decisiones con PIMC exacto
    coste_total: float = 0.0
    coste_promedio: float = 0.0
    coste_max: float = 0.0

    # Por situación
    por_situacion: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Por etapa (baza temprana 1-4, media 5-8, tardía 9-13)
    por_etapa: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Errores más graves (top 10)
    peores_errores: List[ResultadoDecision] = field(default_factory=list)

    # Puntuación promedio del agente
    score_promedio: float = 0.0
    score_std: float = 0.0

    def imprimir(self) -> None:
        """Imprime resumen detallado del análisis."""
        print(f"\n{'='*70}")
        print(f"  {self.nombre}")
        print(f"{'='*70}")
        print(f"  Manos jugadas:          {self.num_manos}")
        print(
            f"  Puntuación promedio:    {self.score_promedio:.1f} ± {self.score_std:.1f}")
        print(f"  Decisiones analizadas:  {self.num_decisiones}")
        print(
            f"    Exactas (enum):       {self.num_exactas} ({self.num_exactas/max(1, self.num_decisiones)*100:.0f}%)")
        print(
            f"    Óptimas (coste=0):    {self.num_optimas} ({self.num_optimas/max(1, self.num_decisiones)*100:.0f}%)")
        print(
            f"    Subóptimas (coste>0): {self.num_suboptimas} ({self.num_suboptimas/max(1, self.num_decisiones)*100:.0f}%)")
        print(f"  Coste total:            {self.coste_total:.1f} pts")
        print(f"  Coste promedio:         {self.coste_promedio:.2f} pts/error")
        print(f"  Coste máximo:           {self.coste_max:.1f} pts")

        print(f"\n  ── Por situación ──")
        for sit, stats in sorted(self.por_situacion.items()):
            n = stats["n"]
            pct_opt = stats["optimas"] / max(1, n) * 100
            coste_prom = stats["coste_total"] / max(1, stats["suboptimas"])
            print(f"    {sit:12s}: {n:3d} decis | "
                  f"{pct_opt:5.0f}% óptimas | "
                  f"coste prom subópt: {coste_prom:.2f}")

        print(f"\n  ── Por etapa ──")
        for etapa, stats in sorted(self.por_etapa.items()):
            n = stats["n"]
            pct_opt = stats["optimas"] / max(1, n) * 100
            coste_prom = stats["coste_total"] / max(1, stats["suboptimas"])
            print(f"    {etapa:12s}: {n:3d} decis | "
                  f"{pct_opt:5.0f}% óptimas | "
                  f"coste prom subópt: {coste_prom:.2f}")

        print(f"\n  ── Peores 5 errores ──")
        _PALO = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
        for i, err in enumerate(self.peores_errores[:5]):
            def _nc(c: str) -> str:
                # c is like "AD", "10H", "QS"
                palo_char = {"T": "♣", "D": "♦", "P": "♠", "C": "♥"}
                pc = palo_char.get(c[-1], c[-1])
                return f"{c[:-1]}{pc}"

            print(f"    B{err.baza:>2} {err.situacion:10s} | "
                  f"Eligió {_nc(err.carta_elegida)} "
                  f"({err.score_elegido:.1f}pts) | "
                  f"Óptima {_nc(err.carta_optima)} "
                  f"({err.score_optimo:.1f}pts) | "
                  f"Coste {err.coste:+.1f} | "
                  f"{'EXACTO' if err.exacto else 'approx'}")


# ──────────────────────────────────────────────────────────────
# Agregación
# ──────────────────────────────────────────────────────────────

def _etapa(baza_num: int) -> str:
    if baza_num <= 4:
        return "temprana (1-4)"
    elif baza_num <= 8:
        return "media (5-8)"
    else:
        return "tardía (9-13)"


def agregar_decisiones(
    decisiones: List[ResultadoDecision],
    scores: List[float],
) -> AgregadoAnalisis:
    """Agrega una lista de decisiones a un AgregadoAnalisis."""
    agg = AgregadoAnalisis()
    agg.num_decisiones = len(decisiones)
    agg.num_optimas = sum(1 for d in decisiones if d.coste <= 0.001)
    agg.num_suboptimas = sum(1 for d in decisiones if d.coste > 0.001)
    agg.num_exactas = sum(1 for d in decisiones if d.exacto)
    agg.coste_total = sum(d.coste for d in decisiones if d.coste > 0)
    agg.coste_promedio = agg.coste_total / max(1, agg.num_suboptimas)
    agg.coste_max = max((d.coste for d in decisiones), default=0.0)

    # Por situación
    for sit in ["liderar", "seguir", "descartar"]:
        agg.por_situacion[sit] = {"n": 0, "optimas": 0,
                                  "suboptimas": 0, "coste_total": 0.0}

    for d in decisiones:
        stats = agg.por_situacion[d.situacion]
        stats["n"] += 1
        if d.coste <= 0.001:
            stats["optimas"] += 1
        else:
            stats["suboptimas"] += 1
            stats["coste_total"] += d.coste

    # Por etapa
    for et in ["temprana (1-4)", "media (5-8)", "tardía (9-13)"]:
        agg.por_etapa[et] = {"n": 0, "optimas": 0,
                             "suboptimas": 0, "coste_total": 0.0}

    for d in decisiones:
        stats = agg.por_etapa[_etapa(d.baza)]
        stats["n"] += 1
        if d.coste <= 0.001:
            stats["optimas"] += 1
        else:
            stats["suboptimas"] += 1
            stats["coste_total"] += d.coste

    # Peores errores
    agg.peores_errores = sorted(
        [d for d in decisiones if d.coste > 0.001],
        key=lambda d: d.coste, reverse=True,
    )

    # Score
    agg.score_promedio = np.mean(scores)
    agg.score_std = np.std(scores)
    agg.num_manos = len(scores)

    return agg


# ──────────────────────────────────────────────────────────────
# Análisis de BotExperto (baseline)
# ──────────────────────────────────────────────────────────────

def analizar_bot_experto(
    num_manos: int,
    seed_base: int,
    rollout_tipo: str = "experto",
) -> AgregadoAnalisis:
    """Analiza las decisiones de BotExperto como baseline."""
    print(f"\n  Analizando BotExperto ({num_manos} manos)...")

    todas_decisiones: List[ResultadoDecision] = []
    todos_scores: List[float] = []

    for i in range(num_manos):
        agente_idx = i % 4
        seed = seed_base + 1000 + i

        from src.agentes.bot_experto import BotExperto
        bot_eval = BotExperto()

        decisiones, puntuacion, _ = analizar_mano_bot(
            seed=seed,
            agente_idx=agente_idx,
            bot_eval=bot_eval,
            rollout_tipo=rollout_tipo,
        )
        todas_decisiones.extend(decisiones)
        todos_scores.append(float(puntuacion[agente_idx]))

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{num_manos} manos completadas")

    agg = agregar_decisiones(todas_decisiones, todos_scores)
    agg.nombre = "BotExperto (baseline)"
    return agg


def analizar_mano_bot(
    seed: int,
    agente_idx: int,
    bot_eval: BotExperto,
    rollout_tipo: str = "experto",
) -> Tuple[List[ResultadoDecision], List[int], int]:
    """Juega una mano con BotExperto como agente y analiza sus decisiones."""
    import random
    random.seed(seed)
    rng = np.random.default_rng(seed)

    bots_op = [BotExperto() for _ in range(4)]

    motor = MotorCorazones()
    motor.repartir()

    decisiones: List[ResultadoDecision] = []

    while motor._mano_activa:
        if len(motor.mesa) == 4:
            motor.resolver_baza()
            if not motor._mano_activa:
                break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        if not legales:
            break

        if idx == agente_idx:
            carta_elegida = bot_eval(motor, idx, legales)

            if len(legales) >= 2:
                try:
                    resultado = analizar_decision(
                        motor, agente_idx, legales, carta_elegida,
                        vacios=None,
                        rollout_tipo=rollout_tipo,
                        rng=rng,
                    )
                    decisiones.append(resultado)
                except Exception:
                    pass

            motor.jugar_carta(idx, carta_elegida)
        else:
            carta = bots_op[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)

    puntuacion = motor.calcular_puntuacion_mano()
    q_capturador = -1
    for i, j in enumerate(motor.jugadores):
        for c in j.bazas_ganadas:
            if c.es_dama_de_picas:
                q_capturador = i

    return decisiones, puntuacion, q_capturador


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  ANÁLISIS PIMC DETALLADO — MODELOS v3")
    print("=" * 70)
    print(f"  Manos por modelo: {NUM_MANOS}")
    print(f"  Rollout: {ROLLOUT_TIPO}")
    print(f"  Device: {DEVICE}")
    print()

    resultados: Dict[str, AgregadoAnalisis] = {}

    # --- Baseline: BotExperto ---
    t0 = time.time()
    agg_bot = analizar_bot_experto(NUM_MANOS, SEED_BASE, ROLLOUT_TIPO)
    agg_bot.imprimir()
    resultados["BotExperto"] = agg_bot
    print(f"  ⏱️  Tiempo: {time.time() - t0:.1f}s")

    # --- Modelos v3 ---
    for nombre, ruta in MODELOS_A_ANALIZAR:
        print(f"\n{'─'*70}")
        print(f"  ANALIZANDO: {nombre}")
        print(f"{'─'*70}")
        t0 = time.time()

        modelo, vecnorm_path = cargar_modelo_v3(ruta, DEVICE)
        politica = PoliticaSB3V3(
            model=modelo, agente_idx=0, vecnorm_path=vecnorm_path)

        todas_decisiones: List[ResultadoDecision] = []
        todos_scores: List[float] = []

        for i in range(NUM_MANOS):
            agente_idx = i % 4
            seed = SEED_BASE + i

            # Actualizar agente_idx para esta mano
            politica.agente_idx = agente_idx

            decisiones, puntuacion, q_capt = analizar_mano(
                seed=seed,
                agente_idx=agente_idx,
                politica=politica,
                rollout_tipo=ROLLOUT_TIPO,
            )
            todas_decisiones.extend(decisiones)
            todos_scores.append(float(puntuacion[agente_idx]))

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{NUM_MANOS} manos completadas "
                      f"({time.time() - t0:.1f}s)")

        agg = agregar_decisiones(todas_decisiones, todos_scores)
        agg.nombre = nombre
        agg.imprimir()
        resultados[nombre] = agg
        print(f"  ⏱️  Tiempo total: {time.time() - t0:.1f}s")

    # --- Comparativa final ---
    print(f"\n{'='*70}")
    print(f"  COMPARATIVA FINAL")
    print(f"{'='*70}")
    print(
        f"  {'Modelo':<22s} {'Score':>6s} {'%Ópt':>6s} {'Coste':>6s} {'Err/Mano':>8s}")
    print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")

    for nombre, agg in resultados.items():
        pct_opt = agg.num_optimas / max(1, agg.num_decisiones) * 100
        err_por_mano = agg.num_suboptimas / max(1, agg.num_manos)
        coste_por_mano = agg.coste_total / max(1, agg.num_manos)
        print(f"  {nombre:<22s} {agg.score_promedio:6.1f} {pct_opt:5.0f}% {coste_por_mano:5.1f} {err_por_mano:7.1f}")

    # --- Clasificación de errores ---
    print(f"\n{'='*70}")
    print(f"  CLASIFICACIÓN DE ERRORES POR SITUACIÓN")
    print(f"{'='*70}")

    for nombre, agg in resultados.items():
        print(f"\n  {nombre}:")
        for sit in ["liderar", "seguir", "descartar"]:
            stats = agg.por_situacion.get(sit, {})
            n = stats.get("n", 0)
            sub = stats.get("suboptimas", 0)
            if n > 0:
                print(f"    {sit:12s}: {sub}/{n} errores "
                      f"({sub/max(1, n)*100:.0f}%) | "
                      f"coste acum: {stats.get('coste_total', 0):.1f}")

    print("\n✅ Análisis completado.")


if __name__ == "__main__":
    main()
