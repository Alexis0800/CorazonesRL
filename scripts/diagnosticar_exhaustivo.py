#!/usr/bin/env python
"""
Diagnóstico EXHAUSTIVO de decisiones usando PIMC con alto muestreo.

A diferencia de diagnosticar_modelo.py (que solo analiza baza ≥ 8 con
enumeración exacta), este script analiza TODAS las bazas (1-13) usando:

  - Bazas 1-7:   PIMC sampling 500 mundos (BotExperto rollout)
  - Bazas 8-9:   PIMC sampling 500 mundos (exacto no viable aún)
  - Bazas 10-13: Enumeración completa (ground truth)

Además categoriza errores por TIPO (captura Q, corazones, no descarte, etc.)
y por CAUSA RAÍZ (void tracking, card counting, risk assessment).

Uso:
    python scripts/diagnosticar_exhaustivo.py \
        --modelo models/v2_1/snapshots/snapshot_0002600000.zip \
        --manos 50 --mundos 500 --incluir-experto --seed 42
"""

from __future__ import annotations
from src.mcts.analisis import (
    pimc_exacto,
    _num_mundos_posibles,
    _MAX_MUNDOS_EXACTO,
    _BAZA_MINIMA_EXACTA,
)
from src.mcts.pimc import (
    _clonar_motor,
    determinizar,
    simular_resto_mano,
    _puntaje_esperado_por_carta,
    crear_bots_rollout,
    mcts_mejor_jugada,
)
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ──────────────────────────────────────────────────────────────
# Tipos de error
# ──────────────────────────────────────────────────────────────


@dataclass
class ErrorDetallado:
    """Un error individual con todo el contexto."""
    baza: int
    situacion: str          # "liderar", "seguir", "descartar"
    carta_elegida: str      # nombre "12P"
    carta_optima: str       # nombre
    score_elegido: float
    score_optimo: float
    coste: float            # score_elegido - score_optimo (> 0)
    n_opciones: int
    es_exacto: bool
    tipo_error: str         # categoría del error (ver abajo)
    causa_raiz: str         # causa raíz inferida


@dataclass
class DiagnosticoExhaustivo:
    """Resultado completo del diagnóstico."""
    nombre: str
    num_manos: int
    num_mundos: int         # mundos de sampling
    puntuacion_media: float = 0.0
    puntuacion_std: float = 0.0
    tasa_cero: float = 0.0
    tasa_q: float = 0.0

    # Decisiones analizadas
    total_decisiones: int = 0
    decisiones_exactas: int = 0     # bazas 10-13 con enum
    decisiones_sampling: int = 0    # bazas 1-9 con sampling
    total_errores: int = 0
    coste_total: float = 0.0

    # Por fase
    errores_early: int = 0      # baza 1-4
    errores_mid: int = 0        # baza 5-8
    errores_late: int = 0       # baza 9-13
    coste_early: float = 0.0
    coste_mid: float = 0.0
    coste_late: float = 0.0

    # Por situación
    errores_liderar: int = 0
    errores_seguir: int = 0
    errores_descartar: int = 0
    coste_liderar: float = 0.0
    coste_seguir: float = 0.0
    coste_descartar: float = 0.0

    # Por tipo de error
    errores_por_tipo: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    coste_por_tipo: Dict[str, float] = field(
        default_factory=lambda: defaultdict(float))

    # Por causa raíz
    errores_por_causa: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))

    # Lista de errores graves (coste ≥ 3)
    errores_graves: List[ErrorDetallado] = field(default_factory=list)

    # Puntuaciones
    puntuaciones: List[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Clasificación de errores
# ──────────────────────────────────────────────────────────────

def _clasificar_error(
    motor: MotorCorazones,
    agente_idx: int,
    carta_elegida: Carta,
    carta_optima: Carta,
    situacion: str,
    coste: float,
) -> Tuple[str, str]:
    """Clasifica un error por tipo y causa raíz.

    Returns:
        (tipo_error, causa_raiz)
    """
    tipo = "otro"
    causa = "risk_assessment"

    # ── Tipo de error ──
    if carta_elegida.es_dama_de_picas and coste > 5:
        # Eligió Q♠ cuando había mejor opción
        tipo = "captura_q_evitable"
    elif carta_optima.es_dama_de_picas and coste > 5:
        # Debía jugar Q♠ (para dump) pero no lo hizo
        tipo = "no_descarte_q"
    elif carta_elegida.es_corazon and not carta_optima.es_corazon and coste >= 1:
        tipo = "captura_corazon"
    elif situacion == "liderar" and coste >= 2:
        tipo = "liderar_malo"
    elif situacion == "seguir" and carta_elegida.palo == motor.palo_de_salida:
        tipo = "seguir_malo"
    elif situacion == "descartar":
        tipo = "descartar_malo"

    # Refinar: ¿bloqueo de pozo?
    # Verificar si algún rival tiene ≥6 corazones capturados
    for i, j in enumerate(motor.jugadores):
        if i == agente_idx:
            continue
        pts_corazones = sum(1 for c in j.bazas_ganadas if c.es_corazon)
        if pts_corazones >= 5:
            if carta_elegida.es_corazon and coste >= 1:
                tipo = "no_bloqueo_pozo"
            break

    # ── Causa raíz ──
    # Detectar voids: si el agente jugó un palo donde un rival es void
    # (esto es inferido, no exacto)
    palo_elegido = carta_elegida.palo
    palo_optimo = carta_optima.palo

    if situacion == "seguir" and palo_elegido == palo_optimo:
        # Mismo palo, diferente carta → problema de card counting
        causa = "card_counting"
    elif situacion == "liderar" and coste >= 2:
        causa = "planning"
    elif situacion == "descartar" and coste >= 2:
        causa = "void_tracking_or_risk"
    elif carta_elegida.es_corazon and not carta_optima.es_corazon:
        causa = "risk_assessment"
    elif carta_elegida.es_dama_de_picas:
        causa = "qs_management"

    return tipo, causa


# ──────────────────────────────────────────────────────────────
# Carga de modelo v2_1
# ──────────────────────────────────────────────────────────────

def _cargar_modelo_v2(ruta: str) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray]]:
    """Carga modelo MaskablePPO + VecNormalize."""
    import pickle
    from sb3_contrib import MaskablePPO

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
# Diagnóstico exhaustivo
# ──────────────────────────────────────────────────────────────

def diagnosticar_exhaustivo(
    politica: Callable[[MotorCorazones, int, List[Carta]], Carta],
    nombre: str,
    num_manos: int = 50,
    num_mundos: int = 500,
    seed_inicial: int = 42,
    rollout_tipo: str = "experto",
    verbose: bool = True,
) -> DiagnosticoExhaustivo:
    """Diagnostica UNA política en TODAS las bazas usando PIMC de alto muestreo.

    Para cada decisión del agente en cada baza:
    1. Si baza ≥ 10 y mundos ≤ 100K → enumeración completa (ground truth)
    2. Si no → PIMC con mundos adaptativos:
       - Bazas 1-4: 100 mundos (alta incertidumbre, poca precisión necesaria)
       - Bazas 5-7: 200 mundos
       - Bazas 8-9: 400 mundos
       El parámetro num_mundos escala estos valores base.

    Args:
        politica: Callable(motor, idx, legales) -> Carta
        nombre: Etiqueta
        num_manos: Manos a evaluar
        num_mundos: Factor base de mundos (default 500 → 100/200/400/500)
        seed_inicial: Semilla base
        rollout_tipo: Política de rollout para PIMC
        verbose: Mostrar progreso

    Returns:
        DiagnosticoExhaustivo
    """
    from src.agentes.bot_experto import BotExperto

    diag = DiagnosticoExhaustivo(
        nombre=nombre, num_manos=num_manos, num_mundos=num_mundos,
    )
    rng = np.random.default_rng(seed_inicial)

    # Mundos adaptativos por fase
    factor = num_mundos / 500.0  # escalar con el parámetro
    MUNDOS_EARLY = max(50, int(100 * factor))    # bazas 1-4
    MUNDOS_MID = max(100, int(200 * factor))     # bazas 5-7
    MUNDOS_LATE_SAMPLE = max(150, int(400 * factor))  # bazas 8-9

    _PALO_LABEL = {0: "T", 1: "D", 2: "P", 3: "C"}

    def _nombre(c: Carta) -> str:
        return f"{c.valor}{_PALO_LABEL[c.palo]}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  DIAGNÓSTICO EXHAUSTIVO: {nombre}")
        print(f"  Manos: {num_manos} | Rollout: {rollout_tipo}")
        print(f"  Bazas: 1-13 | Exacto≥10 | "
              f"Sample: early={MUNDOS_EARLY}w mid={MUNDOS_MID}w late={MUNDOS_LATE_SAMPLE}w")
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
                # ── Turno del agente (modelo o BotExperto) ──
                carta_elegida = politica(motor, 0, legales)

                # Analizar decisión si hay ≥2 opciones
                if len(legales) >= 2:
                    diag.total_decisiones += 1
                    baza = motor.numero_baza

                    # Determinar si podemos hacer enumeración exacta
                    n_mundos_posibles = _num_mundos_posibles(motor, 0)
                    puede_exacto = (
                        baza >= _BAZA_MINIMA_EXACTA
                        and n_mundos_posibles <= _MAX_MUNDOS_EXACTO
                    )

                    if puede_exacto:
                        # ── Enumeración completa (ground truth) ──
                        _, scores, exacto = pimc_exacto(
                            motor, 0, legales,
                            vacios={},
                            rollout_tipo=rollout_tipo,
                            rng=rng,
                        )
                        diag.decisiones_exactas += 1
                    else:
                        # ── PIMC sampling adaptativo ──
                        if baza <= 4:
                            mundos_sample = MUNDOS_EARLY
                        elif baza <= 7:
                            mundos_sample = MUNDOS_MID
                        else:
                            mundos_sample = MUNDOS_LATE_SAMPLE
                        scores = _puntaje_esperado_por_carta(
                            motor, 0, legales,
                            vacios={},
                            num_mundos=mundos_sample,
                            rng=rng,
                            crear_bots=lambda: crear_bots_rollout(
                                tipo=rollout_tipo, rng=rng),
                        )
                        diag.decisiones_sampling += 1
                        exacto = False

                    # Encontrar la carta óptima según PIMC
                    carta_optima = min(legales, key=lambda c: scores[c.id])
                    score_elegido = scores[carta_elegida.id]
                    score_optimo = scores[carta_optima.id]
                    coste = score_elegido - score_optimo

                    if coste > 0.001:
                        diag.total_errores += 1
                        diag.coste_total += coste

                        # Clasificar error
                        tipo_err, causa_err = _clasificar_error(
                            motor, 0, carta_elegida, carta_optima,
                            _situacion(motor, legales), coste,
                        )
                        diag.errores_por_tipo[tipo_err] += 1
                        diag.coste_por_tipo[tipo_err] += coste
                        diag.errores_por_causa[causa_err] += 1

                        # Por fase
                        if baza <= 4:
                            diag.errores_early += 1
                            diag.coste_early += coste
                        elif baza <= 8:
                            diag.errores_mid += 1
                            diag.coste_mid += coste
                        else:
                            diag.errores_late += 1
                            diag.coste_late += coste

                        # Por situación
                        situacion = _situacion(motor, legales)
                        if situacion == "liderar":
                            diag.errores_liderar += 1
                            diag.coste_liderar += coste
                        elif situacion == "seguir":
                            diag.errores_seguir += 1
                            diag.coste_seguir += coste
                        else:
                            diag.errores_descartar += 1
                            diag.coste_descartar += coste

                        # Errores graves
                        if coste >= 3.0:
                            diag.errores_graves.append(ErrorDetallado(
                                baza=baza,
                                situacion=situacion,
                                carta_elegida=_nombre(carta_elegida),
                                carta_optima=_nombre(carta_optima),
                                score_elegido=round(score_elegido, 2),
                                score_optimo=round(score_optimo, 2),
                                coste=round(coste, 2),
                                n_opciones=len(legales),
                                es_exacto=exacto,
                                tipo_error=tipo_err,
                                causa_raiz=causa_err,
                            ))

                motor.jugar_carta(0, carta_elegida)
            else:
                # ── Turno de oponente ──
                carta = oponentes[idx](motor, idx, legales)
                motor.jugar_carta(idx, carta)

        # Puntuación final
        mi_score = motor.jugadores[0].contar_puntos_bazas()
        diag.puntuaciones.append(mi_score)

        if verbose and (mano_idx + 1) % max(1, num_manos // 10) == 0:
            media = np.mean(diag.puntuaciones)
            print(f"  Mano {mano_idx+1:4d}/{num_manos} | "
                  f"Score: {media:.1f} | Err: {diag.total_errores} | "
                  f"Coste: {diag.coste_total:.1f} pts | "
                  f"Exactas: {diag.decisiones_exactas} | "
                  f"Sampling: {diag.decisiones_sampling}")

    # Métricas finales
    arr = np.array(diag.puntuaciones, dtype=np.float64)
    diag.puntuacion_media = float(np.mean(arr))
    diag.puntuacion_std = float(np.std(arr))
    diag.tasa_cero = float(np.mean(arr == 0))
    diag.tasa_q = float(
        np.mean([1 for s in diag.puntuaciones if s >= 13 and s < 26]))

    return diag


def _situacion(motor: MotorCorazones, legales: List[Carta]) -> str:
    """Determina la situación actual."""
    if not motor.mesa:
        return "liderar"
    palo = motor.palo_de_salida
    if any(c.palo == palo for c in legales):
        return "seguir"
    return "descartar"


# ──────────────────────────────────────────────────────────────
# Reporte
# ──────────────────────────────────────────────────────────────

def imprimir_reporte_exhaustivo(diag: DiagnosticoExhaustivo) -> None:
    """Imprime reporte detallado."""
    print(f"\n{'='*60}")
    print(f"  REPORTE EXHAUSTIVO: {diag.nombre}")
    print(f"{'='*60}")

    print(f"\n  📊 Métricas globales ({diag.num_manos} manos):")
    print(
        f"    Puntuación media : {diag.puntuacion_media:.2f} ± {diag.puntuacion_std:.2f}")
    print(f"    Tasa 0 puntos    : {diag.tasa_cero*100:.1f}%")
    print(f"    Tasa captura Q♠  : {diag.tasa_q*100:.1f}%")

    print(f"\n  🔍 Cobertura de análisis:")
    print(f"    Total decisiones : {diag.total_decisiones}")
    print(f"    Exactas (enum)   : {diag.decisiones_exactas} "
          f"({diag.decisiones_exactas/max(1, diag.total_decisiones)*100:.0f}%)")
    print(f"    Sampling ({diag.num_mundos}w) : {diag.decisiones_sampling} "
          f"({diag.decisiones_sampling/max(1, diag.total_decisiones)*100:.0f}%)")
    print(f"    Total errores    : {diag.total_errores} "
          f"({diag.total_errores/max(1, diag.total_decisiones)*100:.1f}%)")
    print(f"    Coste total      : {diag.coste_total:.2f} pts")
    print(
        f"    Coste medio/err  : {diag.coste_total/max(1, diag.total_errores):.2f} pts")
    print(
        f"    Coste medio/dec  : {diag.coste_total/max(1, diag.total_decisiones):.4f} pts")

    print(f"\n  📅 Errores por fase:")
    for nombre, n, coste in [
        ("Bazas 1-4 (early)", diag.errores_early, diag.coste_early),
        ("Bazas 5-8 (mid)  ", diag.errores_mid, diag.coste_mid),
        ("Bazas 9-13 (late)", diag.errores_late, diag.coste_late),
    ]:
        if n > 0:
            print(f"    {nombre}: {n:3d} errores | coste={coste:.1f} pts | "
                  f"media={coste/n:.2f} pts")

    print(f"\n  🎯 Errores por situación:")
    for nombre, n, coste in [
        ("Liderar  ", diag.errores_liderar, diag.coste_liderar),
        ("Seguir   ", diag.errores_seguir, diag.coste_seguir),
        ("Descartar", diag.errores_descartar, diag.coste_descartar),
    ]:
        if n > 0:
            print(f"    {nombre}: {n:3d} errores | coste={coste:.1f} pts | "
                  f"media={coste/n:.2f} pts")

    print(f"\n  🏷️ Errores por tipo:")
    for tipo in sorted(diag.errores_por_tipo, key=lambda t: diag.coste_por_tipo[t], reverse=True):
        n = diag.errores_por_tipo[tipo]
        c = diag.coste_por_tipo[tipo]
        print(
            f"    {tipo:22s}: {n:3d} errores | coste={c:.1f} pts | media={c/n:.2f} pts")

    print(f"\n  🔬 Causa raíz inferida:")
    for causa in sorted(diag.errores_por_causa, key=lambda t: diag.errores_por_causa[t], reverse=True):
        n = diag.errores_por_causa[causa]
        print(f"    {causa:25s}: {n:3d} errores")

    if diag.errores_graves:
        print(
            f"\n  💀 Errores graves (coste ≥ 3 pts): {len(diag.errores_graves)}")
        peores = sorted(diag.errores_graves,
                        key=lambda d: d.coste, reverse=True)[:8]
        for d in peores:
            exacto_str = "✓" if d.es_exacto else "~"
            print(f"    B{d.baza:2d} {d.situacion:9s} | "
                  f"Eligió {d.carta_elegida:4s} ({d.score_elegido:5.1f}) → "
                  f"Ópt {d.carta_optima:4s} ({d.score_optimo:5.1f}) | "
                  f"Δ={d.coste:+.1f} [{exacto_str}] | "
                  f"{d.tipo_error} | {d.causa_raiz}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnóstico EXHAUSTIVO con PIMC alto muestreo en TODAS las bazas",
    )
    parser.add_argument("--modelo", type=str, required=True,
                        help="Ruta al snapshot .zip del modelo.")
    parser.add_argument("--manos", type=int, default=50,
                        help="Manos a evaluar (default 50).")
    parser.add_argument("--mundos", type=int, default=500,
                        help="Mundos de sampling para PIMC en bazas 1-9 (default 500).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout", type=str, default="experto",
                        choices=["evasivo", "experto", "mixto", "conservador", "agresivo"])
    parser.add_argument("--incluir-experto", action="store_true",
                        help="También diagnosticar BotExperto como baseline.")
    parser.add_argument("--output", type=str, default=None,
                        help="Guardar resultados JSON.")

    args = parser.parse_args()

    # ── Cargar modelo ──
    print(f"Cargando modelo: {args.modelo}")
    modelo, obs_mean, obs_var = _cargar_modelo_v2(args.modelo)

    def politica_modelo(motor, idx, legales):
        return _elegir_con_modelo(modelo, obs_mean, obs_var, motor, idx, legales)

    # ── Diagnosticar modelo ──
    t0 = time.time()
    diag_modelo = diagnosticar_exhaustivo(
        politica=politica_modelo,
        nombre=f"v2_1 ({os.path.basename(args.modelo)})",
        num_manos=args.manos,
        num_mundos=args.mundos,
        seed_inicial=args.seed,
        rollout_tipo=args.rollout,
    )
    elapsed = time.time() - t0
    print(f"\n  ⏱ Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    imprimir_reporte_exhaustivo(diag_modelo)

    # ── Baseline BotExperto ──
    diag_experto = None
    if args.incluir_experto:
        from src.agentes.bot_experto import BotExperto
        experto = BotExperto()

        def politica_experto(motor, idx, legales):
            return experto(motor, idx, legales)

        print(f"\n{'─'*60}")
        print("  Ejecutando baseline BotExperto...")
        t0 = time.time()
        diag_experto = diagnosticar_exhaustivo(
            politica=politica_experto,
            nombre="BotExperto (baseline)",
            num_manos=args.manos,
            num_mundos=args.mundos,
            seed_inicial=args.seed,
            rollout_tipo=args.rollout,
        )
        elapsed = time.time() - t0
        print(f"\n  ⏱ Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        imprimir_reporte_exhaustivo(diag_experto)

    # ── Comparativa ──
    if diag_experto:
        print(f"\n{'='*60}")
        print(f"  COMPARATIVA v2_1 vs BOTEXPERTO")
        print(f"{'='*60}")
        rows = [
            ("Puntuación media", diag_modelo.puntuacion_media,
             diag_experto.puntuacion_media),
            ("Tasa 0 puntos (%)", diag_modelo.tasa_cero *
             100, diag_experto.tasa_cero*100),
            ("Tasa error (%)",
             diag_modelo.total_errores /
             max(1, diag_modelo.total_decisiones)*100,
             diag_experto.total_errores/max(1, diag_experto.total_decisiones)*100),
            ("Coste total (pts)", diag_modelo.coste_total, diag_experto.coste_total),
            ("Coste medio/error", diag_modelo.coste_total/max(1, diag_modelo.total_errores),
             diag_experto.coste_total/max(1, diag_experto.total_errores)),
            ("Errores early (1-4)", float(diag_modelo.errores_early),
             float(diag_experto.errores_early)),
            ("Errores mid (5-8)", float(diag_modelo.errores_mid),
             float(diag_experto.errores_mid)),
            ("Errores late (9-13)", float(diag_modelo.errores_late),
             float(diag_experto.errores_late)),
        ]
        print(f"  {'Métrica':25s} {'v2_1':>10s} {'Experto':>10s} {'Δ':>10s}")
        print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
        for nombre, v, e in rows:
            print(f"  {nombre:25s} {v:10.2f} {e:10.2f} {v-e:+10.2f}")

    # ── Guardar JSON ──
    if args.output:
        def _diag_to_dict(d: DiagnosticoExhaustivo) -> dict:
            return {
                "nombre": d.nombre,
                "num_manos": d.num_manos,
                "num_mundos": d.num_mundos,
                "puntuacion_media": d.puntuacion_media,
                "puntuacion_std": d.puntuacion_std,
                "tasa_cero": d.tasa_cero,
                "tasa_q": d.tasa_q,
                "total_decisiones": d.total_decisiones,
                "decisiones_exactas": d.decisiones_exactas,
                "decisiones_sampling": d.decisiones_sampling,
                "total_errores": d.total_errores,
                "coste_total": round(d.coste_total, 2),
                "errores_early": d.errores_early,
                "errores_mid": d.errores_mid,
                "errores_late": d.errores_late,
                "coste_early": round(d.coste_early, 2),
                "coste_mid": round(d.coste_mid, 2),
                "coste_late": round(d.coste_late, 2),
                "errores_liderar": d.errores_liderar,
                "errores_seguir": d.errores_seguir,
                "errores_descartar": d.errores_descartar,
                "errores_por_tipo": dict(d.errores_por_tipo),
                "coste_por_tipo": {k: round(v, 2) for k, v in d.coste_por_tipo.items()},
                "errores_por_causa": dict(d.errores_por_causa),
                "errores_graves_count": len(d.errores_graves),
                "top_errores_graves": [
                    {
                        "baza": e.baza, "situacion": e.situacion,
                        "carta_elegida": e.carta_elegida,
                        "carta_optima": e.carta_optima,
                        "coste": e.coste, "exacto": e.es_exacto,
                        "tipo": e.tipo_error, "causa": e.causa_raiz,
                    }
                    for e in sorted(d.errores_graves, key=lambda x: x.coste, reverse=True)[:10]
                ],
            }
        data = {"modelo": _diag_to_dict(diag_modelo)}
        if diag_experto:
            data["experto"] = _diag_to_dict(diag_experto)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Guardado: {args.output}")


if __name__ == "__main__":
    main()
