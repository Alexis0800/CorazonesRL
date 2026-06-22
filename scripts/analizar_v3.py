#!/usr/bin/env python
"""
Analisis detallado de modelo v3 contra el campo estandar.

Evalua el modelo v3 (250-dim, Transformer, single-hand) en manos
individuales contra el campo canonico:

    Mesa: [Modelo, BotExperto, BotExperto, BotRotativo]

Donde BotRotativo alterna entre {conservador, agresivo, evasivo}
cada mano para garantizar diversidad tactica.

Metricas:
  - Win rate (% de manos con <=8 pts)
  - Score promedio, mediana, distribucion
  - % de manos con 0 puntos (perfect hand)
  - Captura de Q♠
  - Desglose por posicion (seat 0/1/2/3)
  - Pozo (shooting the moon) logrado o bloqueado

Uso:
    python scripts/analizar_v3.py
    python scripts/analizar_v3.py --modelo models/v3/snapshots/snapshot_0001900000 --manos 300
"""

from __future__ import annotations
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
import numpy as np

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

_MODELO_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "v3", "snapshots", "snapshot_0001900000",
)


# ------------------------------------------------------------------
# Carga del modelo v3
# ------------------------------------------------------------------

def cargar_modelo_v3(ruta_modelo: str, device: str = "cpu"):
    """Carga MaskablePPO v3 + VecNormalize desde un snapshot.

    Args:
        ruta_modelo: Ruta al snapshot (sin .zip o con .zip).
        device: Dispositivo ('cpu', 'cuda', 'dml').

    Returns:
        Tuple (modelo, vecnorm_path).
    """
    ruta_clean = ruta_modelo.replace(".zip", "")
    modelo = MaskablePPO.load(ruta_clean, device=device)

    # Cargar VecNormalize asociado
    vecnorm_path = ruta_clean + "_vecnorm.pkl"
    if not os.path.exists(vecnorm_path):
        vecnorm_path = ruta_clean.replace(
            "snapshots/", "vecnorm/").replace("snapshot_", "v3_") + ".pkl"
        if not os.path.exists(vecnorm_path):
            vecnorm_path = None

    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"   VecNormalize: {os.path.basename(vecnorm_path)}")
    else:
        print("   ⚠️  VecNormalize no encontrado")

    return modelo, vecnorm_path


# ------------------------------------------------------------------
# Estructuras de datos
# ------------------------------------------------------------------

@dataclass
class ResultadoMano:
    """Estadísticas de una mano individual."""
    score: int = 0               # puntos del agente en esta mano
    q_capturada: bool = False    # ¿el agente capturó Q♠?
    pozo_logrado: bool = False   # ¿el agente hizo pozo?
    pozo_bloqueado: bool = False  # ¿el agente bloqueó un pozo enemigo?
    cero_puntos: bool = False    # ¿0 puntos?
    seat: int = 0                # posición del agente (0-3)


@dataclass
class Agregado:
    """Métricas agregadas de N manos."""
    victorias: int = 0
    total_manos: int = 0
    score_total: float = 0.0
    scores: List[float] = field(default_factory=list)
    manos_cero: int = 0
    q_capturadas: int = 0
    pozos_logrados: int = 0
    pozos_bloqueados: int = 0
    por_seat: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.por_seat:
            self.por_seat = {s: [] for s in range(4)}


# ------------------------------------------------------------------
# Evaluación single-hand
# ------------------------------------------------------------------

def evaluar_manos(
    modelo,
    vecnorm_path: Optional[str],
    num_manos: int,
    seed: int,
    verbose: bool = False,
) -> Agregado:
    """Evalua N manos contra el campo estandar [Experto, Experto, BotRotativo].

    Args:
        modelo: MaskablePPO cargado.
        vecnorm_path: Ruta al VecNormalize .pkl.
        num_manos: Numero de manos a jugar.
        seed: Semilla base.
        verbose: Si True, imprime detalle por mano.

    Returns:
        Agregado con metricas.
    """
    from src.v3.entorno import CorazonesEnvV3
    from src.v3.evaluacion import _construir_oponentes_estandar

    agg = Agregado()

    for i in range(num_manos):
        agente_idx = i % 4  # rotar posicion
        rng = np.random.default_rng(seed + i)

        # Campo estandar: [Experto, Experto, BotRotativo]
        politicas = _construir_oponentes_estandar(agente_idx, seed + i)

        env = CorazonesEnvV3(
            agente_idx=agente_idx,
            politicas_oponentes=politicas,
        )
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)

        # Extraer metricas
        puntos_agente = info.get("puntos_agente", 26)
        puntos_mano = info.get("puntos_mano", [0, 0, 0, 0])

        resultado = ResultadoMano(
            score=puntos_agente,
            q_capturada=_detectar_q_capturada(env, agente_idx),
            pozo_logrado=(puntos_agente == 0 and any(
                p == 26 for p in puntos_mano)),
            pozo_bloqueado=_detectar_pozo_bloqueado(
                env, agente_idx, puntos_mano),
            cero_puntos=(puntos_agente == 0),
            seat=agente_idx,
        )

        # Acumular
        agg.total_manos += 1
        agg.score_total += resultado.score
        agg.scores.append(float(resultado.score))
        if resultado.score <= 8:
            agg.victorias += 1
        if resultado.cero_puntos:
            agg.manos_cero += 1
        if resultado.q_capturada:
            agg.q_capturadas += 1
        if resultado.pozo_logrado:
            agg.pozos_logrados += 1
        if resultado.pozo_bloqueado:
            agg.pozos_bloqueados += 1
        agg.por_seat[agente_idx].append(float(resultado.score))

        if verbose and i < 10:
            qs = "Q♠" if resultado.q_capturada else ""
            pozo = "POZO!" if resultado.pozo_logrado else ""
            print(f"  M#{i+1} seat={agente_idx} "
                  f"score={resultado.score} {qs} {pozo}")

        env.close()

    return agg


def _detectar_q_capturada(env, agente_idx: int) -> bool:
    """Detecta si el agente capturó Q♠ en la mano."""
    motor = env.motor
    for c in motor.jugadores[agente_idx].bazas_ganadas:
        if c.es_dama_de_picas:
            return True
    return False


def _detectar_pozo_bloqueado(env, agente_idx: int, puntos_mano: List[int]) -> bool:
    """Detecta si el agente bloqueó un pozo enemigo.

    Un pozo bloqueado ocurre cuando:
    - El agente NO hizo pozo (no tiene 0 con 26s para otros)
    - Pero algún rival capturó ≥9 corazones y el agente le quitó Q♠
      o capturó algún corazón que impidió el pozo.
    """
    motor = env.motor

    # Contar corazones por jugador
    corazones_por_j = [0, 0, 0, 0]
    for j in range(4):
        for c in motor.jugadores[j].bazas_ganadas:
            if c.es_corazon:
                corazones_por_j[j] += 1

    # Si alguien tiene ≥9 corazones pero no hizo pozo, el agente puede haber bloqueado
    for j in range(4):
        if j != agente_idx and corazones_por_j[j] >= 9:
            if puntos_mano[j] != 0:  # no hizo pozo
                # ¿El agente capturó al menos 1 corazón del rival?
                if corazones_por_j[agente_idx] > 0:
                    return True
                # ¿O el agente capturó Q♠ cuando el rival tenía todos los corazones?
                if _detectar_q_capturada(env, agente_idx):
                    return True

    return False


# ------------------------------------------------------------------
# Impresión de resultados
# ------------------------------------------------------------------

def imprimir_resultados(
    agg: Agregado,
    num_manos: int,
    modelo_path: str,
) -> None:
    """Imprime tabla de resultados de evaluacion estandar."""
    print(f"\n{'═'*72}")
    print(
        f"ANALISIS v3 — {num_manos} manos | Campo: [Modelo, Experto, Experto, Bot]")
    print(f"Modelo: {modelo_path}")
    print(f"{'─'*72}")

    if agg.total_manos == 0:
        print("  Sin datos.")
        return

    # Tabla principal
    print(f"  {'Metrica':<28} {'Valor':>14}")
    print(f"  {'─'*28} {'─'*14}")

    _fila_simple("Win Rate (<=8 pts)", agg,
                 lambda a: f"{a.victorias/a.total_manos:.1%}")
    _fila_simple("Score Promedio", agg,
                 lambda a: f"{a.score_total/a.total_manos:.1f}")
    _fila_simple("Score Mediana", agg,
                 lambda a: f"{np.median(a.scores):.1f}" if a.scores else "N/A")
    _fila_simple("Score Minimo", agg,
                 lambda a: f"{min(a.scores):.0f}" if a.scores else "N/A")
    _fila_simple("Score Maximo", agg,
                 lambda a: f"{max(a.scores):.0f}" if a.scores else "N/A")
    _fila_simple("Manos 0 Puntos", agg,
                 lambda a: f"{a.manos_cero/a.total_manos:.1%}")
    _fila_simple("Q♠ Capturada", agg,
                 lambda a: f"{a.q_capturadas/a.total_manos:.1%}")
    _fila_simple("Pozos Logrados", agg,
                 lambda a: f"{a.pozos_logrados}")
    _fila_simple("Pozos Bloqueados", agg,
                 lambda a: f"{a.pozos_bloqueados}")

    print(f"{'─'*72}")

    # Distribucion de scores
    _imprimir_distribucion(agg)

    # Desglose por posicion
    _imprimir_por_seat(agg, "Campo Estandar")

    print(f"{'═'*72}")


def _fila_simple(nombre: str, a: Agregado, fn) -> None:
    va = fn(a) if a.total_manos else "N/A"
    print(f"  {nombre:<28} {str(va):>14}")


def _fila(nombre: str, a: Agregado, b: Agregado, fn) -> None:
    va = fn(a) if a.total_manos else "N/A"
    vb = fn(b) if b.total_manos else "N/A"
    try:
        delta = float(str(fn(b)).rstrip("%")) - float(str(fn(a)).rstrip("%")) \
            if "%" in str(fn(a)) else 0
    except (ValueError, TypeError):
        delta = 0
    delta_str = f"{delta:+.1f}" if isinstance(delta, float) else ""
    print(f"  {nombre:<28} {str(va):>14} {str(vb):>14} {delta_str:>10}")


def _imprimir_distribucion(agg: Agregado) -> None:
    """Imprime distribucion de puntuaciones."""
    print(f"\n  DISTRIBUCION DE PUNTOS POR MANO:")
    print(f"  {'Rango':<14} {'% Manos':>12}")
    print(f"  {'─'*14} {'─'*12}")

    rangos = [
        ("0 pts", lambda x: x == 0),
        ("1-3 pts", lambda x: 1 <= x <= 3),
        ("4-8 pts", lambda x: 4 <= x <= 8),
        ("9-13 pts", lambda x: 9 <= x <= 13),
        ("14-25 pts", lambda x: 14 <= x <= 25),
        ("26 pts", lambda x: x == 26),
    ]

    for nombre, cond in rangos:
        p = sum(1 for s in agg.scores if cond(s)) / max(agg.total_manos, 1)
        print(f"  {nombre:<14} {p:>11.1%}")


def _imprimir_por_seat(agg: Agregado, label: str) -> None:
    """Imprime desglose por posición (seat)."""
    print(f"\n  DESGLOSE POR POSICIÓN — {label}:")
    print(f"  {'Seat':<8} {'Manos':>8} {'Score Avg':>10} {'WR':>8} {'0-pt':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

    for seat in range(4):
        scores = agg.por_seat.get(seat, [])
        n = len(scores)
        if n == 0:
            continue
        avg = np.mean(scores)
        wr = sum(1 for s in scores if s <= 8) / n
        cero = sum(1 for s in scores if s == 0) / n
        print(f"  {seat:<8} {n:>8} {avg:>10.1f} {wr:>7.1%} {cero:>7.1%}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analisis detallado de modelo v3 con campo estandar")
    parser.add_argument("--modelo", default=_MODELO_DEFAULT,
                        help="Ruta al snapshot v3 (sin .zip)")
    parser.add_argument("--manos", type=int, default=200,
                        help="Manos a evaluar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu",
                        help="Dispositivo (cpu, cuda, dml)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar primeras 10 manos")
    args = parser.parse_args()

    print(f"\nCargando modelo v3: {args.modelo}")
    t0 = time.time()
    modelo, vecnorm_path = cargar_modelo_v3(args.modelo, args.device)
    print(f"   Modelo cargado en {time.time() - t0:.1f}s")
    print(f"   Obs dim: {modelo.observation_space.shape[0]}")

    print(f"\n{'─'*60}")
    print(f"EVALUANDO con campo estandar [Modelo, Experto, Experto, Bot]")
    print(f"({args.manos} manos)...")
    t0 = time.time()
    agg = evaluar_manos(
        modelo, vecnorm_path, args.manos, args.seed,
        verbose=args.verbose)
    print(f"   Completado en {time.time() - t0:.1f}s")

    imprimir_resultados(agg, args.manos, args.modelo)


if __name__ == "__main__":
    main()
