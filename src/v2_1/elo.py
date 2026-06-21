"""
Torneo Elo para v2_1 — Manos independientes.

Evalúa snapshots v2_1 en manos individuales (cada una MotorCorazones fresco,
puntuacion_historica=[0,0,0,0]) contra bots + BotExperto.

Uso:
    python -m src.v2_1.elo --directorio models/v2_1/snapshots --manos 50 --incluir-bots --incluir-experto
"""

from __future__ import annotations

import os
import sys
import re
import argparse
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ELO_INICIAL: float = 1500.0
ELO_K: float = 32.0
ELO_ESCALA: float = 400.0


class JugadorV2:
    """Representa un jugador del torneo: snapshot v2 o bot heurístico."""

    def __init__(
        self,
        nombre: str,
        es_bot: bool = False,
        es_experto: bool = False,
        bot_idx: int = 0,
        modelo_path: Optional[str] = None,
    ) -> None:
        self.nombre = nombre
        self.es_bot = es_bot
        self.es_experto = es_experto
        self.bot_idx = bot_idx
        self.modelo_path = modelo_path
        self._modelo: Any = None
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_var: Optional[np.ndarray] = None

    def cargar(self) -> None:
        if self._modelo is not None:
            return
        if self.modelo_path is None:
            raise ValueError("Modelo sin modelo_path")
        import torch
        from sb3_contrib import MaskablePPO
        self._modelo = MaskablePPO.load(self.modelo_path)

        vn_path = self.modelo_path.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vn_path):
            with open(vn_path, "rb") as f:
                vn = pickle.load(f)
            self._obs_mean = vn.obs_rms.mean.copy()
            self._obs_var = vn.obs_rms.var.copy()
        else:
            self._obs_mean = None
            self._obs_var = None


# ------------------------------------------------------------------
# Elo
# ------------------------------------------------------------------

def calcular_elo(
    resultados: List[Tuple[str, str, float]],
    inicial: float = ELO_INICIAL,
    k: float = ELO_K,
    escala: float = ELO_ESCALA,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Calcula ratings Elo por el método iterativo estándar (Elo clásico)."""
    jugadores_set = set()
    for a, b, _ in resultados:
        jugadores_set.add(a)
        jugadores_set.add(b)
    jugadores = sorted(jugadores_set)

    idx_map = {j: i for i, j in enumerate(jugadores)}
    n = len(jugadores)
    ratings = np.full(n, inicial, dtype=np.float64)

    for _ in range(max_iter):
        delta_max = 0.0
        for a, b, score_a in resultados:
            i, j = idx_map[a], idx_map[b]
            diff = ratings[j] - ratings[i]
            expected_a = 1.0 / (1.0 + 10.0 ** (diff / escala))
            delta = k * (score_a - expected_a)
            ratings[i] += delta
            ratings[j] -= delta
            delta_max = max(delta_max, abs(delta))
        if delta_max < tol:
            break

    return {j: float(r) for j, r in zip(jugadores, ratings)}


# ------------------------------------------------------------------
# Juego de manos independientes
# ------------------------------------------------------------------

def _elegir_carta_modelo_sin_historial(
    jug: JugadorV2,
    motor,
    idx: int,
    legales,
    seed: int,
):
    """Elige carta con modelo v2_1, puntuacion_historica=[0,0,0,0]."""
    from src.entorno.observacion import ObservacionBuilder
    from src.entorno.dimensiones import DIM_ENTRENAMIENTO
    from src.dominio.carta import Carta

    builder = ObservacionBuilder(dim=DIM_ENTRENAMIENTO)
    obs_raw = builder.construir(
        motor, idx,
        vacios=[set() for _ in range(4)],
        puntuacion_historica=[0, 0, 0, 0],
        puntos_mano_actual=[j.contar_puntos_bazas() for j in motor.jugadores],
        dama_picas_en=None,
    )

    jug.cargar()
    if jug._obs_mean is not None:
        obs = np.clip(
            (obs_raw - jug._obs_mean) /
            (np.sqrt(jug._obs_var) + 1e-8), -10, 10,
        )
    else:
        obs = obs_raw

    mask = np.zeros(52, dtype=np.bool_)
    for c in legales:
        mask[c.id] = True

    action, _ = jug._modelo.predict(
        obs, action_masks=mask, deterministic=True,
    )
    return Carta._TODAS[int(action)]


def jugar_manos_1v3(
    jugador_principal: JugadorV2,
    oponentes: List[JugadorV2],
    num_manos: int = 50,
    seed_base: int = 0,
) -> Dict[str, float]:
    """Juega N manos INDEPENDIENTES: 1 modelo vs 3 oponentes."""
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
    from src.dominio.motor import MotorCorazones
    from src.dominio.carta import Carta

    bots_pool = [bot_conservador, bot_agresivo, bot_evasivo]

    # Cargar modelo principal (si lo tiene)
    if not jugador_principal.es_bot and not jugador_principal.es_experto:
        jugador_principal.cargar()

    # Determinar política del jugador principal
    politica_principal = None
    if jugador_principal.es_experto:
        from src.agentes.bot_experto import BotExperto
        politica_principal = BotExperto()
    elif jugador_principal.es_bot:
        politica_principal = bots_pool[jugador_principal.bot_idx % 3]

    posiciones: List[int] = []
    puntuaciones: List[int] = []

    # Preparar oponentes
    politicas_base: Dict[int, Any] = {}
    for i, op in enumerate(oponentes):
        idx = i + 1
        if op.es_experto:
            from src.agentes.bot_experto import BotExperto
            politicas_base[idx] = BotExperto()
        else:
            politicas_base[idx] = bots_pool[op.bot_idx % 3]

    for h in range(num_manos):
        seed = seed_base + h
        motor = MotorCorazones()
        motor.repartir()

        for _ in range(13):
            for _ in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)

                if idx == 0:
                    if politica_principal is not None:
                        carta = politica_principal(motor, 0, legales)
                    else:
                        carta = _elegir_carta_modelo_sin_historial(
                            jugador_principal, motor, 0, legales, seed)
                elif idx in politicas_base:
                    carta = politicas_base[idx](motor, idx, legales)
                else:
                    carta = legales[0]

                motor.jugar_carta(idx, carta)
            motor.resolver_baza()

        mi_score = motor.jugadores[0].contar_puntos_bazas()
        rivales = [motor.jugadores[i].contar_puntos_bazas()
                   for i in range(1, 4)]
        todas = [mi_score] + rivales
        ranking = sorted(range(4), key=lambda i: todas[i])
        pos = ranking.index(0)
        posiciones.append(pos)
        puntuaciones.append(mi_score)

    arr = np.array(puntuaciones, dtype=np.float64)
    return {
        "num_manos": num_manos,
        "pct_primero": sum(1 for p in posiciones if p == 0) / num_manos,
        "pct_top2": sum(1 for p in posiciones if p in (0, 1)) / num_manos,
        "pct_cuarto": sum(1 for p in posiciones if p == 3) / num_manos,
        "avg_score": float(np.mean(arr)),
        "median_score": float(np.median(arr)),
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torneo Elo para v2_1 — Manos independientes",
    )
    parser.add_argument("--directorio", type=str, required=True,
                        help="Directorio con snapshots .zip de v2_1.")
    parser.add_argument("--manos", type=int, default=50,
                        help="Manos independientes por enfrentamiento.")
    parser.add_argument("--incluir-bots", action="store_true", default=True,
                        help="Incluir bots heurísticos.")
    parser.add_argument("--incluir-experto", action="store_true", default=True,
                        help="Incluir BotExperto.")

    args = parser.parse_args()

    # Listar snapshots
    snaps = []
    if os.path.isdir(args.directorio):
        for f in sorted(os.listdir(args.directorio)):
            if f.startswith("snapshot_") and f.endswith(".zip"):
                snaps.append(os.path.join(args.directorio, f))

    # Tomar todos los snapshots (el entrenamiento generó ~28)
    # snapshots más antiguos primero, todos incluidos para ver la evolución

    if not snaps:
        print("No se encontraron snapshots.")
        return

    print("=" * 60)
    print("  TORNEO ELO v2_1 (enhanced tactical rewards)")
    print("=" * 60)
    print(f"  Snapshots: {len(snaps)}")
    print(f"  Manos por enfrentamiento: {args.manos}")
    print(f"  Incluir bots: {args.incluir_bots}")
    print(f"  Incluir BotExperto: {args.incluir_experto}")
    print("-" * 60)

    # Crear jugadores
    jugadores: Dict[str, JugadorV2] = {}

    for snap in snaps:
        nombre = os.path.splitext(os.path.basename(snap))[0]
        jugadores[nombre] = JugadorV2(
            nombre=nombre,
            modelo_path=snap,
        )

    # Bots
    if args.incluir_bots:
        for nombre, idx in [("conservador", 0), ("agresivo", 1), ("evasivo", 2)]:
            jugadores[f"[BOT] {nombre}"] = JugadorV2(
                nombre=f"[BOT] {nombre}",
                es_bot=True,
                bot_idx=idx,
            )

    if args.incluir_experto:
        nombre = "[BOT] experto"
        jugadores[nombre] = JugadorV2(
            nombre=nombre, es_experto=True)

    nombres = list(jugadores.keys())

    # Round-robin: 1v3
    resultados: List[Tuple[str, str, float]] = []
    n = len(nombres)
    total_enfrentamientos = n * (n - 1) // 2
    count = 0

    print(f"\n  Jugando {total_enfrentamientos} enfrentamientos...")
    for i in range(n):
        for j in range(i + 1, n):
            count += 1
            a_name = nombres[i]
            b_name = nombres[j]
            a = jugadores[a_name]
            b = jugadores[b_name]

            oponentes_b = [
                JugadorV2(nombre=b_name, es_bot=b.es_bot,
                          es_experto=b.es_experto, bot_idx=b.bot_idx,
                          modelo_path=b.modelo_path)
                for _ in range(3)
            ]

            print(f"\r  {count}/{total_enfrentamientos}: {a_name} vs {b_name}...",
                  end="", flush=True)

            res = jugar_manos_1v3(a, oponentes_b,
                                  num_manos=args.manos,
                                  seed_base=count * 1000)

            score_a = res["pct_primero"]
            resultados.append((a_name, b_name, score_a))

            print(f"\r  {count}/{total_enfrentamientos}: {a_name} vs {b_name}: "
                  f"WR={score_a:.1%} | Avg={res['avg_score']:.0f}")

    # Calcular Elo
    print("\n  Calculando ratings Elo...")
    ratings = calcular_elo(resultados)

    # Ordenar
    ranking = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("  CLASIFICACIÓN FINAL")
    print("=" * 60)
    print(f"  {'Pos':<4} {'Jugador':<35} {'Elo':>6}  {'Diff':>6}")
    print("  " + "-" * 55)

    baseline = ELO_INICIAL
    for pos, (nombre, rating) in enumerate(ranking, 1):
        diff = rating - baseline
        print(f"  {pos:<4} {nombre:<35} {rating:>6.0f}  {diff:>+6.0f}")
        baseline = rating

    print("=" * 60)


if __name__ == "__main__":
    main()
