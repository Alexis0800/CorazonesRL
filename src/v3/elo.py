"""Torneo Elo para v3 — Duelos directos con 2 BotExpertos de filler.

Formato de cada enfrentamiento:
    Mesa: [A, B, BotExperto, BotExperto]
    - A y B rotan asientos entre manos para eliminar sesgo de posicion.
    - Los 2 BotExpertos son independientes (cada uno decide por su cuenta).
    - Metrica: en cada mano, A "gana" si score(A) < score(B).

Ventajas sobre el diseno anterior (1 vs 3 clones):
    - Comparacion directa y simetrica: A y B comparten la misma mesa.
    - Sin clonacion: cada BotExperto toma decisiones independientes.
    - Elo significativo: mide habilidad relativa real entre pares.

Uso:
    python -m src.v3.elo --directorio models/v3/snapshots --manos 50
"""

from __future__ import annotations

import os
import sys
import re
import argparse
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Asegurar que el proyecto esta en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

K_FACTOR: int = 32
ELO_INICIAL: int = 1500


# ------------------------------------------------------------------
# Jugador single-hand
# ------------------------------------------------------------------

class JugadorV3:
    """Juega UNA mano usando modelo v3 o bot heuristico."""

    def __init__(
        self,
        nombre: str,
        modelo_path: Optional[str] = None,
        es_bot: bool = False,
        es_experto: bool = False,
        bot_idx: int = 0,
    ):
        self.nombre = nombre
        self.modelo_path = modelo_path
        self.es_bot = es_bot
        self.es_experto = es_experto
        self.bot_idx = bot_idx
        self._modelo = None
        self._obs_mean = None
        self._obs_var = None

    def cargar(self) -> None:
        """Carga modelo y VecNormalize si no es bot."""
        if self.es_bot or self.es_experto:
            return
        if self._modelo is not None:
            return

        from sb3_contrib import MaskablePPO
        ruta = self.modelo_path
        if not ruta.endswith(".zip"):
            ruta += ".zip"
        self._modelo = MaskablePPO.load(ruta, device="cpu")

        # Cargar VecNormalize stats
        vn_path = ruta.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vn_path):
            with open(vn_path, "rb") as f:
                vn = pickle.load(f)
            self._obs_mean = vn.obs_rms.mean
            self._obs_var = vn.obs_rms.var


# ------------------------------------------------------------------
# Politicas de bot
# ------------------------------------------------------------------

def _crear_bot_politica(nombre: str) -> Any:
    """Crea una politica de bot por nombre."""
    if nombre == "experto":
        from src.agentes.bot_experto import BotExperto
        return BotExperto()
    elif nombre == "conservador":
        from src.agentes.heuristicos import bot_conservador
        return bot_conservador
    elif nombre == "agresivo":
        from src.agentes.heuristicos import bot_agresivo
        return bot_agresivo
    else:
        from src.agentes.heuristicos import bot_evasivo
        return bot_evasivo


# ------------------------------------------------------------------
# Duelo directo: A vs B con 2 BotExpertos de filler
# ------------------------------------------------------------------

def jugar_duelo(
    jugador_a: JugadorV3,
    jugador_b: JugadorV3,
    num_manos: int = 50,
    seed_base: int = 0,
) -> Dict[str, float]:
    """Juega N manos con mesa [A, B, Experto, Experto].

    En cada mano, A y B se comparan directamente en la misma mesa,
    con 2 BotExpertos independientes como fillers. Los asientos
    rotan entre manos para eliminar sesgo de posicion.

    Args:
        jugador_a: Jugador A.
        jugador_b: Jugador B.
        num_manos: Manos independientes a jugar.
        seed_base: Semilla base.

    Returns:
        Dict con:
          - pct_a_gana: fraccion de manos donde score(A) < score(B)
          - pct_b_gana: fraccion de manos donde score(B) < score(A)
          - pct_empate: fraccion de manos donde score(A) == score(B)
          - avg_score_a, avg_score_b: puntuacion promedio
          - num_manos: total de manos jugadas
    """
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
    from src.agentes.bot_experto import BotExperto
    from src.dominio.motor import MotorCorazones
    from src.dominio.carta import Carta

    # Cargar modelos si es necesario
    jugador_a.cargar()
    jugador_b.cargar()

    # ── Preparar politicas ──
    def _politica(jug: JugadorV3):
        if jug.es_experto:
            return BotExperto()
        elif jug.es_bot:
            bots_pool = [bot_conservador, bot_agresivo, bot_evasivo]
            return bots_pool[jug.bot_idx % 3]
        else:
            return None  # modelo → usar _elegir_carta_*

    pol_a = _politica(jugador_a)
    pol_b = _politica(jugador_b)
    pol_e1 = BotExperto()
    pol_e2 = BotExperto()

    # ── Jugar manos ──
    a_gana = 0
    b_gana = 0
    empates = 0
    scores_a: List[float] = []
    scores_b: List[float] = []

    for h in range(num_manos):
        seed = seed_base + h
        motor = MotorCorazones()
        motor.repartir()

        # Asignar asientos con rotacion (4 posiciones)
        # A y B ocupan 2 de las 4 posiciones, E1 y E2 las otras 2
        # Rotacion para eliminar sesgo: A toma (h % 4), B toma ((h+2) % 4)
        seat_a = h % 4
        seat_b = (h + 2) % 4  # opuesto a A en la mesa
        # Las 2 posiciones restantes para los Expertos
        seats_expertos = [s for s in range(4) if s not in (seat_a, seat_b)]

        # Mapear seat → politica
        politicas = {
            seat_a: (pol_a, jugador_a, "a"),
            seat_b: (pol_b, jugador_b, "b"),
            seats_expertos[0]: (pol_e1, None, "e1"),
            seats_expertos[1]: (pol_e2, None, "e2"),
        }

        # Jugar las 13 bazas
        for _ in range(13):
            for _ in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)
                pol, jug, tag = politicas[idx]

                if pol is not None:
                    carta = pol(motor, idx, legales)
                else:
                    carta = _elegir_carta_modelo_sin_historial(
                        jug, motor, idx, legales, seed)

                motor.jugar_carta(idx, carta)
            motor.resolver_baza()

        # Resultado de esta mano
        score_a_mano = motor.jugadores[seat_a].contar_puntos_bazas()
        score_b_mano = motor.jugadores[seat_b].contar_puntos_bazas()

        scores_a.append(float(score_a_mano))
        scores_b.append(float(score_b_mano))

        if score_a_mano < score_b_mano:
            a_gana += 1
        elif score_b_mano < score_a_mano:
            b_gana += 1
        else:
            empates += 1

    return {
        "num_manos": num_manos,
        "pct_a_gana": a_gana / num_manos,
        "pct_b_gana": b_gana / num_manos,
        "pct_empate": empates / num_manos,
        "avg_score_a": float(np.mean(scores_a)),
        "avg_score_b": float(np.mean(scores_b)),
    }


def _elegir_carta_modelo_sin_historial(
    jug: JugadorV3,
    motor,
    idx: int,
    legales,
    seed: int,
):
    """Elige carta usando el modelo, con puntuacion_historica=[0,0,0,0].

    CORREGIDO: El modelo v2/v3 fue entrenado sin contexto multi-mano,
    asi que la observacion debe tener puntuacion_historica cero.
    """
    from src.v3.observacion import ObservacionBuilderV3, DIM_V3
    from src.dominio.carta import Carta

    builder = ObservacionBuilderV3(dim=DIM_V3)
    obs_raw = builder.construir(
        motor, idx,
        vacios=[set() for _ in range(4)],
        puntuacion_historica=[0, 0, 0, 0],  # ← CORREGIDO: modelo single-hand
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


# ------------------------------------------------------------------
# Elo rating
# ------------------------------------------------------------------

def calcular_elo(
    resultados: List[Tuple[str, str, float]],
    elo_inicial: int = ELO_INICIAL,
    k: int = K_FACTOR,
) -> Dict[str, float]:
    """Calcula ratings Elo por minimos cuadrados.

    Args:
        resultados: Lista de (nombre_A, nombre_B, score_A).
          score_A = 1 si A gana todas las manos, 0 si B gana todas.
        elo_inicial: Rating inicial.
        k: Factor K.

    Returns:
        Dict {nombre: rating}.
    """
    nombres = sorted(set(
        [a for a, b, _ in resultados] + [b for a, b, _ in resultados]))
    ratings = {n: float(elo_inicial) for n in nombres}

    for _ in range(10):  # iteraciones de convergencia
        for a, b, score_a in resultados:
            ea = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += k * (score_a - ea)
            ratings[b] += k * ((1.0 - score_a) - (1.0 - ea))

    return ratings


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torneo Elo para v3 — Manos independientes",
    )
    parser.add_argument("--directorio", type=str, required=True,
                        help="Directorio con snapshots .zip de v2/v3.")
    parser.add_argument("--manos", type=int, default=50,
                        help="Manos independientes por enfrentamiento.")
    parser.add_argument("--incluir-bots", action="store_true", default=True,
                        help="Incluir bots heuristicos.")
    parser.add_argument("--incluir-experto", action="store_true", default=True,
                        help="Incluir BotExperto.")
    parser.add_argument("--output", type=str, default=None,
                        help="Archivo de salida para resultados.")

    args = parser.parse_args()

    # Listar snapshots
    snaps = []
    if os.path.isdir(args.directorio):
        for f in sorted(os.listdir(args.directorio)):
            if f.startswith("snapshot_") and f.endswith(".zip"):
                snaps.append(os.path.join(args.directorio, f))

    if not snaps:
        print("No se encontraron snapshots.")
        return

    print("=" * 60)
    print("  TORNEO ELO v3 — Manos independientes")
    print("=" * 60)
    print(f"  Snapshots: {len(snaps)}")
    print(f"  Manos por enfrentamiento: {args.manos}")
    print(f"  Incluir bots: {args.incluir_bots}")
    print(f"  Incluir BotExperto: {args.incluir_experto}")
    print("-" * 60)

    # Crear jugadores
    jugadores: Dict[str, JugadorV3] = {}

    for snap in snaps:
        match = re.search(r'snapshot_(\d+)', snap)
        nombre = match.group(0) if match else os.path.basename(snap)[:25]
        jugadores[nombre] = JugadorV3(nombre=nombre, modelo_path=snap)

    if args.incluir_bots:
        for i, bn in enumerate(["conservador", "agresivo", "evasivo"]):
            nombre = f"[BOT] {bn}"
            jugadores[nombre] = JugadorV3(
                nombre=nombre, es_bot=True, bot_idx=i)

    if args.incluir_experto:
        nombre = "[BOT] experto"
        jugadores[nombre] = JugadorV3(
            nombre=nombre, es_experto=True)

    nombres = list(jugadores.keys())

    # Round-robin: duelos directos A vs B con 2 Expertos de filler
    resultados: List[Tuple[str, str, float]] = []
    n = len(nombres)
    total_enfrentamientos = n * (n - 1) // 2
    count = 0

    print(f"\n  Jugando {total_enfrentamientos} duelos directos...")
    print(f"  Formato: [A, B, BotExperto, BotExperto] — asientos rotados\n")

    for i in range(n):
        for j in range(i + 1, n):
            count += 1
            a_name = nombres[i]
            b_name = nombres[j]
            a = jugadores[a_name]
            b = jugadores[b_name]

            print(f"\r  {count}/{total_enfrentamientos}: {a_name} vs {b_name}...",
                  end="", flush=True)

            res = jugar_duelo(a, b,
                              num_manos=args.manos,
                              seed_base=count * 1000)

            # score_a = fraccion de manos donde A supera a B
            score_a = res["pct_a_gana"] + 0.5 * res["pct_empate"]
            resultados.append((a_name, b_name, score_a))

            print(f"\r  {count}/{total_enfrentamientos}: {a_name} vs {b_name}: "
                  f"A={res['pct_a_gana']:.1%} B={res['pct_b_gana']:.1%} "
                  f"AvgA={res['avg_score_a']:.1f} AvgB={res['avg_score_b']:.1f}")

    # Calcular Elo
    print("\n  Calculando ratings Elo...")
    ratings = calcular_elo(resultados)

    # Ordenar
    ranking = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("  CLASIFICACION FINAL")
    print("=" * 60)
    print(f"  {'Pos':<4} {'Jugador':<35} {'Elo':>6}  {'Diff':>6}")
    print("  " + "-" * 55)

    baseline = ELO_INICIAL
    for pos, (nombre, rating) in enumerate(ranking, 1):
        diff = rating - baseline
        print(f"  {pos:<4} {nombre:<35} {rating:>6.0f}  {diff:>+6.0f}")
        baseline = rating

    print("=" * 60)

    # Guardar resultados a archivo si se solicitó
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  TORNEO ELO v3 — Manos independientes\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Snapshots: {len(snaps)}\n")
            f.write(f"  Manos por enfrentamiento: {args.manos}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  {'Pos':<4} {'Jugador':<35} {'Elo':>6}  {'Diff':>6}\n")
            f.write("  " + "-" * 55 + "\n")
            baseline_out = ELO_INICIAL
            for pos, (nombre, rating) in enumerate(ranking, 1):
                diff = rating - baseline_out
                f.write(
                    f"  {pos:<4} {nombre:<35} {rating:>6.0f}  {diff:>+6.0f}\n")
                baseline_out = rating
            f.write("=" * 60 + "\n")
        print(f"   Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
