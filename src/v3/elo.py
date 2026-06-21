"""
Torneo Elo para v3 — Por rondas independientes.

Juega N manos individuales (cada una desde cero, sin acumular puntuacion)
usando modelos v3 (single-hand) + bots + BotExperto. Calcula rating Elo
para medir progreso real.

A diferencia de la version original de v2_ronda, este torneo evalua
manos independientes (no partidas multi-mano), respetando que el modelo
fue entrenado con puntuacion_historica = [0,0,0,0].

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
# Evaluacion por manos independientes (CORREGIDO)
# ------------------------------------------------------------------

def jugar_manos_1v3(
    jugador_principal: JugadorV3,
    oponentes: List[JugadorV3],
    num_manos: int = 50,
    seed_base: int = 0,
) -> Dict[str, float]:
    """Juega N manos INDEPENDIENTES: 1 modelo vs 3 oponentes.

    Cada mano es desde cero: MotorCorazones fresco, sin puntuacion
    historica. Esto respeta el entrenamiento del modelo v2/v3 que
    siempre vio puntuacion_historica = [0,0,0,0].

    Args:
        jugador_principal: Modelo a evaluar.
        oponentes: 3 oponentes (bots o BotExperto).
        num_manos: Manos independientes a jugar.
        seed_base: Semilla base.

    Returns:
        Dict con pct_primero, pct_top2, pct_cuarto, avg_score, etc.
    """
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
    from src.dominio.motor import MotorCorazones
    from src.dominio.carta import Carta

    # Cargar modelo principal
    jugador_principal.cargar()

    posiciones: List[int] = []
    puntuaciones: List[int] = []

    # Preparar oponentes
    bots_pool = [bot_conservador, bot_agresivo, bot_evasivo]
    politicas_base: Dict[int, Any] = {}
    for i, op in enumerate(oponentes):
        idx = i + 1  # oponentes ocupan slots 1, 2, 3
        if op.es_experto:
            from src.agentes.bot_experto import BotExperto
            politicas_base[idx] = BotExperto()
        else:
            politicas_base[idx] = bots_pool[op.bot_idx % 3]

    for h in range(num_manos):
        seed = seed_base + h

        # ── Nueva mano desde cero ──
        motor = MotorCorazones()
        motor.repartir()

        # Jugar las 13 bazas
        for _ in range(13):
            for _ in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)

                if idx == 0:
                    # Modelo principal
                    carta = _elegir_carta_modelo_sin_historial(
                        jugador_principal, motor, 0, legales, seed)
                elif idx in politicas_base:
                    carta = politicas_base[idx](motor, idx, legales)
                else:
                    carta = legales[0]

                motor.jugar_carta(idx, carta)
            motor.resolver_baza()

        # Evaluar resultado de ESTA mano (sin acumular)
        mi_score = motor.jugadores[0].contar_puntos_bazas()
        rivales = [motor.jugadores[i].contar_puntos_bazas()
                   for i in range(1, 4)]
        todas = [mi_score] + rivales
        ranking = sorted(range(4), key=lambda i: todas[i])
        pos = ranking.index(0)  # 0 = mejor (menos puntos)
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
    from src.entorno.observacion import ObservacionBuilder
    from src.entorno.dimensiones import DIM_ENTRENAMIENTO
    from src.dominio.carta import Carta

    builder = ObservacionBuilder(dim=DIM_ENTRENAMIENTO)
    obs_raw = builder.construir(
        motor, idx,
        vacios=[set() for _ in range(4)],
        puntuacion_historica=[0, 0, 0, 0],  # ← CORREGIDO
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

    args = parser.parse_args()

    # Listar snapshots
    snaps = []
    if os.path.isdir(args.directorio):
        for f in sorted(os.listdir(args.directorio)):
            if f.startswith("snapshot_") and f.endswith(".zip"):
                snaps.append(os.path.join(args.directorio, f))

    # Limitar a los mas recientes
    snaps = snaps[-10:] if len(snaps) > 10 else snaps

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

            # A vs B: A como principal, B como 3 oponentes
            oponentes_b = [
                JugadorV3(nombre=b_name, es_bot=b.es_bot,
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
                  f"WR={score_a:.1%} | Avg={res['avg_score']:.1f}")

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


if __name__ == "__main__":
    main()
