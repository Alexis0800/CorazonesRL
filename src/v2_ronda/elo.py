"""
Torneo Elo para v2_ronda — Multi-hand, score-based models.

Juega partidas completas de Corazones (multiple manos hasta 100 puntos)
usando modelos v2 (single-hand) + bots + BotExperto. Calcula rating Elo
para medir progreso real.

Formula Elo:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    R_A' = R_A + K * (S_A - E_A)

Uso:
    python -m src.v2_ronda.elo --directorio models/v2_rl/snapshots --partidas 30
"""

from __future__ import annotations

import os
import sys
import re
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Asegurar que el proyecto esta en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

K_FACTOR: int = 32
ELO_INICIAL: int = 1500


# ------------------------------------------------------------------
# Jugador multi-mano
# ------------------------------------------------------------------

class JugadorV2:
    """Juega una mano usando modelo v2 o bot heuristico."""

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
        """Carga modelo v2 y VecNormalize si no es bot."""
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
            import pickle
            with open(vn_path, "rb") as f:
                vn = pickle.load(f)
            self._obs_mean = vn.obs_rms.mean
            self._obs_var = vn.obs_rms.var

    def jugar_mano(
        self,
        agente_idx: int,
        politicas_oponentes: Dict[int, Any],
        seed: int,
    ) -> int:
        """Juega UNA mano como agente_idx. Retorna puntos tomados."""
        from src.v2_ronda.entorno import CorazonesEnvSingleHand

        env = CorazonesEnvSingleHand(
            agente_idx=agente_idx,
            politicas_oponentes=politicas_oponentes,
        )
        obs_raw, _ = env.reset(seed=seed)

        if self._obs_mean is not None:
            obs = np.clip(
                (obs_raw - self._obs_mean) /
                (np.sqrt(self._obs_var) + 1e-8), -10, 10,
            )
        else:
            obs = obs_raw

        terminated = False
        while not terminated:
            mask = env.action_masks()
            action, _ = self._modelo.predict(
                obs, action_masks=mask, deterministic=True,
            )
            obs_raw, _, terminated, truncated, _ = env.step(int(action))
            if self._obs_mean is not None:
                obs = np.clip(
                    (obs_raw - self._obs_mean) /
                    (np.sqrt(self._obs_var) + 1e-8), -10, 10,
                )
            else:
                obs = obs_raw

        return env._puntos_mano_actual[agente_idx]


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
# Partida multi-mano
# ------------------------------------------------------------------

def jugar_partida(
    jugadores: List[JugadorV2],
    seed: int = 0,
) -> List[int]:
    """Juega UNA partida multi-mano (hasta 100 puntos).

    Args:
        jugadores: 4 jugadores (modelos o bots).
        seed: Semilla.

    Returns:
        Puntuacion final de cada jugador [p0, p1, p2, p3].
    """
    puntuacion = [0, 0, 0, 0]
    mano_seed = seed

    while max(puntuacion) < 100:
        # Determinar politicas para esta mano
        politicas: Dict[int, Any] = {}
        agentes_modelo: List[Tuple[int, JugadorV2]] = []

        for i, jug in enumerate(jugadores):
            if jug.es_bot or jug.es_experto:
                nombre = (
                    "experto" if jug.es_experto
                    else ["conservador", "agresivo", "evasivo"][jug.bot_idx % 3]
                )
                politicas[i] = _crear_bot_politica(nombre)
            else:
                agentes_modelo.append((i, jug))

        # Si no hay modelos, usar bots para todos
        if not agentes_modelo:
            # Partida solo bots: simular rapido con motor
            from src.dominio.motor import MotorCorazones
            from src.agentes.heuristicos import (
                bot_conservador, bot_agresivo, bot_evasivo,
            )
            bots_pool = [bot_conservador, bot_agresivo, bot_evasivo]
            motor = MotorCorazones()
            while max(puntuacion) < 100:
                motor.repartir()
                motor.jugar_mano(lambda m, idx, leg: bots_pool[idx % 3](m, idx, leg))
                puntuaciones = motor.aplicar_puntuacion()
                for i in range(4):
                    puntuacion[i] = motor.jugadores[i].puntuacion_historica
                mano_seed += 1
            return puntuacion

        # Modelos juegan como agentes (uno a la vez)
        # Turnamos: en cada mano, un modelo diferente es el "agente"
        mano_idx = mano_seed % len(agentes_modelo)
        agente_idx, agente_jug = agentes_modelo[mano_idx]

        # Construir politicas para esta mano
        poli_mano = dict(politicas)
        for ai, aj in agentes_modelo:
            if ai != agente_idx:
                # Otro modelo actua como oponente (usa su propia politica)
                poli_mano[ai] = _crear_politica_modelo_oponente(aj)

        pts = agente_jug.jugar_mano(agente_idx, poli_mano, mano_seed)

        # Acumular: solo el agente_modelo recibe sus puntos reales
        # Los demas reciben los puntos que reporto el entorno
        # Simplificacion: todos reciben los puntos que el entorno calculo
        # Para partidas multi-modelo, cada modelo juega una mano como agente
        # y acumulamos sus puntos

        # Avanzar puntuacion
        for i in range(4):
            if i in poli_mano:
                # Usar el entorno del agente para obtener puntos de todos
                pass

        # Simplificacion mayor: asignar pts al agente, y rotar
        puntuacion[agente_idx] += pts
        mano_seed += 1

        # Para que todos los modelos acumulen puntos proporcionalmente,
        # alternamos quien es el agente cada mano
        # Pero necesitamos los puntos de TODOS los jugadores.
        # Solucion: jugar una mano con el primer modelo como agente,
        # y tomar los puntos de todos del entorno.

    return puntuacion


def _crear_politica_modelo_oponente(jug: JugadorV2) -> Any:
    """Crea un callable que actua como oponente usando el modelo v2."""
    jug.cargar()

    def politica(motor, idx, legales, obs=None, deterministic=False):
        """Politica de oponente: usa el modelo v2 para elegir carta."""
        from src.entorno.observacion import ObservacionBuilder
        from src.entorno.dimensiones import DIM_ENTRENAMIENTO

        builder = ObservacionBuilder(dim=DIM_ENTRENAMIENTO)
        obs_raw = builder.construir(
            motor, idx,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[j.puntuacion_historica for j in motor.jugadores],
            puntos_mano_actual=[j.contar_puntos_bazas() for j in motor.jugadores],
            dama_picas_en=None,
        )

        if jug._obs_mean is not None:
            obs_norm = np.clip(
                (obs_raw - jug._obs_mean) /
                (np.sqrt(jug._obs_var) + 1e-8), -10, 10,
            )
        else:
            obs_norm = obs_raw

        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True

        action, _ = jug._modelo.predict(
            obs_norm, action_masks=mask, deterministic=True,
        )
        from src.dominio.carta import Carta
        return Carta._TODAS[int(action)]

    return politica


# ------------------------------------------------------------------
# Partida simplificada (1 modelo vs 3 bots/experto)
# ------------------------------------------------------------------

def jugar_partida_1v3(
    jugador_principal: JugadorV2,
    oponentes: List[JugadorV2],
    num_partidas: int = 30,
    seed_base: int = 0,
) -> Dict[str, float]:
    """Juega N partidas: 1 modelo vs 3 oponentes (bots/experto).

    Args:
        jugador_principal: Modelo v2 a evaluar.
        oponentes: 3 oponentes (bots o BotExperto).
        num_partidas: Partidas a jugar.
        seed_base: Semilla base.

    Returns:
        Dict con win_rate, avg_score, posiciones.
    """
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo

    # Cargar modelo principal
    jugador_principal.cargar()

    posiciones: List[int] = []
    puntuaciones: List[int] = []

    for p in range(num_partidas):
        seed = seed_base + p
        puntuacion = [0, 0, 0, 0]
        mano_seed = seed

        # Preparar oponentes
        bots_pool = [bot_conservador, bot_agresivo, bot_evasivo]
        politicas_base: Dict[int, Any] = {}
        for i, op in enumerate(oponentes):
            idx = i + 1
            if op.es_experto:
                from src.agentes.bot_experto import BotExperto
                politicas_base[idx] = BotExperto()
            else:
                politicas_base[idx] = bots_pool[op.bot_idx % 3]

        # Crear motor persistente para toda la partida
        from src.dominio.motor import MotorCorazones
        motor = MotorCorazones()

        while max(puntuacion) < 100:
            motor.repartir()

            # Jugar la mano con politicas para todos
            poli_mano = dict(politicas_base)
            for _ in range(13):
                for _ in range(4):
                    idx = motor.obtener_jugador_actual()
                    legales = motor.obtener_jugadas_legales(idx)
                    if idx == 0:
                        # Modelo principal
                        carta = _elegir_carta_modelo(
                            jugador_principal, motor, 0, legales, mano_seed)
                    elif idx in poli_mano:
                        if callable(poli_mano[idx]):
                            carta = poli_mano[idx](motor, idx, legales)
                        else:
                            carta = legales[0]
                    else:
                        carta = legales[0]
                    motor.jugar_carta(idx, carta)
                motor.resolver_baza()

            puntuaciones_mano = motor.aplicar_puntuacion()
            for i in range(4):
                puntuacion[i] = motor.jugadores[i].puntuacion_historica
            mano_seed += 1

        # Determinar posicion (0=mejor)
        ranking = sorted(range(4), key=lambda i: puntuacion[i])
        pos = ranking.index(0)
        posiciones.append(pos)
        puntuaciones.append(puntuacion[0])

    arr = np.array(puntuaciones, dtype=np.float64)
    return {
        "num_partidas": num_partidas,
        "pct_primero": sum(1 for p in posiciones if p == 0) / num_partidas,
        "pct_top2": sum(1 for p in posiciones if p in (0, 1)) / num_partidas,
        "pct_cuarto": sum(1 for p in posiciones if p == 3) / num_partidas,
        "avg_score": float(np.mean(arr)),
        "median_score": float(np.median(arr)),
    }


def _elegir_carta_modelo(
    jug: JugadorV2,
    motor,
    idx: int,
    legales,
    seed: int,
):
    """Elige carta usando el modelo v2 (como oponente en motor)."""
    from src.entorno.observacion import ObservacionBuilder
    from src.entorno.dimensiones import DIM_ENTRENAMIENTO
    from src.dominio.carta import Carta

    builder = ObservacionBuilder(dim=DIM_ENTRENAMIENTO)
    obs_raw = builder.construir(
        motor, idx,
        vacios=[set() for _ in range(4)],
        puntuacion_historica=[j.puntuacion_historica for j in motor.jugadores],
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
          score_A = 1 si A gana, 0.5 empate, 0 si B gana.
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
        description="Torneo Elo para v2_ronda — Multi-hand",
    )
    parser.add_argument("--directorio", type=str, required=True,
                        help="Directorio con snapshots .zip de v2.")
    parser.add_argument("--partidas", type=int, default=30,
                        help="Partidas por enfrentamiento.")
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
    print("  TORNEO ELO v2_ronda")
    print("=" * 60)
    print(f"  Snapshots: {len(snaps)}")
    print(f"  Partidas por enfrentamiento: {args.partidas}")
    print(f"  Incluir bots: {args.incluir_bots}")
    print(f"  Incluir BotExperto: {args.incluir_experto}")
    print("-" * 60)

    # Crear jugadores
    jugadores: Dict[str, JugadorV2] = {}

    for snap in snaps:
        match = re.search(r'snapshot_(\d+)', snap)
        nombre = match.group(0) if match else os.path.basename(snap)[:25]
        jugadores[nombre] = JugadorV2(nombre=nombre, modelo_path=snap)

    if args.incluir_bots:
        for i, bn in enumerate(["conservador", "agresivo", "evasivo"]):
            nombre = f"[BOT] {bn}"
            jugadores[nombre] = JugadorV2(
                nombre=nombre, es_bot=True, bot_idx=i)

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

            # A vs B: A juega como principal, B como oponente (3 copias)
            oponentes_b = [
                JugadorV2(nombre=b_name, es_bot=b.es_bot,
                          es_experto=b.es_experto, bot_idx=b.bot_idx,
                          modelo_path=b.modelo_path)
                for _ in range(3)
            ]

            print(f"\r  {count}/{total_enfrentamientos}: {a_name} vs {b_name}...",
                  end="", flush=True)

            res = jugar_partida_1v3(a, oponentes_b,
                                    num_partidas=args.partidas,
                                    seed_base=count * 1000)

            # Score: fraccion de victorias de A sobre B
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
