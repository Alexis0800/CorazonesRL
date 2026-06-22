"""
Dataset offline: (obs, todos_los_scores, accion_optima) con PIMC.

Mejoras sobre v1:
  - Oponentes con bots heuristicos (no aleatorio) → estados realistas
  - Guarda scores de TODAS las cartas legales → coste de cada error
  - Continuidad real entre bazas (mano completa, 13 bazas)
  - Mundos PIMC adaptativos: 10/20/40 segun baza

Uso:
    python scripts/_generar_dataset.py --manos 10000 --output datasets/pimc_v2
"""
from __future__ import annotations
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
from src.agentes.bot_experto import BotExperto
from src.v3.observacion import ObservacionBuilderV3, DIM_V3
from src.mcts.pimc import _puntaje_esperado_por_carta
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
import argparse
import os
import sys
import time
import random as _random
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


_PALO = {0: "T", 1: "D", 2: "P", 3: "C"}
_BOTS = [BotExperto(), bot_conservador, bot_agresivo, bot_evasivo]
_MAX_CARTAS = 52


def _situacion(motor, idx):
    if not motor.mesa:
        return "liderar"
    p = motor.palo_de_salida
    leg = motor.obtener_jugadas_legales(idx)
    return "seguir" if any(c.palo == p for c in leg) else "descartar"


def _mano_terminada(motor):
    return all(len(j.mano) == 0 for j in motor.jugadores) and len(motor.mesa) == 0


def _jugar_oponentes_con_bots(motor, agente_idx, bots_asignados):
    """Juega oponentes usando bots heuristicos asignados (ESTADOS REALISTAS)."""
    while not _mano_terminada(motor):
        idx = motor.obtener_jugador_actual()
        if idx == agente_idx:
            break
        leg = motor.obtener_jugadas_legales(idx)
        if not leg:
            break
        bot = bots_asignados.get(idx)
        if bot is None:
            motor.jugar_carta(idx, _random.choice(leg))
        elif isinstance(bot, BotExperto):
            motor.jugar_carta(idx, bot(motor, idx, leg))
        else:
            motor.jugar_carta(idx, bot(motor, idx, leg))
        if len(motor.mesa) == 4:
            motor.resolver_baza()


def generar_dataset(num_manos: int, seed: int, output_path: str):
    builder = ObservacionBuilderV3(dim=DIM_V3)
    rng = np.random.default_rng(seed)

    todas_obs = []
    todas_scores = []       # (N, 52) — score de cada carta (-1 si no legal)
    todas_bazas = []
    todas_situaciones = []

    print(
        f"Generando dataset: {num_manos} manos, PIMC adaptativo, oponentes con bots...")
    t0 = time.time()

    for mano_i in range(num_manos):
        motor = MotorCorazones()
        motor.repartir()
        agente = mano_i % 4

        # Asignar bots a oponentes (mezcla para diversidad)
        oponentes = [(agente + d) % 4 for d in (1, 2, 3)]
        bots_asignados = {}
        for op in oponentes:
            # 40% BotExperto, 20% c/u conservador/agresivo/evasivo
            r = _random.random()
            if r < 0.4:
                bots_asignados[op] = BotExperto()
            elif r < 0.6:
                bots_asignados[op] = bot_conservador
            elif r < 0.8:
                bots_asignados[op] = bot_agresivo
            else:
                bots_asignados[op] = bot_evasivo

        _jugar_oponentes_con_bots(motor, agente, bots_asignados)

        while not _mano_terminada(motor):
            idx = motor.obtener_jugador_actual()
            if idx != agente:
                _jugar_oponentes_con_bots(motor, agente, bots_asignados)
                if _mano_terminada(motor):
                    break
                idx = motor.obtener_jugador_actual()

            legales = motor.obtener_jugadas_legales(idx)
            if not legales:
                break

            obs = builder.construir(
                motor, agente,
                [set() for _ in range(4)], [0, 0, 0, 0],
                [0, 0, 0, 0], None,
            )

            if len(legales) >= 2:
                # Mundos PIMC adaptativos
                if motor.numero_baza <= 3:
                    mundos_pimc = 10   # bazas 1-3: alta incertidumbre, rapido
                elif motor.numero_baza <= 7:
                    mundos_pimc = 20   # bazas 4-7: incertidumbre media
                else:
                    mundos_pimc = 40   # bazas 8+: mayor precision

                scores_dict = _puntaje_esperado_por_carta(
                    motor, agente, legales, num_mundos=mundos_pimc, rng=rng,
                )

                # Guardar scores de TODAS las cartas (52-dim)
                score_vector = np.full(_MAX_CARTAS, -1.0, dtype=np.float32)
                for carta in legales:
                    score_vector[carta.id] = float(scores_dict[carta.id])

                todas_obs.append(obs)
                todas_scores.append(score_vector)
                todas_bazas.append(motor.numero_baza)
                todas_situaciones.append(_situacion(motor, agente))

                optima = min(legales, key=lambda c: scores_dict[c.id])
                motor.jugar_carta(idx, optima)
            elif len(legales) == 1:
                motor.jugar_carta(idx, legales[0])
            else:
                break

            if len(motor.mesa) == 4:
                motor.resolver_baza()

        if (mano_i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (mano_i + 1) / elapsed
            eta = (num_manos - mano_i - 1) / rate
            pct = (mano_i + 1) / num_manos * 100
            bar = "=" * int(pct / 3) + ">" + " " * (33 - int(pct / 3))
            print(f"\r  [{bar}] {pct:.0f}% | {mano_i+1:,}/{num_manos:,} manos | "
                  f"{len(todas_obs):,} muestras | {rate:.0f} m/s | ETA {eta:.0f}s",
                  end="", flush=True)

    elapsed = time.time() - t0
    n = len(todas_obs)
    print(f"\nDataset: {n:,} muestras en {elapsed:.0f}s ({elapsed/60:.1f}min)")

    obs_arr = np.stack(todas_obs).astype(np.float32)
    scores_arr = np.stack(todas_scores).astype(np.float32)
    baza_arr = np.array(todas_bazas, dtype=np.int32)
    sit_arr = np.array(todas_situaciones)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=obs_arr,
        all_scores=scores_arr,
        bazas=baza_arr,
        situaciones=sit_arr,
    )

    with open(output_path + "_meta.txt", "w") as f:
        f.write(f"manos: {num_manos}\nmuestras: {n}\nseed: {seed}\n")
        f.write(f"dim_obs: {DIM_V3}\n")
        f.write(f"oponentes: 40% Experto + 20% c/u conservador/agresivo/evasivo\n")
        f.write(f"pimc_mundos: 10/20/40 adaptativo\n")
        for b in range(1, 14):
            cnt = int(sum(1 for x in baza_arr if x == b))
            f.write(f"baza_{b}: {cnt}\n")
        for sit in ["liderar", "seguir", "descartar"]:
            cnt = int(sum(1 for x in sit_arr if x == sit))
            f.write(f"sit_{sit}: {cnt}\n")

    print(
        f"Guardado: {output_path}.npz ({os.path.getsize(output_path+'.npz')/1024/1024:.1f} MB)")

    print(f"\nDistribucion por baza:")
    for b in range(1, 14):
        cnt = sum(1 for x in baza_arr if x == b)
        bar = "=" * (cnt * 40 // max(1, n))
        print(f"  Baza {b:>2}: {cnt:>5} ({cnt/n*100:.0f}%) {bar}")

    # Score optimo medio por baza
    print(f"\nScore optimo medio por baza:")
    for b in range(1, 14):
        mask = baza_arr == b
        if mask.sum() > 0:
            opt_scores = scores_arr[mask].max(
                axis=1)  # mejor score por muestra
            opt_scores = opt_scores[opt_scores >= 0]
            if len(opt_scores) > 0:
                print(
                    f"  Baza {b:>2}: {opt_scores.mean():.2f} +/- {opt_scores.std():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manos", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="datasets/pimc_v2")
    args = parser.parse_args()
    generar_dataset(args.manos, args.seed, args.output)


if __name__ == "__main__":
    main()
