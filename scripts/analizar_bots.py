#!/usr/bin/env python
"""
Análisis verbose de partidas bot vs bot.

Ejecuta partidas entre los bots heurísticos mostrando cada decisión baza a baza.
Útil para detectar errores sistemáticos antes de construir el Bot Experto (Fase 2).

Uso:
    python scripts/analizar_bots.py                  # 10 partidas, c/a/e/c
    python scripts/analizar_bots.py --partidas 5
    python scripts/analizar_bots.py --resumen        # solo estadísticas
    python scripts/analizar_bots.py --bots c c c c   # 4 conservadores
"""

import sys
import os
import argparse
import random
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo

_PALO_STR = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
_VALOR_STR = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
               9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}

BOTS_MAP = {
    "c": ("conservador", bot_conservador),
    "a": ("agresivo", bot_agresivo),
    "e": ("evasivo", bot_evasivo),
}


def _c(carta: Carta) -> str:
    return f"{_VALOR_STR[carta.valor]}{_PALO_STR[carta.palo]}"


def _mano_str(mano: List[Carta]) -> str:
    por_palo: Dict[int, List[Carta]] = {0: [], 1: [], 2: [], 3: []}
    for c in mano:
        por_palo[c.palo].append(c)
    partes = []
    for palo in [0, 1, 2, 3]:
        cs = sorted(por_palo[palo], key=lambda c: c.valor, reverse=True)
        if cs:
            partes.append(_PALO_STR[palo] + "".join(_VALOR_STR[c.valor] for c in cs))
    return " ".join(partes)


def _detectar_error(motor: MotorCorazones, idx: int, carta: Carta,
                    legales: List[Carta]) -> str:
    """Detecta jugadas obviamente malas. Retorna descripción o cadena vacía."""

    # Error: jugar Q♠ siendo void en el palo de salida cuando hay otras opciones
    if carta.es_dama_de_picas and motor.mesa:
        palo_salida = motor.palo_de_salida
        es_void = all(c.palo != palo_salida for c in legales)
        if es_void:
            otras = [c for c in legales if not c.es_dama_de_picas]
            if otras:
                return f"Juega Q♠ pudiendo descartar {_c(otras[0])}"

    # Duda: ganar baza con puntos sin necesidad (hay carta más baja del mismo palo)
    if motor.mesa and carta.palo == motor.palo_de_salida:
        puntos_mesa = sum(c.puntos for _, c in motor.mesa)
        if puntos_mesa > 0:
            carta_alta_actual = max(
                (c for _, c in motor.mesa if c.palo == motor.palo_de_salida),
                key=lambda c: c.valor, default=None,
            )
            menores = [c for c in legales
                       if c.palo == motor.palo_de_salida and c.valor < carta.valor]
            mano_jugador = motor.jugadores[idx].mano
            corazones_en_mano = sum(1 for c in mano_jugador if c.es_corazon)
            tiene_q = any(c.es_dama_de_picas for c in mano_jugador)
            pozo_posible = corazones_en_mano >= 5 and not motor.corazones_rotos

            if (carta_alta_actual is not None
                    and carta.valor > carta_alta_actual.valor
                    and menores
                    and not pozo_posible):
                return (f"Gana baza con {puntos_mesa}pts jugando {_c(carta)} "
                        f"(podría jugar {_c(min(menores, key=lambda c: c.valor))})")

    # Error: liderar corazones cuando no están rotos y hay otras opciones
    if not motor.mesa and carta.es_corazon and not motor.corazones_rotos:
        otras = [c for c in legales if not c.es_corazon]
        if otras:
            return f"Lidera {_c(carta)} sin corazones rotos"

    return ""


def simular_partida(
    bots: List[Tuple[str, callable]],
    seed: int,
    verbose: bool = True,
) -> Dict:
    """Simula una partida completa (múltiples manos hasta 100 puntos).

    Returns:
        Dict con puntuaciones, ganador, y lista de errores detectados.
    """
    random.seed(seed)
    motor = MotorCorazones()
    nombres = [b[0] for b in bots]
    funcs = [b[1] for b in bots]
    errores: List[str] = []
    mano_num = 0

    if verbose:
        print(f"\n{'═' * 65}")
        print(f"PARTIDA #{seed} | {' | '.join(f'J{i}={n}' for i, n in enumerate(nombres))}")

    while all(j.puntuacion_historica < 100 for j in motor.jugadores):
        mano_num += 1

        # Limpiar bazas antes del nuevo reparto
        for j in motor.jugadores:
            j.bazas_ganadas = []

        motor.repartir()

        if verbose:
            print(f"\n  ── Mano #{mano_num} {'─' * 46}")
            for i in range(4):
                print(f"    J{i} ({nombres[i]}): {_mano_str(motor.jugadores[i].mano)}")
            print()

        # 13 bazas por mano
        for baza_num in range(1, 14):
            plays: List[Tuple[int, Carta, str]] = []

            for _ in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)
                carta_elegida = funcs[idx](motor, idx, legales)
                error = _detectar_error(motor, idx, carta_elegida, legales)
                motor.jugar_carta(idx, carta_elegida)
                plays.append((idx, carta_elegida, error))

            ganador = motor.resolver_baza()
            puntos_baza = sum(c.puntos for _, c in motor.mesa or
                              [(0, p[1]) for p in plays])
            # mesa ya fue limpiada por resolver_baza, reconstruir desde plays
            puntos_baza = sum(p[1].puntos for p in plays)
            cartas_str = " ".join(
                f"J{p[0]}:{_c(p[1])}" + ("!" if p[2] else "") for p in plays)

            if verbose:
                q_tag = " Q♠" if any(p[1].es_dama_de_picas for p in plays) else ""
                h_pts = puntos_baza - (13 if any(p[1].es_dama_de_picas for p in plays) else 0)
                h_tag = f" ♥×{h_pts}" if h_pts > 0 else ""
                print(f"    B{baza_num:02d} [{cartas_str}]  → J{ganador}({nombres[ganador]})"
                      f" [{puntos_baza}pts{q_tag}{h_tag}]")
                for idx, carta, error in plays:
                    if error:
                        print(f"         ⚠ J{idx}({nombres[idx]}): {error}")
                        errores.append(
                            f"P#{seed}/M{mano_num}/B{baza_num} "
                            f"J{idx}({nombres[idx]}): {error}")

        # Scoring de la mano
        puntos_mano = motor.calcular_puntuacion_mano()
        motor.aplicar_puntuacion()

        if verbose:
            pleno = next((i for i, p in enumerate(puntos_mano) if p == 0
                          and sum(puntos_mano) == 26), None)
            if pleno is not None and puntos_mano[pleno] == 0:
                print(f"\n    PLENO de J{pleno}({nombres[pleno]})!")
            pts_str = " | ".join(
                f"J{i}({nombres[i]}): {puntos_mano[i]}pts" for i in range(4))
            acum_str = " | ".join(
                f"J{i}: {motor.jugadores[i].puntuacion_historica}" for i in range(4))
            print(f"\n    Pts mano: {pts_str}")
            print(f"    Acum:     {acum_str}")

    puntuaciones = [motor.jugadores[i].puntuacion_historica for i in range(4)]
    ranking = sorted(range(4), key=lambda i: puntuaciones[i])

    if verbose:
        print(f"\n  RESULTADO FINAL:")
        for pos, idx in enumerate(ranking):
            print(f"    {pos+1}º J{idx} ({nombres[idx]}): {puntuaciones[idx]} pts")
        print(f"{'═' * 65}")

    return {
        "seed": seed,
        "puntuaciones": puntuaciones,
        "ranking": ranking,
        "ganador_idx": ranking[0],
        "nombre_ganador": nombres[ranking[0]],
        "errores": errores,
        "num_manos": mano_num,
    }


def imprimir_estadisticas(resultados: List[Dict], nombres: List[str]) -> None:
    n = len(resultados)
    victorias = [0] * 4
    top2 = [0] * 4
    puntos_acum = [0] * 4
    total_errores = [0] * 4

    for r in resultados:
        for pos, idx in enumerate(r["ranking"]):
            puntos_acum[idx] += r["puntuaciones"][idx]
            if pos == 0:
                victorias[idx] += 1
            if pos <= 1:
                top2[idx] += 1
        for err in r["errores"]:
            # "P#N/MN/BN J{idx}(nombre): ..."
            for i in range(4):
                if f"J{i}(" in err.split(":")[0]:
                    total_errores[i] += 1
                    break

    print(f"\n{'═' * 72}")
    print(f"ESTADÍSTICAS AGREGADAS ({n} partidas)")
    print(f"{'─' * 72}")
    print(f"  {'Bot':<20} {'1º':>6} {'Top-2':>7} {'Pts/partida':>12} "
          f"{'Errores':>9} {'Err/partida':>12}")
    print(f"{'─' * 72}")
    for i in range(4):
        pct1 = victorias[i] / n
        pct2 = top2[i] / n
        pts_avg = puntos_acum[i] / n
        err_avg = total_errores[i] / n
        print(f"  J{i} {nombres[i]:<17} {pct1:>5.1%} {pct2:>7.1%} "
              f"{pts_avg:>11.1f} {total_errores[i]:>9} {err_avg:>11.1f}")
    print(f"{'─' * 72}")

    total_errores_global = sum(e for r in resultados for e in [len(r["errores"])])
    if total_errores_global > 0:
        print(f"\nERRORES DETECTADOS ({total_errores_global} total):")
        tipos: Dict[str, int] = {}
        for r in resultados:
            for e in r["errores"]:
                tipo = e.split("): ")[1] if "): " in e else e
                # Normalizar tipo de error (primera parte antes de "pudiendo" / "sin")
                tipo_corto = tipo.split(" pudiendo")[0].split(" sin ")[0]
                tipos[tipo_corto] = tipos.get(tipo_corto, 0) + 1
        for tipo, cnt in sorted(tipos.items(), key=lambda x: -x[1]):
            print(f"  × {cnt:3d}  {tipo}")
    print(f"{'═' * 72}")


def main():
    parser = argparse.ArgumentParser(description="Análisis verbose de bots Corazones")
    parser.add_argument("--partidas", type=int, default=10)
    parser.add_argument("--bots", type=str, nargs=4, default=["c", "a", "e", "c"],
                        metavar=("J0", "J1", "J2", "J3"),
                        help="c=conservador, a=agresivo, e=evasivo")
    parser.add_argument("--resumen", action="store_true",
                        help="Solo estadísticas, sin detalle de bazas")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bots: List[Tuple[str, callable]] = []
    for b in args.bots:
        key = b[0].lower()
        if key not in BOTS_MAP:
            print(f"Bot desconocido: '{b}'. Usar c/a/e.")
            sys.exit(1)
        bots.append(BOTS_MAP[key])

    nombres = [b[0] for b in bots]
    print(f"Configuración: {' | '.join(f'J{i}={n}' for i, n in enumerate(nombres))}")
    print(f"Partidas: {args.partidas}  Semilla inicial: {args.seed}")

    resultados = []
    for i in range(args.partidas):
        r = simular_partida(bots, seed=args.seed + i, verbose=not args.resumen)
        resultados.append(r)

    imprimir_estadisticas(resultados, nombres)


if __name__ == "__main__":
    main()
