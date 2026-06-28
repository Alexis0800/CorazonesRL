"""
Probar un modelo entrenado de Corazones (v10b/v10c con pase, o v10 sin pase).

Dos modos:
  --ver       Observar al MODELO jugar una partida completa vs bots, con traza
              legible (pase + cada baza + marcador). No interactivo.
  --humano    Jugar TÚ una partida contra el modelo + bots (entrada por teclado).

El modelo se carga desde un checkpoint (elite o snapshot). Detecta solo si usa
pase (obs_dim>=228) y reproduce la fase de pase.

Uso:
    python jugar_modelo.py --modelo models/v10b/elite/elite_000018636800 --ver
    python jugar_modelo.py --modelo models/v10b/elite/elite_000018636800 --humano
    python jugar_modelo.py --modelo models/v10_bc_ppo/elite/elite_000011182080 --ver
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.agentes.bot_experto import BotExperto
from src.agentes.bot_castigador import BotCastigador
from src.agentes.heuristicos import bot_evasivo
from src.agentes.pase import pase_heuristico
from src.rllib.utils import cargar_policy_desde_checkpoint

_PALO_SIMBOLO = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
_VALOR_STR = {11: "J", 12: "Q", 13: "K", 14: "A"}


def _cstr(c: Carta) -> str:
    v = _VALOR_STR.get(c.valor, str(c.valor))
    return f"{v}{_PALO_SIMBOLO[c.palo]}"


def _mano_str(cartas: List[Carta]) -> str:
    por_palo: Dict[int, List[Carta]] = {0: [], 1: [], 2: [], 3: []}
    for c in cartas:
        por_palo[c.palo].append(c)
    partes = []
    for p in (2, 3, 0, 1):  # ♠♥♣♦
        if por_palo[p]:
            cs = " ".join(_VALOR_STR.get(c.valor, str(c.valor))
                          for c in sorted(por_palo[p], key=lambda c: -c.valor))
            partes.append(f"{_PALO_SIMBOLO[p]} {cs}")
    return "  ".join(partes)


def _fin_mano(m: MotorCorazones) -> bool:
    return m.numero_baza > 13 or all(len(j.mano) == 0 for j in m.jugadores)


class ModeloJugador:
    """Envuelve un SnapshotPolicy con obs COMPLETA del env (juego + pase)."""

    def __init__(self, ckpt: str):
        self.snap = cargar_policy_desde_checkpoint(ckpt)
        self.obs_dim = self.snap._obs_dim
        self.con_pase = self.obs_dim >= 228

    def pasar(self, motor: MotorCorazones, idx: int) -> List[Carta]:
        if self.con_pase:
            return self.snap.pasar(motor, idx)
        return pase_heuristico(motor, idx)

    def jugar(self, motor: MotorCorazones, idx: int, legales: List[Carta],
              obs_vec) -> Carta:
        return self.snap(motor, idx, legales, obs_vec=obs_vec)


def _construir_obs_juego(builder, motor, idx, vacios, dama_picas_en):
    """Obs completa de juego para el modelo (perspectiva de idx)."""
    scores = motor.puntuaciones_historicas()
    puntos_mano = [j.contar_puntos_bazas() for j in motor.jugadores]
    return builder.construir(
        motor=motor, agente_idx=idx, vacios=vacios,
        puntuacion_historica=scores, puntos_mano_actual=puntos_mano,
        dama_picas_en=dama_picas_en, fase_pase=0.0,
    )


def jugar_partida(modelo: ModeloJugador, seats: Dict[int, str], modelo_idx: int,
                  ver: bool, humano_idx: Optional[int]) -> List[int]:
    """Juega una partida completa. seats[idx] = 'modelo'|'experto'|'castigador'|
    'evasivo'|'humano'. Devuelve puntuaciones finales."""
    from src.entorno.observacion import ObservacionBuilder
    builder = ObservacionBuilder(dim=modelo.obs_dim)

    motor = MotorCorazones()
    motor.nueva_partida()
    bots = {0: bot_evasivo, 1: bot_evasivo, 2: bot_evasivo, 3: bot_evasivo}
    for i, tipo in seats.items():
        if tipo == "experto":
            bots[i] = BotExperto()
        elif tipo == "castigador":
            bots[i] = BotCastigador()
        elif tipo == "evasivo":
            bots[i] = bot_evasivo

    def log(msg=""):
        if ver or humano_idx is not None:
            print(msg)

    while not motor.partida_terminada(100):
        vacios = [set() for _ in range(4)]
        dama_picas_en = None

        # ── Fase de pase ──
        direccion = motor.direccion_pase()
        if direccion is not None and modelo.con_pase:
            log(f"\n━━━ Mano {motor.numero_mano} — pase: {direccion} ━━━")
            selecciones = {}
            for i in range(4):
                tipo = seats[i]
                if tipo == "humano":
                    selecciones[i] = _pedir_pase_humano(motor, i)
                elif tipo == "modelo":
                    selecciones[i] = modelo.pasar(motor, i)
                else:
                    pasar = getattr(bots[i], "pasar", None)
                    selecciones[i] = pasar(motor, i) if callable(pasar) \
                        else pase_heuristico(motor, i)
                if ver and tipo == "modelo":
                    log(f"  [modelo seat {i}] pasa: "
                        f"{' '.join(_cstr(c) for c in selecciones[i])}")
            motor.ejecutar_pase(selecciones)
        else:
            log(f"\n━━━ Mano {motor.numero_mano} — sin pase ━━━")

        # ── Juego de la mano ──
        while not _fin_mano(motor):
            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)
            palo_salida = motor.palo_de_salida
            tipo = seats[idx]
            if tipo == "humano":
                carta = _pedir_jugada_humano(motor, idx, legales)
            elif tipo == "modelo":
                obs = _construir_obs_juego(builder, motor, idx, vacios, dama_picas_en)
                carta = modelo.jugar(motor, idx, legales, obs)
            else:
                carta = bots[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)
            if palo_salida is not None and carta.palo != palo_salida:
                vacios[idx].add(palo_salida)
            if any(c.es_dama_de_picas for _, c in motor.mesa):
                pass
            if len(motor.mesa) == 4:
                cartas = motor.mesa[:]
                ganador = motor.resolver_baza()
                if any(c.es_dama_de_picas for _, c in cartas):
                    dama_picas_en = ganador
                if ver:
                    jugadas = "  ".join(
                        f"{'›' if s == ganador else ' '}{s}:{_cstr(c)}"
                        for s, c in cartas)
                    log(f"  baza {motor.numero_baza-1:>2}: {jugadas}  → gana {ganador}")

        puntos = motor.calcular_puntuacion_mano()
        motor.aplicar_puntuacion()
        log(f"  puntos mano: {puntos}   marcador: {motor.puntuaciones_historicas()}")
        if not motor.partida_terminada(100):
            motor.repartir()
            for b in bots.values():
                r = getattr(b, "reset", None)
                if callable(r):
                    try: r()
                    except Exception: pass

    return motor.puntuaciones_historicas()


def _pedir_pase_humano(motor, idx) -> List[Carta]:
    mano = sorted(motor.jugadores[idx].mano, key=lambda c: (c.palo, -c.valor))
    print(f"\nTu mano: {_mano_str(mano)}")
    print("Elige 3 cartas para pasar (ej: '2♠ A♥ 5♣' o índices).")
    print("  " + "  ".join(f"[{i}]{_cstr(c)}" for i, c in enumerate(mano)))
    while True:
        try:
            idxs = [int(x) for x in input("  > ").split()]
            if len(idxs) == 3 and all(0 <= i < len(mano) for i in idxs) and len(set(idxs)) == 3:
                return [mano[i] for i in idxs]
        except Exception:
            pass
        print("  Entrada inválida; ingresa 3 índices distintos.")


def _pedir_jugada_humano(motor, idx, legales) -> Carta:
    legales_s = sorted(legales, key=lambda c: (c.palo, -c.valor))
    mesa = "  ".join(f"{s}:{_cstr(c)}" for s, c in motor.mesa) or "(lideras)"
    print(f"\nMesa: {mesa}")
    print(f"Legales: " + "  ".join(f"[{i}]{_cstr(c)}" for i, c in enumerate(legales_s)))
    while True:
        try:
            i = int(input("  juega > "))
            if 0 <= i < len(legales_s):
                return legales_s[i]
        except Exception:
            pass
        print("  Índice inválido.")


def main() -> None:
    p = argparse.ArgumentParser(description="Probar un modelo de Corazones")
    p.add_argument("--modelo", required=True, help="Ruta al checkpoint (elite/snapshot)")
    p.add_argument("--ver", action="store_true", help="Ver al modelo jugar (no interactivo)")
    p.add_argument("--humano", action="store_true", help="Jugar tú contra el modelo")
    p.add_argument("--partidas", type=int, default=1, help="Nº de partidas (modo --ver)")
    p.add_argument("--rivales", default="experto",
                   choices=["experto", "castigador", "evasivo"],
                   help="Tipo de los 3 rivales del modelo (modo --ver)")
    args = p.parse_args()

    modelo = ModeloJugador(args.modelo)
    print(f"Modelo: {args.modelo}  (obs_dim {modelo.obs_dim}, "
          f"{'CON' if modelo.con_pase else 'SIN'} pase)")

    if args.humano:
        # Humano seat 0, modelo seat 1, 2 rivales experto.
        seats = {0: "humano", 1: "modelo", 2: "experto", 3: "experto"}
        scores = jugar_partida(modelo, seats, modelo_idx=1, ver=True, humano_idx=0)
        ranking = sorted(range(4), key=lambda i: scores[i])
        nombres = {0: "TÚ", 1: "modelo", 2: "experto", 3: "experto"}
        print(f"\n=== FIN === marcador {scores}")
        print(f"Ganador: {nombres[ranking[0]]}  | tu puesto: {ranking.index(0)+1}")
        return

    # Modo --ver: modelo seat 0 vs 3 rivales.
    seats = {0: "modelo", 1: args.rivales, 2: args.rivales, 3: args.rivales}
    puestos = []
    for g in range(args.partidas):
        if args.partidas > 1:
            print(f"\n##### PARTIDA {g+1}/{args.partidas} #####")
        scores = jugar_partida(modelo, seats, modelo_idx=0,
                               ver=(args.partidas == 1), humano_idx=None)
        puesto = 1 + sum(1 for i in range(1, 4) if scores[i] < scores[0])
        puestos.append(puesto)
        print(f"Partida {g+1}: marcador {scores} → modelo puesto {puesto}")
    if args.partidas > 1:
        import numpy as _np
        ps = _np.array(puestos)
        print(f"\nResumen {args.partidas} partidas vs {args.rivales}: "
              f"win {(ps==1).mean():.2f}  top2 {(ps<=2).mean():.2f}  "
              f"puesto medio {ps.mean():.2f}")


if __name__ == "__main__":
    main()
