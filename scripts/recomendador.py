"""
Recomendador / copiloto de Corazones — usa un modelo entrenado para aconsejarte
en una partida REAL (presencial u online), turno a turno.

Tú le pasas:
  - tu mano (13 cartas),
  - la dirección del pase de la mano (izquierda/derecha/frente/sin),
  - qué cartas se van jugando,
y el modelo te recomienda qué 3 cartas pasar y qué carta jugar en cada baza.

Funciona con info imperfecta: solo necesita TU mano + lo público (cartas jugadas,
marcador). Las manos rivales se estiman para features menores (prob. de pozo).

Formato de cartas: <valor><palo>, p.ej. 'AP' (As de picas), '10C' (10 de
corazones), 'QT' (Q de tréboles), '2D'. Palos: T=trébol ♣, D=diamante ♦,
P=pica ♠, C=corazón ♥. También acepta símbolos (A♠) y letras inglesas s/h/d/c.

Uso:
    python recomendador.py --modelo models/v10b/elite/elite_000018636800
    python recomendador.py --modelo ... --demo      # prueba no interactiva
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import sys
from typing import Dict, List, Optional, Set

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.observacion import ObservacionBuilder
from src.rllib.utils import cargar_policy_desde_checkpoint

_PALO_SIMBOLO = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
_VALOR_STR = {11: "J", 12: "Q", 13: "K", 14: "A"}
_PALO_DE_LETRA = {
    "T": 0, "♣": 0, "C_CLUB": 0,
    "D": 1, "♦": 1,
    "P": 2, "S": 2, "♠": 2,
    "C": 3, "H": 3, "♥": 3,
}
_DIRECCION_A_MANO = {"izquierda": 1, "derecha": 2, "frente": 3, "enfrente": 3, "sin": 4}

_LOOKUP = {(c.palo, c.valor): c for c in Carta._TODAS}


def cstr(c: Carta) -> str:
    return f"{_VALOR_STR.get(c.valor, str(c.valor))}{_PALO_SIMBOLO[c.palo]}"


def parse_carta(s: str) -> Carta:
    s = s.strip().upper().replace("10", "T_DIEZ")
    # palo = último carácter; valor = resto
    palo_ch = s[-1]
    valor_s = s[:-1].replace("T_DIEZ", "10")
    if palo_ch not in _PALO_DE_LETRA:
        raise ValueError(f"Palo no reconocido en '{s}' (usa T/D/P/C o ♣♦♠♥)")
    palo = _PALO_DE_LETRA[palo_ch]
    mapa_valor = {"A": 14, "K": 13, "Q": 12, "J": 11}
    valor = mapa_valor.get(valor_s, None)
    if valor is None:
        valor = int(valor_s)
    if (palo, valor) not in _LOOKUP:
        raise ValueError(f"Carta inválida: {s}")
    return _LOOKUP[(palo, valor)]


def parse_cartas(linea: str) -> List[Carta]:
    return [parse_carta(t) for t in linea.replace(",", " ").split()]


def mano_str(cartas: List[Carta]) -> str:
    por_palo: Dict[int, List[Carta]] = {0: [], 1: [], 2: [], 3: []}
    for c in cartas:
        por_palo[c.palo].append(c)
    partes = []
    for p in (2, 3, 0, 1):
        if por_palo[p]:
            cs = " ".join(_VALOR_STR.get(c.valor, str(c.valor))
                          for c in sorted(por_palo[p], key=lambda c: -c.valor))
            partes.append(f"{_PALO_SIMBOLO[p]} {cs}")
    return "   ".join(partes)


def _moon_prob(motor: MotorCorazones, jugador_idx: int) -> float:
    """Réplica de la heurística de P(Moon) del env (aprox. para rivales)."""
    for i, jug in enumerate(motor.jugadores):
        if i != jugador_idx and jug.contar_puntos_bazas() > 0:
            return 0.0
    jug = motor.jugadores[jugador_idx]
    todas = list(jug.mano) + list(jug.bazas_ganadas)
    high_hearts = sum(1 for c in todas if c.es_corazon and c.valor >= 10)
    hearts_ganados = sum(1 for c in jug.bazas_ganadas if c.es_corazon)
    qs_control = any(c.es_dama_de_picas for c in todas)
    hearts_en_rivales = sum(
        1 for i, jr in enumerate(motor.jugadores)
        if i != jugador_idx for c in jr.mano if c.es_corazon)
    control = (high_hearts / 5.0) * 0.60
    qs = 0.20 if qs_control else 0.0
    prog = min(hearts_ganados / 13.0, 1.0) * 0.10
    esc = min(hearts_en_rivales * 0.015, 0.10)
    return max(0.0, min(1.0, control + qs + prog - esc))


class Recomendador:
    """Mantiene el estado público de la partida y consulta al modelo."""

    def __init__(self, ckpt: str, mi_idx: int = 0):
        self.snap = cargar_policy_desde_checkpoint(ckpt)
        self.obs_dim = self.snap._obs_dim
        self.con_pase = self.obs_dim >= 228
        self.builder = ObservacionBuilder(dim=self.obs_dim)
        self.me = mi_idx
        self.scores = [0, 0, 0, 0]
        self.reset_mano([])

    def reset_mano(self, mi_mano: List[Carta]):
        self.mano: List[Carta] = list(mi_mano)
        self.cementerio: Dict[int, List[Carta]] = {i: [] for i in range(4)}
        self.vacios: List[Set[int]] = [set() for _ in range(4)]
        self.corazones_rotos = False
        self.numero_baza = 1
        self.dama_picas_en: Optional[int] = None

    # ---- reconstrucción del motor desde el estado público ----
    def _motor(self, mesa: List, numero_mano: int = 1) -> MotorCorazones:
        m = MotorCorazones()
        jugadas = set()
        for i in range(4):
            for c in self.cementerio[i]:
                jugadas.add(c.id)
            m.jugadores[i].bazas_ganadas = list(self.cementerio[i])
            m.jugadores[i].puntuacion_historica = self.scores[i]
        for _, c in mesa:
            jugadas.add(c.id)
        for c in self.mano:
            jugadas.add(c.id)
        m.jugadores[self.me].mano = list(self.mano)
        # Repartir las cartas desconocidas entre los rivales (estimación),
        # respetando los vacíos ya conocidos (mismo patrón que
        # ObservacionBuilder._calcular_prob_q_picas).
        desconocidas = [c for c in Carta._TODAS if c.id not in jugadas]
        otros = [i for i in range(4) if i != self.me]
        for k, c in enumerate(desconocidas):
            validos = [o for o in otros if c.palo not in self.vacios[o]] or otros
            m.jugadores[validos[k % len(validos)]].mano.append(c)
        m.mesa = list(mesa)
        m.palo_de_salida = mesa[0][1].palo if mesa else None
        m.corazones_rotos = self.corazones_rotos
        m.numero_baza = self.numero_baza
        m.numero_mano = numero_mano
        m.indice_jugador_inicial = (self.me - len(mesa)) % 4
        return m

    def _obs(self, m: MotorCorazones):
        scores = m.puntuaciones_historicas()
        mp_ag = _moon_prob(m, self.me)
        mp_riv = max(_moon_prob(m, i) for i in range(4) if i != self.me)
        puedo_alim = any(scores[j] >= 85 for j in range(4) if j != self.me)
        return self.builder.construir(
            motor=m, agente_idx=self.me, vacios=self.vacios,
            puntuacion_historica=scores,
            puntos_mano_actual=[j.contar_puntos_bazas() for j in m.jugadores],
            dama_picas_en=self.dama_picas_en,
            moon_prob_agente=mp_ag, moon_prob_rival=mp_riv, puedo_alimentar=puedo_alim)

    def recomendar_pase(self, direccion: str) -> List[Carta]:
        nm = _DIRECCION_A_MANO[direccion]
        m = self._motor(mesa=[], numero_mano=nm)
        if self.con_pase and hasattr(self.snap, "pasar"):
            return self.snap.pasar(m, self.me)
        from src.agentes.pase import pase_heuristico
        return pase_heuristico(m, self.me)

    def recomendar_jugada(self, mesa_antes: List) -> Carta:
        """mesa_antes = lista de (jugador_idx, Carta) jugadas antes de mí en esta baza."""
        m = self._motor(mesa=mesa_antes)
        legales = m.obtener_jugadas_legales(self.me)
        obs = self._obs(m)
        carta = self.snap(m, self.me, legales, obs_vec=obs)
        return carta if carta in legales else legales[0]

    def recomendar_jugada_entre(self, mesa_antes: List,
                                candidatas: List[Carta]) -> Carta:
        """Como `recomendar_jugada`, pero RESTRINGE la elección a `candidatas`
        (las cartas que la app deja jugar — detectadas por brillo en pantalla).

        La legalidad real la dicta la app, no la mesa reconstruida (que puede
        leerse mal): intersecta con las legales del motor solo si la intersección
        no queda vacía; si no, confía en `candidatas`. Así nunca propone una carta
        que la app no permite."""
        if not candidatas:
            return self.recomendar_jugada(mesa_antes)
        m = self._motor(mesa=mesa_antes)
        legales = m.obtener_jugadas_legales(self.me)
        permitidas = [c for c in candidatas if c in legales] or list(candidatas)
        if len(permitidas) == 1:
            return permitidas[0]
        obs = self._obs(m)
        carta = self.snap(m, self.me, permitidas, obs_vec=obs)
        return carta if carta in permitidas else permitidas[0]

    # ---- actualización de estado tras una baza completa ----
    def registrar_baza(self, jugadas: List, ganador: int):
        """jugadas = [(idx, carta), ...] en orden; ganador = idx que ganó."""
        palo_salida = jugadas[0][1].palo
        for idx, c in jugadas:
            if c.es_corazon:
                self.corazones_rotos = True
            if c.palo != palo_salida and idx != jugadas[0][0]:
                self.vacios[idx].add(palo_salida)
            if c.es_dama_de_picas:
                self.dama_picas_en = ganador
            if idx == self.me and c in self.mano:
                self.mano.remove(c)
        self.cementerio[ganador].extend([c for _, c in jugadas])
        self.numero_baza += 1

    # ---- snapshot para logging/depuración (comparar vs estado del bridge) ----
    def estado_actual(self) -> dict:
        return {
            "me": self.me,
            "scores": list(self.scores),
            "mano": sorted(c.id for c in self.mano),
            "cementerio": {i: sorted(c.id for c in cs) for i, cs in self.cementerio.items()},
            "vacios": {i: sorted(v) for i, v in enumerate(self.vacios)},
            "corazones_rotos": self.corazones_rotos,
            "numero_baza": self.numero_baza,
            "dama_picas_en": self.dama_picas_en,
        }


# ───────────────────────── interfaz interactiva ─────────────────────────

def _input_cartas(prompt: str, n: Optional[int] = None) -> List[Carta]:
    while True:
        try:
            cartas = parse_cartas(input(prompt))
            if n is not None and len(cartas) != n:
                print(f"  (se esperaban {n} cartas)"); continue
            return cartas
        except Exception as e:
            print(f"  entrada inválida: {e}")


def _input_cartas_opt(prompt: str, maxn: int) -> List[Carta]:
    """Lee 0..maxn cartas (vacío permitido). Reintenta si hay error."""
    while True:
        s = input(prompt).strip()
        if not s:
            return []
        try:
            cartas = parse_cartas(s)
            if len(cartas) > maxn:
                print(f"  (máximo {maxn} cartas aquí)"); continue
            return cartas
        except Exception as e:
            print(f"  entrada inválida: {e} — reintenta (ej: '2T 5C')")


def _input_carta_default(prompt: str, default: Carta) -> Carta:
    """Lee 1 carta o usa el default si se deja vacío."""
    while True:
        s = input(prompt).strip()
        if not s:
            return default
        try:
            return parse_carta(s)
        except Exception as e:
            print(f"  entrada inválida: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Recomendador de Corazones")
    p.add_argument("--modelo", required=True)
    p.add_argument("--demo", action="store_true", help="Prueba no interactiva")
    args = p.parse_args()

    rec = Recomendador(args.modelo)
    print(f"Modelo: {args.modelo}  (obs {rec.obs_dim}, {'CON' if rec.con_pase else 'SIN'} pase)")

    if args.demo:
        _demo(rec); return

    print("\nFormato: AP=A♠  10C=10♥  QT=Q♣  2D=2♦  (palos T♣ D♦ P♠ C♥)\n")
    while True:
        mano = _input_cartas("Tu mano (13 cartas): ")
        rec.reset_mano(mano)
        print(f"  → {mano_str(mano)}")

        if rec.con_pase:
            d = input("Dirección del pase [izquierda/derecha/frente/sin]: ").strip().lower()
            if d in _DIRECCION_A_MANO and d != "sin":
                sug = rec.recomendar_pase(d)
                print(f"  ★ PASA: {'  '.join(cstr(c) for c in sug)}")
                # aplicar el pase real
                quito = _input_cartas("  ¿Qué 3 pasaste? ", 3)
                for c in quito:
                    if c in rec.mano: rec.mano.remove(c)
                recibo = _input_cartas("  ¿Qué 3 recibiste? ", 3)
                rec.mano.extend(recibo)
                print(f"  Mano tras pase: {mano_str(rec.mano)}")

        print("\n— Juego — solo ingresa CARTAS (sin índices). Enter = vacío.")
        for baza in range(13):
            if not rec.mano:
                break
            print(f"\nBaza {baza+1}  (tu mano: {mano_str(rec.mano)})")
            antes = _input_cartas_opt(
                "  cartas jugadas ANTES de ti (vacío si lideras): ", maxn=3)
            k = len(antes)
            lider = (rec.me - k) % 4
            mesa = [((lider + i) % 4, antes[i]) for i in range(k)]

            sug = rec.recomendar_jugada(mesa)
            print(f"  ★ JUEGA: {cstr(sug)}")
            jugada = _input_carta_default(
                f"  ¿qué jugaste? (Enter = {cstr(sug)}): ", default=sug)

            despues = _input_cartas_opt(
                "  cartas jugadas DESPUÉS de ti (vacío si eres el último): ",
                maxn=3 - k)
            full = (mesa + [(rec.me, jugada)]
                    + [((rec.me + 1 + i) % 4, despues[i]) for i in range(len(despues))])
            if len(full) != 4:
                print(f"  ⚠ baza con {len(full)} cartas (esperaba 4); registro lo ingresado.")
            lead = full[0][1].palo
            ganador = max((sc for sc in full if sc[1].palo == lead),
                          key=lambda sc: sc[1].valor)[0]
            rec.registrar_baza(full, ganador)
            quien = "TÚ" if ganador == rec.me else f"seat {ganador}"
            print(f"  baza: {'  '.join(cstr(c) for _, c in full)}  → ganó {quien}")
        res = input("\nPuntos de la mano 'p0 p1 p2 p3' (enter para terminar): ").strip()
        if not res:
            break
        pts = [int(x) for x in res.split()]
        rec.scores = [rec.scores[i] + pts[i] for i in range(4)]
        print(f"Marcador: {rec.scores}")


def _demo(rec: Recomendador):
    """Smoke no interactivo: una mano arbitraria, recomendar pase y una jugada."""
    mano = parse_cartas("AP KP 2P 10C 5C 3C QT 9T 4T AD 7D 6D 2D")
    rec.reset_mano(mano)
    print(f"Mano demo: {mano_str(mano)}")
    if rec.con_pase:
        sug = rec.recomendar_pase("izquierda")
        print(f"Recomendación de PASE (izquierda): {'  '.join(cstr(c) for c in sug)}")
    # jugada liderando (mesa vacía)
    jug = rec.recomendar_jugada([])
    print(f"Recomendación de JUGADA (lidero, baza 1): {cstr(jug)}")
    # jugada siguiendo: alguien lideró 2♣
    jug2 = rec.recomendar_jugada([(2, parse_carta("2T"))])
    print(f"Recomendación de JUGADA (sigo a 2♣): {cstr(jug2)}")
    print("DEMO OK")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nSaliendo. ¡Suerte en la partida!")
