"""
`AdaptadorManual` — fuente de captura por CONSOLA, sin ADB ni visión.

El usuario, observando una partida real (presencial u online en su móvil),
narra las jugadas. Es el camino "entrada manual" del ROADMAP: el 90% del valor
con el 10% del esfuerzo, y funciona HOY. Calcula el marcador por su cuenta
(reglas de Corazones) para no pedírselo al humano.

Entrada/salida son inyectables (`entrada`/`salida`) para poder testear sin TTY.
"""
from __future__ import annotations

from typing import Callable, Iterator, List

from src.dominio.carta import Carta
from src.captura.modelos import carta_a_str, str_a_carta_id
from src.captura.puerto import (
    AdaptadorJuego, Evento, FinMano, FinPartida, InicioMano, InicioPartida,
    JugadaObservada, PaseAgente,
)

_DIRECCION = {0: "izquierda", 1: "derecha", 2: "enfrente", 3: None}


def _ganador_baza(cartas: List[int], asiento_inicial: int) -> int:
    """Asiento ganador: carta más alta del palo de salida."""
    palo_salida = Carta._TODAS[cartas[0]].palo
    mejor_i, mejor_val = 0, Carta._TODAS[cartas[0]].valor
    for i, cid in enumerate(cartas):
        c = Carta._TODAS[cid]
        if c.palo == palo_salida and c.valor > mejor_val:
            mejor_i, mejor_val = i, c.valor
    return (asiento_inicial + mejor_i) % 4


def _puntuacion_mano(bazas: List[tuple]) -> List[int]:
    """`bazas` = lista de (asiento_ganador, [4 carta_id]). Aplica regla de pleno."""
    crudos = [0, 0, 0, 0]
    for ganador, cartas in bazas:
        crudos[ganador] += sum(Carta._TODAS[c].puntos for c in cartas)
    for i, pts in enumerate(crudos):
        if pts == 26:  # pleno (shoot the moon)
            res = [26, 26, 26, 26]
            res[i] = 0
            return res
    return crudos


class AdaptadorManual(AdaptadorJuego):
    def __init__(
        self,
        asiento_agente: int = 0,
        limite: int = 100,
        entrada: Callable[[str], str] = input,
        salida: Callable[[str], None] = print,
    ) -> None:
        self.asiento_agente = asiento_agente
        self.limite = limite
        self._in = entrada
        self._out = salida

    # --- parsing de líneas ---
    def _pedir_cartas(self, prompt: str, n: int, opcional: bool = False) -> List[int]:
        while True:
            crudo = self._in(prompt).strip()
            if opcional and not crudo:
                return []
            try:
                ids = [str_a_carta_id(t) for t in crudo.split()]
            except ValueError as e:
                self._out(f"  ⚠ {e} — reintenta.")
                continue
            if len(ids) != n:
                self._out(f"  ⚠ esperaba {n} cartas, recibí {len(ids)} — reintenta.")
                continue
            return ids

    def _pedir_int(self, prompt: str, lo: int, hi: int) -> int:
        while True:
            try:
                v = int(self._in(prompt).strip())
            except ValueError:
                self._out("  ⚠ número inválido — reintenta.")
                continue
            if lo <= v <= hi:
                return v
            self._out(f"  ⚠ fuera de rango [{lo},{hi}] — reintenta.")

    def eventos(self) -> Iterator[Evento]:  # noqa: C901 (flujo lineal de UI)
        self._out("=== Captura manual de partida de Corazones ===")
        self._out("Cartas: <valor><palo>, palos T♣ D♦ P♠ C♥. Ej: 2T QP 10D AH")
        yield InicioPartida(asiento_agente=self.asiento_agente, fuente="manual")

        marcador = [0, 0, 0, 0]
        numero_mano = 0
        while True:
            numero_mano += 1
            direccion = _DIRECCION[(numero_mano - 1) % 4]
            self._out(f"\n--- Mano {numero_mano} (pase: {direccion or 'sin pase'}) ---")
            mano_agente = self._pedir_cartas("Tu mano (13 cartas): ", 13)
            yield InicioMano(numero_mano=numero_mano, direccion_pase=direccion,
                             mano_agente=mano_agente)

            if direccion is not None:
                dadas = self._pedir_cartas("Cartas que pasas (3): ", 3)
                recibidas = self._pedir_cartas(
                    "Cartas que recibes (3, enter si no las sabes): ", 3, opcional=True
                )
                yield PaseAgente(dadas=dadas, recibidas=recibidas)

            bazas: List[tuple] = []
            lider = self._pedir_int("Asiento que abre la baza 1 [0-3]: ", 0, 3)
            for baza in range(1, 14):
                cartas = self._pedir_cartas(
                    f"Baza {baza} — 4 cartas en orden desde asiento {lider}: ", 4
                )
                for k, cid in enumerate(cartas):
                    asiento = (lider + k) % 4
                    yield JugadaObservada(asiento=asiento, carta_id=cid, baza=baza)
                ganador = _ganador_baza(cartas, lider)
                bazas.append((ganador, cartas))
                lider = ganador

            puntuacion = _puntuacion_mano(bazas)
            for i in range(4):
                marcador[i] += puntuacion[i]
            self._out(f"Puntuación mano: {puntuacion} | marcador: {marcador}")
            yield FinMano(puntuacion=puntuacion)

            if max(marcador) >= self.limite:
                break
            if self._in("¿Otra mano? [s/n]: ").strip().lower().startswith("n"):
                break

        yield FinPartida(marcador=marcador)


__all__ = ["AdaptadorManual"]
