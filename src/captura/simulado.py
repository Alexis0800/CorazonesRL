"""
`AdaptadorSimulado` — fuente de eventos a partir de partidas auto-jugadas por el
motor (jugadas legales aleatorias, con pase real).

No captura nada del mundo real: sirve para VALIDAR el pipeline de captura
(eventos → RegistroPartida → JSONL → encoder) de punta a punta, sin móvil ni
teclear. Es la fuente del `--fuente demo` de scripts/capturar.py.
"""
from __future__ import annotations

import random
from typing import Iterator, List

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.captura.puerto import (
    AdaptadorJuego, Evento, FinMano, FinPartida, InicioMano, InicioPartida,
    JugadaObservada, PaseAgente,
)


class AdaptadorSimulado(AdaptadorJuego):
    def __init__(
        self, asiento_agente: int = 0, max_manos: int = 2,
        limite: int = 100, seed: int = 0,
    ) -> None:
        self.asiento_agente = asiento_agente
        self.max_manos = max_manos
        self.limite = limite
        self.rng = random.Random(seed)

    def _pase_aleatorio(self, motor: MotorCorazones) -> dict:
        sel = {}
        for i in range(4):
            sel[i] = self.rng.sample(motor.jugadores[i].mano, 3)
        return sel

    def eventos(self) -> Iterator[Evento]:
        ag = self.asiento_agente
        motor = MotorCorazones()
        motor.nueva_partida()  # reparte mano 1 y pone marcador a 0
        yield InicioPartida(asiento_agente=ag, fuente="simulado")

        manos = 0
        while True:
            manos += 1
            if manos > 1:
                motor.repartir()
            direccion = motor.direccion_pase()
            mano_inicial = [c.id for c in motor.jugadores[ag].mano]
            yield InicioMano(numero_mano=motor.numero_mano,
                             direccion_pase=direccion, mano_agente=mano_inicial)

            if direccion is not None:
                antes = set(motor.jugadores[ag].mano)
                sel = self._pase_aleatorio(motor)
                dadas = [c.id for c in sel[ag]]
                motor.ejecutar_pase(sel)
                recibidas = [c.id for c in motor.jugadores[ag].mano if c not in antes]
                yield PaseAgente(dadas=dadas, recibidas=recibidas)

            for baza in range(1, 14):
                for _ in range(4):
                    actual = motor.obtener_jugador_actual()
                    legales = motor.obtener_jugadas_legales(actual)
                    carta = legales[self.rng.randrange(len(legales))]
                    yield JugadaObservada(asiento=actual, carta_id=carta.id, baza=baza)
                    motor.jugar_carta(actual, carta)
                motor.resolver_baza()

            puntuacion = motor.aplicar_puntuacion()
            yield FinMano(puntuacion=puntuacion)

            marcador = motor.puntuaciones_historicas()
            if motor.partida_terminada(self.limite) or manos >= self.max_manos:
                yield FinPartida(marcador=marcador)
                break


__all__ = ["AdaptadorSimulado"]
