"""Roundtrip del encoder: simula una mano real en el motor y la re-juega."""
from __future__ import annotations

import random

from src.captura.modelos import Jugada, RegistroMano, RegistroPartida
from src.captura.replay import (
    ejemplos_de_mano, ejemplos_de_partida, partidas_a_arrays, reconstruir_manos,
)
from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_ENTORNO


def _simular_mano(seed: int, agente: int = 0) -> RegistroPartida:
    """Juega una mano completa con jugadas legales aleatorias y la registra."""
    rng = random.Random(seed)
    motor = MotorCorazones()
    motor.repartir()
    mano_inicial = [c.id for c in motor.jugadores[agente].mano]
    jugadas = []
    for baza in range(1, 14):
        for _ in range(4):
            actual = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(actual)
            carta = legales[rng.randrange(len(legales))]
            jugadas.append(Jugada(asiento=actual, carta_id=carta.id, baza=baza))
            motor.jugar_carta(actual, carta)
        motor.resolver_baza()
    punt = motor.calcular_puntuacion_mano()
    rm = RegistroMano(
        numero_mano=1, direccion_pase="izquierda",
        mano_inicial_agente=mano_inicial, jugadas=jugadas, puntuacion_mano=punt,
    )
    return RegistroPartida(
        partida_id="sim", timestamp="t", asiento_agente=agente, fuente="test",
        manos=[rm], marcador_final=punt,
        ranking_final=sorted(range(4), key=lambda s: punt[s]),
    )


def test_reconstruir_manos_completas():
    p = _simular_mano(1)
    manos = reconstruir_manos(p.manos[0])
    assert all(len(m) == 13 for m in manos)
    ids = {c.id for m in manos for c in m}
    assert len(ids) == 52


def test_replay_produce_13_ejemplos_por_mano():
    p = _simular_mano(0, agente=0)
    ejemplos = ejemplos_de_partida(p)
    assert len(ejemplos) == 13
    for obs, accion in ejemplos:
        assert obs.shape == (DIM_ENTORNO,)
        assert 0 <= accion < 52
    # sin pase ejecutado: el agente jugó exactamente sus 13 cartas repartidas
    assert {a for _, a in ejemplos} == set(p.manos[0].mano_inicial_agente)


def test_replay_para_cualquier_asiento():
    for ag in range(4):
        p = _simular_mano(7 + ag, agente=ag)
        assert len(ejemplos_de_partida(p)) == 13


def test_replay_con_remate_resto():
    """Una mano que termina por concesión se reconstruye y re-juega bien."""
    rng = random.Random(5)
    motor = MotorCorazones()
    motor.repartir()
    agente = 0
    ini = [c.id for c in motor.jugadores[agente].mano]
    jug = []
    k_bazas = 8
    for baza in range(1, k_bazas + 1):
        for _ in range(4):
            a = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(a)
            c = legales[rng.randrange(len(legales))]
            jug.append(Jugada(asiento=a, carta_id=c.id, baza=baza))
            motor.jugar_carta(a, c)
        motor.resolver_baza()
    restantes = [[c.id for c in motor.jugadores[s].mano] for s in range(4)]
    rm = RegistroMano(
        numero_mano=1, direccion_pase="izquierda", mano_inicial_agente=ini,
        jugadas=jug, remate_asiento=motor.indice_jugador_inicial,
        manos_restantes=restantes, puntuacion_mano=[],
    )
    manos = reconstruir_manos(rm)
    assert all(len(m) == 13 for m in manos)
    from src.entorno.observacion import ObservacionBuilder
    ejemplos = ejemplos_de_mano(rm, agente, ObservacionBuilder(dim=DIM_ENTORNO))
    assert len(ejemplos) == k_bazas  # una decisión del agente por baza jugada


def test_partidas_a_arrays_shapes():
    partidas = [_simular_mano(s) for s in range(3)]
    X, y = partidas_a_arrays(partidas)
    assert X.shape == (39, DIM_ENTORNO)   # 3 partidas * 13 turnos
    assert y.shape == (39,)
    assert X.dtype.name == "float32"
