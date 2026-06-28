"""Tests del AdaptadorManual (consola) con entrada/salida inyectadas."""
from __future__ import annotations

import random

from src.captura.manual import AdaptadorManual, _ganador_baza, _puntuacion_mano
from src.captura.modelos import carta_a_str
from src.captura.recolector import RecolectorPartidas


def _entrada_de(lineas):
    it = iter(lineas)
    return lambda prompt="": next(it)


def test_ganador_baza_palo_de_salida():
    # lider 0 abre ♣: 2♣(0) 5♣(3) K♣(11) 7♣(5) -> gana K♣ (índice 2) -> asiento 2
    assert _ganador_baza([0, 3, 11, 5], 0) == 2
    # no asisten al palo: gana igual la más alta del palo de salida (la primera)
    assert _ganador_baza([5, 0 + 13, 1 + 13, 2 + 13], 1) == 1


def test_puntuacion_qs_y_corazones():
    # asiento 1 captura la Q♠ (id 36) en una baza de tréboles
    assert _puntuacion_mano([(1, [36, 0, 1, 2])]) == [0, 13, 0, 0]


def test_flujo_completo_una_mano():
    rng = random.Random(0)
    ids = list(range(52))
    rng.shuffle(ids)
    mano = ids[:13]
    pase = mano[:3]
    tricks = [ids[i * 4:(i + 1) * 4] for i in range(13)]

    lineas = [" ".join(carta_a_str(c) for c in mano)]   # tu mano
    lineas.append(" ".join(carta_a_str(c) for c in pase))  # pase dadas
    lineas.append("")                                      # recibidas (opcional)
    lineas.append("0")                                     # asiento que abre
    lineas += [" ".join(carta_a_str(c) for c in t) for t in tricks]
    lineas.append("n")                                     # ¿otra mano? no

    adaptador = AdaptadorManual(
        asiento_agente=0, entrada=_entrada_de(lineas), salida=lambda *a, **k: None,
    )
    partidas = RecolectorPartidas(adaptador=adaptador).ejecutar()

    assert len(partidas) == 1
    p = partidas[0]
    assert len(p.manos) == 1
    assert len(p.manos[0].jugadas) == 52
    assert p.manos[0].pase_dado == pase
    # todos los puntos del juego (26) se reparten; con pleno serían 78
    assert sum(p.manos[0].puntuacion_mano) in (26, 78)
    assert sum(p.marcador_final) == sum(p.manos[0].puntuacion_mano)
