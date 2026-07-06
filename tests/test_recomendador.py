"""Test del reparto estimado en Recomendador._motor() (sin cargar checkpoint)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recomendador import Recomendador, parse_carta, parse_cartas
from src.dominio.carta import Carta


def _recomendador_sin_modelo(mi_idx: int = 0) -> Recomendador:
    r = Recomendador.__new__(Recomendador)
    r.me = mi_idx
    r._rng = np.random.default_rng(0)
    r.scores = [0, 0, 0, 0]
    r.reset_mano([])
    return r


def test_motor_reparte_todas_las_desconocidas():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    m = r._motor(mesa=[])
    repartidas = sum(len(m.jugadores[i].mano) for i in range(4) if i != r.me)
    assert repartidas == len(Carta._TODAS) - len(r.mano)


def test_motor_asigna_tamano_correcto_a_medio_baza():
    """Un rival que YA jugó en la baza en curso debe quedar con 1 carta menos
    que los que aún no juegan (antes: round-robin plano, ignoraba esto)."""
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")  # baza 1: mano completa
    mesa = [(1, parse_carta("2T")), (2, parse_carta("3T"))]  # rivales 1 y 2 ya jugaron

    m = r._motor(mesa=mesa)

    assert len(m.jugadores[1].mano) == 12  # ya jugó esta baza: 13-1
    assert len(m.jugadores[2].mano) == 12
    assert len(m.jugadores[3].mano) == 13  # no ha jugado esta baza todavía


def test_mundos_moon_no_asigna_carta_a_rival_void_en_ese_palo():
    """El contenido rival ya no se fija en _motor() -- lo decide determinizar()
    (reutilizado del PIMC del proyecto) al construir cada mundo, respetando vacíos."""
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")  # 13 picas
    r.vacios[1].add(3)  # rival 1 (izquierda) es void en corazones

    m = r._motor(mesa=[])
    for mundo in r._mundos_moon(m, n=15):
        for c in mundo.jugadores[1].mano:
            assert c.palo != 3, f"rival void en corazones recibió {c.id}"


def test_mundos_moon_devuelve_n_mundos_validos():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    m = r._motor(mesa=[])
    mundos = r._mundos_moon(m, n=10)
    assert len(mundos) == 10
    for mundo in mundos:
        repartidas = sum(len(mundo.jugadores[i].mano) for i in range(4) if i != r.me)
        assert repartidas == len(Carta._TODAS) - len(r.mano)
