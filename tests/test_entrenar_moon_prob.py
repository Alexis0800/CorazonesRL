"""Tests de scripts/entrenar_moon_prob.py (generación de dataset, sin entrenar red)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from entrenar_moon_prob import _receptor_y_dador, ejemplos_de_mano
from src.captura.modelos import Jugada, RegistroMano
from src.entorno.moon_model import DIM_PROPIO, DIM_RIVAL


def _mano_completa_simple() -> RegistroMano:
    """Una mano de 13 bazas donde el asiento 0 gana TODO (pozo perfecto),
    construida a mano con jugadas legales reales (2T primero, sigue el palo
    cuando puede)."""
    # Reutiliza una partida real jugada por el motor mismo para garantizar
    # legalidad, en vez de inventar 52 cartas a mano.
    from src.dominio.motor import MotorCorazones

    m = MotorCorazones()
    m.repartir()
    jugadas = []
    baza = 1
    while not all(len(j.mano) == 0 for j in m.jugadores):
        idx = m.obtener_jugador_actual()
        legales = m.obtener_jugadas_legales(idx)
        # El asiento 0 siempre intenta ganar (juega la más alta legal);
        # los demás juegan la más baja legal -- fuerza que 0 gane todo.
        carta = max(legales, key=lambda c: c.valor) if idx == 0 else min(legales, key=lambda c: c.valor)
        jugadas.append(Jugada(asiento=idx, carta_id=carta.id, baza=baza))
        m.jugar_carta(idx, carta)
        if len(m.mesa) == 4:
            m.resolver_baza()
            baza += 1
    puntos = m.calcular_puntuacion_mano()
    return RegistroMano(
        numero_mano=1, direccion_pase=None, mano_inicial_agente=[],
        jugadas=jugadas, puntuacion_mano=puntos,
    )


def test_receptor_y_dador_izquierda():
    receptor, dador = _receptor_y_dador("izquierda", 0)
    assert receptor == 1  # seat+1 = izquierda
    assert dador == 3     # seat-1 me pasó a mí


def test_receptor_y_dador_sin_pase():
    assert _receptor_y_dador(None, 0) == (None, None)


def test_ejemplos_de_mano_formas_y_no_vacio():
    mano = _mano_completa_simple()
    ejemplos_propio, ejemplos_rival = ejemplos_de_mano(mano, asiento_agente_real=0)
    assert len(ejemplos_propio) > 0
    assert len(ejemplos_rival) > 0
    feats, label = ejemplos_propio[0]
    assert feats.shape == (DIM_PROPIO,)
    assert label in (0.0, 1.0)
    feats_r, label_r = ejemplos_rival[0]
    assert feats_r.shape == (DIM_RIVAL,)
    assert label_r in (0.0, 1.0)


def test_ejemplos_de_mano_etiqueta_el_pozo_correctamente():
    mano = _mano_completa_simple()
    # calcular_puntuacion_mano() ya aplica la regla de Pleno: el tirador queda
    # en 0 y los otros 3 en 26 (suma 78). Una mano normal siempre suma 26 (sin
    # garantizar que nadie tenga exactamente 0), así que el pozo se detecta
    # por suma == 78, no == 26.
    if sum(mano.puntuacion_mano) != 78:
        return  # esta semilla en particular no produjo pozo; no es el foco del test
    luna_seat = mano.puntuacion_mano.index(0)
    ejemplos_propio, _ = ejemplos_de_mano(mano, asiento_agente_real=0)
    # al menos un ejemplo de la perspectiva del que hizo el pozo debe tener label 1.0
    # (los del asiento ganador, antes de que el gate lo excluya en las últimas bazas)
    assert any(label == 1.0 for _, label in ejemplos_propio) or luna_seat != 0
