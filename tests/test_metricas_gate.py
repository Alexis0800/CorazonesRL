"""Test de la métrica duck_innecesario_temprano (scripts/metricas_gate.py)
con casos construidos a mano sobre el motor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from metricas_gate import clasificar_duck, dama_capturada
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones

_DIAMANTE, _PICA, _TREBOL = 1, 2, 0


def _motor_baza_media(con_dama_capturada: bool = True) -> MotorCorazones:
    """Motor a mitad de mano: baza 5, asiento 3 lideró 5♦, sin puntos en mesa."""
    m = MotorCorazones()
    m.numero_baza = 5
    m.mesa = [(3, Carta(_DIAMANTE, 5))]
    m.palo_de_salida = _DIAMANTE
    if con_dama_capturada:
        m.jugadores[1].bazas_ganadas = [Carta(_PICA, 12)]  # Q♠ ya comida
    return m


def test_duck_siguiendo_palo():
    motor = _motor_baza_media()
    legales = [Carta(_DIAMANTE, 14), Carta(_DIAMANTE, 2)]  # A♦ y 2♦
    assert dama_capturada(motor)
    # Jugar el 2♦ teniendo el A♦ ganador = duck.
    assert clasificar_duck(motor, Carta(_DIAMANTE, 2), legales) is True
    # Jugar el A♦ = descargó la alta.
    assert clasificar_duck(motor, Carta(_DIAMANTE, 14), legales) is False


def test_duck_descarte_void():
    motor = _motor_baza_media()
    legales = [Carta(_TREBOL, 13), Carta(_TREBOL, 4)]  # void en ♦: K♣ y 4♣
    assert clasificar_duck(motor, Carta(_TREBOL, 4), legales) is True
    assert clasificar_duck(motor, Carta(_TREBOL, 13), legales) is False


def test_no_oportunidad():
    legales = [Carta(_DIAMANTE, 14), Carta(_DIAMANTE, 2)]
    # Q♠ sin capturar: hay riesgo -> no cuenta.
    assert clasificar_duck(_motor_baza_media(False), Carta(_DIAMANTE, 2), legales) is None
    # Baza tardía (>8): no cuenta.
    motor = _motor_baza_media()
    motor.numero_baza = 9
    assert clasificar_duck(motor, Carta(_DIAMANTE, 2), legales) is None
    # Puntos en mesa (corazón): no cuenta.
    motor = _motor_baza_media()
    motor.mesa.append((0, Carta(3, 7)))  # 7♥
    assert clasificar_duck(motor, Carta(_DIAMANTE, 2), legales) is None
    # Liderando (mesa vacía): no cuenta.
    motor = _motor_baza_media()
    motor.mesa = []
    motor.palo_de_salida = None
    assert clasificar_duck(motor, Carta(_DIAMANTE, 2), legales) is None
    # Sin alta ganadora (la mesa lleva el A♦): no cuenta.
    motor = _motor_baza_media()
    motor.mesa = [(3, Carta(_DIAMANTE, 14))]
    assert clasificar_duck(motor, Carta(_DIAMANTE, 2),
                           [Carta(_DIAMANTE, 12), Carta(_DIAMANTE, 2)]) is None
