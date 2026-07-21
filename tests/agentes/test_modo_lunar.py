"""
Tests de ModoLunar — ofensiva de pozo por composición (gate + BotLunatico).

Usa un estimador stub (P fija) para no depender de pesos entrenados: aquí se
prueba el CABLEADO (umbrales, gate duro, reset por mano), no el modelo.
"""
from __future__ import annotations

from src.agentes.modo_lunar import ModoLunar
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones


class _EstimadorStub:
    def __init__(self, p: float):
        self.p = p

    def propio(self, **kwargs) -> float:
        return self.p


def _motor_13() -> MotorCorazones:
    """Motor con 13 cartas por jugador repartidas de forma determinista."""
    m = MotorCorazones()
    for i in range(4):
        m.jugadores[i].mano = [Carta._TODAS[i * 13 + k] for k in range(13)]
        m.jugadores[i].bazas_ganadas = []
    m.numero_baza = 1
    m.mesa = []
    m.palo_de_salida = None
    m._mano_activa = True
    m.indice_jugador_inicial = 0
    return m


def test_pase_no_aplica_bajo_umbral():
    modo = ModoLunar(_EstimadorStub(0.05), umbral_pase=0.10)
    assert modo.elegir_pase(_motor_13(), 0) is None
    assert modo.stats["pases_ofensivos"] == 0


def test_pase_ofensivo_conserva_altas():
    modo = ModoLunar(_EstimadorStub(0.50), umbral_pase=0.10)
    m = _motor_13()
    cartas = modo.elegir_pase(m, 0)
    assert cartas is not None and len(cartas) == 3
    # Constructivo: suelta las BAJAS de la mano (nunca los controles altos).
    valores_pasados = {c.valor for c in cartas}
    valores_mano = sorted(c.valor for c in m.jugadores[0].mano)
    assert max(valores_pasados) <= valores_mano[5]  # dentro de la mitad baja
    assert modo.stats["pases_ofensivos"] == 1


def test_compromiso_y_gate_duro():
    modo = ModoLunar(_EstimadorStub(0.50), umbral_juego=0.15)
    m = _motor_13()
    legales = m.jugadores[0].mano
    # Primera decisión con mano de 13 → se compromete y devuelve carta.
    assert modo.elegir_jugada(m, 0, legales) is not None
    assert modo.comprometida
    # Un rival captura puntos → gate mata el compromiso, devuelve None.
    m.jugadores[1].bazas_ganadas = [Carta._TODAS[41]]  # un corazón
    assert modo.elegir_jugada(m, 0, legales) is None
    assert not modo.comprometida
    assert modo.stats["abortos_gate"] == 1
    # Y ya no vuelve a intentar esta mano.
    assert modo.elegir_jugada(m, 0, legales) is None


def test_abort_blando_por_prob():
    stub = _EstimadorStub(0.50)
    modo = ModoLunar(stub, umbral_juego=0.15, umbral_abort=0.10)
    m = _motor_13()
    assert modo.elegir_jugada(m, 0, m.jugadores[0].mano) is not None  # comprometida
    # Mid-mano (mano < 13) y P se desploma → abort blando definitivo.
    m.jugadores[0].mano = m.jugadores[0].mano[:9]
    m.jugadores[0].bazas_ganadas = [Carta._TODAS[39], Carta._TODAS[40],
                                    Carta._TODAS[41], Carta._TODAS[42]]
    stub.p = 0.02
    assert modo.elegir_jugada(m, 0, m.jugadores[0].mano) is None
    assert modo.stats["abortos_prob"] == 1
    assert not modo.comprometida


def test_sin_compromiso_devuelve_none():
    modo = ModoLunar(_EstimadorStub(0.01), umbral_juego=0.15)
    m = _motor_13()
    assert modo.elegir_jugada(m, 0, m.jugadores[0].mano) is None
    assert modo.stats["manos_comprometidas"] == 0


def test_reset_en_mano_nueva_hold():
    modo = ModoLunar(_EstimadorStub(0.50), umbral_juego=0.15)
    m = _motor_13()
    modo.elegir_jugada(m, 0, m.jugadores[0].mano)
    assert modo.comprometida
    # Simular mano nueva: bazas ganadas vuelven a cero (repartir limpia).
    m2 = _motor_13()
    m.jugadores[0].bazas_ganadas = [Carta._TODAS[0]] * 8  # avanza la mano vieja
    modo.elegir_jugada(m, 0, m.jugadores[0].mano)
    carta = modo.elegir_jugada(m2, 0, m2.jugadores[0].mano)  # cartas vistas caen → reset
    assert carta is not None  # re-evaluó y se comprometió de nuevo
    assert modo.stats["manos_comprometidas"] == 2
