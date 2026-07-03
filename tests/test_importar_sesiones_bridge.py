from scripts.importar_sesiones_bridge import partida_desde_lineas, swap_suit
from src.captura.replay import mano_reconstruible


def test_swap_suit_es_su_propia_inversa():
    for cable in range(52):
        modelo = swap_suit(cable)
        assert swap_suit(modelo) == cable
    assert swap_suit(26) == 0  # 2♣ de cable (suit 2) -> 2♣ del modelo (suit 0)


def _linea(tipo, **kv):
    return {"t": "2026-01-01T00:00:00.000Z", "type": tipo, **kv}


def test_mano_completa_de_13_bazas_es_reconstruible():
    """Una mano donde vimos las 52 jugadas (13 bazas x 4 asientos) debe ser reproducible."""
    cartas_cable = list(range(13))
    lineas = [
        _linea("handDealt", seat=0, cartas=cartas_cable, scores=[0, 0, 0, 0]),
        _linea("pass", ids=cartas_cable[:3], dir="izquierda"),
        _linea("passReceived", dir=1, cards=[40, 41, 42]),
    ]
    for baza in range(13):
        for asiento in range(4):
            lineas.append(_linea("cardPlayed", seat=asiento, cardId=(baza * 4 + asiento) % 52, trick=baza))
    lineas.append(_linea("handFinal", puntos=[10, 5, 6, 5], lunaSeat=None, scores=[10, 5, 6, 5], complete=True))
    lineas.append(_linea("matchEnded", reason=0, marcadorFinal=[100, 40, 60, 50]))

    partida = partida_desde_lineas(lineas, "sesion-test")
    assert partida is not None
    assert len(partida.manos) == 1
    mano = partida.manos[0]
    assert len(mano.jugadas) == 52
    assert mano.direccion_pase == "izquierda"
    assert mano.pase_dado == [swap_suit(c) for c in cartas_cable[:3]]
    assert mano_reconstruible(mano)
    # peor->mejor por marcador (mayor puntaje = peor en Corazones)
    assert partida.ranking_final == [0, 2, 3, 1]


def test_mano_con_remate_se_guarda_pero_no_es_reconstruible():
    """Cuando la app resuelve el resto, el bridge no sabe qué asiento tenía cada carta
    del barrido -- la mano debe guardarse (con su puntuación) pero replay.py debe
    marcarla como no reproducible jugada-a-jugada."""
    lineas = [
        _linea("handDealt", seat=0, cartas=list(range(13)), scores=[0, 0, 0, 0]),
        _linea("pass", ids=[0, 1, 2], dir="derecha"),
        _linea("passReceived", dir=2, cards=[40, 41, 42]),
    ]
    for baza in range(3):  # solo 3 de 13 bazas antes del remate
        for asiento in range(4):
            lineas.append(_linea("cardPlayed", seat=asiento, cardId=(baza * 4 + asiento) % 52, trick=baza))
    lineas.append(_linea("handFinal", puntos=[26, 0, 0, 0], lunaSeat=None, scores=[26, 0, 0, 0], complete=True))

    partida = partida_desde_lineas(lineas, "sesion-test-remate")
    mano = partida.manos[0]
    assert len(mano.jugadas) == 12
    assert mano.puntuacion_mano == [26, 0, 0, 0]
    assert not mano_reconstruible(mano)  # sin manos_restantes, no se puede completar a 13 c/u


def test_eco_tardio_duplicado_se_ignora_con_aviso():
    lineas = [
        _linea("handDealt", seat=0, cartas=list(range(13)), scores=[0, 0, 0, 0]),
        _linea("cardPlayed", seat=0, cardId=0, trick=0),
        _linea("cardPlayed", seat=0, cardId=0, trick=0),  # eco tardío duplicado
        _linea("cardPlayed", seat=1, cardId=1, trick=0),
        _linea("cardPlayed", seat=2, cardId=2, trick=0),
        _linea("cardPlayed", seat=3, cardId=3, trick=0),
        _linea("handFinal", puntos=[0, 0, 0, 0], lunaSeat=None, scores=[0, 0, 0, 0], complete=True),
    ]
    avisos = []
    partida = partida_desde_lineas(lineas, "sesion-eco", avisos=avisos)
    assert len(partida.manos[0].jugadas) == 4
    assert any("eco tardío" in a for a in avisos)


def test_sesion_sin_manos_devuelve_none():
    lineas = [_linea("roomLeft", roomId=1, requeueing=False)]
    assert partida_desde_lineas(lineas, "sesion-vacia") is None
