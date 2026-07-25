"""Tests de FiltroQS — restricción del conjunto legal para no comer Q♠ evitable."""
from __future__ import annotations

from src.agentes.filtro_qs import FiltroQS
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones

_QS = Carta._TODAS[36]   # Q♠
_KS = Carta._TODAS[37]   # K♠
_AS = Carta._TODAS[38]   # A♠


def _c(id_: int) -> Carta:
    return Carta._TODAS[id_]


def _motor(mesa=(), corazones_rotos=True, scores=(0, 0, 0, 0)) -> MotorCorazones:
    m = MotorCorazones()
    for i in range(4):
        m.jugadores[i].mano = []
        m.jugadores[i].bazas_ganadas = []
        m.jugadores[i].puntuacion_historica = scores[i]
    m.mesa = list(mesa)
    m.palo_de_salida = mesa[0][1].palo if mesa else None
    m.corazones_rotos = corazones_rotos
    m.numero_baza = 5
    m._mano_activa = True
    m.indice_jugador_inicial = mesa[0][0] if mesa else 0
    return m


def test_r1_veta_ganadoras_con_qs_en_mesa():
    # Mesa: 5♠ y luego Q♠. K♠ GANARÍA la baza (comería la Q); 3♠ pierde.
    m = _motor(mesa=[(1, _c(29)), (2, _QS)])  # 29 = 5♠
    legales = [_KS, _c(27)]  # K♠, 3♠
    out = FiltroQS().filtrar(m, 0, legales)
    assert out == [_c(27)]  # solo la perdedora


def test_r1_no_interviene_si_nadie_supera_la_qs():
    # 9♠ NO gana a la Q♠ en mesa → jugarlo es seguro, sin restricción.
    m = _motor(mesa=[(1, _c(29)), (2, _QS)])
    legales = [_c(33), _c(27)]  # 9♠, 3♠ — ambas pierden
    out = FiltroQS().filtrar(m, 0, legales)
    assert out == legales


def test_r1_sin_perdedora_no_cambia():
    m = _motor(mesa=[(1, _c(28)), (2, _QS)])  # 4♠, Q♠
    legales = [_KS, _AS]  # ambas ganan a la Q — no hay escape
    out = FiltroQS().filtrar(m, 0, legales)
    assert out == legales


def test_r2_no_liderar_qs_con_ka_vivas():
    m = _motor()
    legales = [_QS, _c(2), _c(15)]
    out = FiltroQS().filtrar(m, 0, legales)
    assert _QS not in out and len(out) == 2


def test_r2_veta_liderar_qs_incluso_con_ka_fuera():
    # Con K/A ya salidas, liderar la Q es AUTOCOMIDA garantizada -> veto igual.
    m = _motor()
    m.jugadores[1].bazas_ganadas = [_KS, _AS]
    legales = [_QS, _c(2)]
    out = FiltroQS().filtrar(m, 0, legales)
    assert _QS not in out


def test_r2_veta_qs_pero_deja_liderar_ka():
    m = _motor()
    m.jugadores[0].mano = [_QS, _KS, _AS]
    legales = [_QS, _KS, _AS]
    out = FiltroQS().filtrar(m, 0, legales)
    assert _QS not in out and _KS in out and _AS in out


def test_r3_veta_qs_propia_que_ganaria():
    # Picas lideradas con J♠ (35); nuestra Q♠ ganaría. Hay alternativa (2♠).
    m = _motor(mesa=[(3, _c(35))])
    legales = [_QS, _c(26)]  # Q♠, 2♠
    out = FiltroQS().filtrar(m, 0, legales)
    assert out == [_c(26)]


def test_r3_conserva_qs_que_pierde_bajo_as():
    # A♠ en mesa: jugar la Q♠ es DESCARGA (pierde) — no vetar.
    m = _motor(mesa=[(3, _AS)])
    legales = [_QS, _c(26)]
    out = FiltroQS().filtrar(m, 0, legales)
    assert out == legales


def test_descarte_offsuit_de_qs_no_se_veta():
    # Corazones liderados, somos void: descartar la Q♠ es regalo — intacto.
    m = _motor(mesa=[(1, _c(45))])  # un corazón
    legales = [_QS, _c(5)]
    out = FiltroQS().filtrar(m, 0, legales)
    assert out == legales


def test_endgame_desactiva_el_filtro():
    m = _motor(mesa=[(1, _c(29)), (2, _QS)], scores=(70, 90, 10, 20))
    legales = [_c(33), _c(27)]
    out = FiltroQS().filtrar(m, 0, legales)
    assert out == legales  # score>=74 presente → no aplica


def test_nunca_vacia_y_cuenta_stats():
    f = FiltroQS()
    m = _motor(mesa=[(1, _c(29)), (2, _QS)])
    out = f.filtrar(m, 0, [_KS, _c(27)])
    assert out and f.stats["intervenciones"] == 1 and f.stats["consultas"] == 1
