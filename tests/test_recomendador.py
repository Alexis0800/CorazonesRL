"""Test del reparto estimado en Recomendador._motor() (sin cargar checkpoint)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recomendador import Recomendador, parse_carta, parse_cartas
from src.dominio.carta import Carta
from src.entorno.moon_model import EntradaBaza
from src.entorno.observacion import ObservacionBuilder


def _recomendador_sin_modelo(mi_idx: int = 0) -> Recomendador:
    from src.entorno.moon_model import EstimadorMoonProb
    r = Recomendador.__new__(Recomendador)
    r.me = mi_idx
    r._estimador_moon = EstimadorMoonProb(dir_modelos="models/moon_inexistente")
    r.builder = ObservacionBuilder()
    r.scores = [0, 0, 0, 0]
    r.reset_mano([])
    return r


def test_registrar_baza_agrega_al_historial():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    jugadas = [(0, parse_carta("2T")), (1, parse_carta("5T")),
               (2, parse_carta("9C")), (3, parse_carta("3T"))]  # 9 de corazones: tiene puntos
    r.registrar_baza(jugadas, ganador=2)
    assert len(r.historial_bazas) == 1
    entrada = r.historial_bazas[0]
    assert entrada.lider == 0
    assert entrada.ganador == 2
    assert entrada.tenia_puntos is True
    assert entrada.lidero_corazon_o_dama is False  # lideró con 2♣, no con corazón/Q♠


def test_registrar_resto_agrega_al_historial_sin_lider():
    r = _recomendador_sin_modelo()
    r.mano = []
    r.registrar_resto(ganador=1, cartas_restantes=parse_cartas("QP 5C"))
    assert len(r.historial_bazas) == 1
    entrada = r.historial_bazas[0]
    assert entrada.lider is None
    assert entrada.ganador == 1
    assert entrada.tenia_puntos is True


def test_reset_mano_limpia_el_historial():
    r = _recomendador_sin_modelo()
    r.historial_bazas = [EntradaBaza(0, 0, True, False)]
    r.reset_mano([])
    assert r.historial_bazas == []


def test_registrar_pase_izquierda_calcula_receptor_y_dador():
    r = _recomendador_sin_modelo()
    dadas = parse_cartas("2T 3T 4T")
    recibidas = parse_cartas("5T 6T 7T")
    r.registrar_pase("izquierda", dadas, recibidas)
    assert r.receptor == 1  # asiento 0 + 1 = izquierda
    assert r.dador == 3     # asiento 3 me pasó a mí (0 - 1 mod 4)
    assert r.cartas_dadas == [c.id for c in dadas]
    assert r.cartas_recibidas == [c.id for c in recibidas]


def test_registrar_pase_sin_pase_deja_todo_en_none():
    r = _recomendador_sin_modelo()
    r.registrar_pase("sin", [], [])
    assert r.receptor is None
    assert r.dador is None
    assert r.cartas_dadas == []
    assert r.cartas_recibidas == []


def test_reset_mano_limpia_el_estado_del_pase():
    r = _recomendador_sin_modelo()
    r.registrar_pase("izquierda", parse_cartas("2T"), parse_cartas("3T"))
    r.reset_mano([])
    assert r.receptor is None
    assert r.dador is None
    assert r.cartas_dadas == []
    assert r.cartas_recibidas == []


def test_motor_no_fabrica_manos_rivales():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    m = r._motor(mesa=[])
    assert m.jugadores[0].mano == r.mano
    for i in (1, 2, 3):
        assert m.jugadores[i].mano == []


def test_obs_usa_el_estimador_moon_y_cae_a_cero_sin_pesos():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    m = r._motor(mesa=[])
    obs = r._obs(m)
    assert obs[187] == 0.0  # moon_prob_agente: sin pesos entrenados -> 0.0
    assert obs[188] == 0.0  # moon_prob_rival: idem
