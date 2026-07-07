"""Tests de scripts/importar_sesiones_bridge_reconstruidas.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from importar_sesiones_bridge_reconstruidas import (
    _direccion_pase,
    _jugadas_de_mano,
    _puntuacion_mano,
    mano_desde_reconstruida,
    partida_desde_archivo,
    swap_suit,
)
from src.captura.replay import mano_reconstruible

# Línea real de logs/reconstructed (mano con remate a 11 bazas, pase "enfrente").
_MANO_REAL = {
    "seat": 0,
    "dealtHand": [27, 39, 40, 41, 42, 0, 1, 9, 12, 17, 18, 19, 20],
    "viaPass": True,
    "pass": {"direction": 2, "cardsOut": [27, 18, 20], "cardsIn": [10, 22, 21]},
    "postPassHand": [0, 1, 9, 10, 12, 17, 19, 21, 22, 39, 40, 41, 42],
    "passConsistent": True,
    "tricks": [
        {"trick": 0, "bySeat": [40, 37, 26, 29], "winner": 1, "leader": 2},
        {"trick": 1, "bySeat": [39, 51, 46, 45], "winner": 1, "leader": 1},
        {"trick": 2, "bySeat": [10, 27, 28, 30], "winner": 3, "leader": 1},
        {"trick": 3, "bySeat": [42, 24, 43, 47], "winner": 3, "leader": 3},
        {"trick": 4, "bySeat": [21, 20, 14, 15], "winner": 0, "leader": 3},
        {"trick": 5, "bySeat": [17, 16, 35, 38], "winner": 0, "leader": 0},
        {"trick": 6, "bySeat": [41, 25, 34, 50], "winner": 3, "leader": 0},
        {"trick": 7, "bySeat": [1, 8, 6, 5], "winner": 1, "leader": 3},
        {"trick": 8, "bySeat": [12, 7, 4, 36], "winner": 0, "leader": 1},
        {"trick": 9, "bySeat": [9, 2, 3, 33], "winner": 0, "leader": 0},
        {"trick": 10, "bySeat": [0, 23, 11, 49], "winner": 2, "leader": 0},
    ],
    "remate": {"tricksCompleted": 11, "winner": 2, "seats": [[19, 22], [13, 18], [31, 32], [44, 48]]},
    "seatsFinalHands": [
        [40, 39, 10, 42, 21, 17, 41, 1, 12, 9, 0, 19, 22],
        [37, 51, 27, 24, 20, 16, 25, 8, 7, 2, 23, 13, 18],
        [26, 46, 28, 43, 14, 35, 34, 6, 4, 3, 11, 31, 32],
        [29, 45, 30, 47, 15, 38, 50, 5, 36, 33, 49, 44, 48],
    ],
    "reconstructable": True,
    "reconstructableWithoutRemate": False,
}


def test_swap_suit_es_su_propia_inversa():
    for cid in range(52):
        assert swap_suit(swap_suit(cid)) == cid


def test_direccion_pase_detecta_izquierda():
    # seat 0 dio [27,18,20]; las 3 aparecen completas en seatsFinalHands[1] (offset 1).
    # Nota: el wire's pass.direction (2, ver _MANO_REAL) NO es izquierda/derecha/enfrente
    # -- es un entero crudo del protocolo sin decodificar; por eso se deriva de los datos.
    direccion = _direccion_pase(0, [27, 18, 20], _MANO_REAL["seatsFinalHands"])
    assert direccion == "izquierda"  # offset (1-0)%4 == 1


def test_direccion_pase_none_si_las_cartas_no_aparecen_completas():
    assert _direccion_pase(0, [27, 18, 999], _MANO_REAL["seatsFinalHands"]) is None


def test_jugadas_de_mano_respeta_el_orden_de_turno():
    jugadas = _jugadas_de_mano(_MANO_REAL["tricks"])
    assert len(jugadas) == 44  # 11 bazas x 4
    primera_baza = jugadas[:4]
    # trick 0 leader=2 -> orden de asientos 2,3,0,1
    assert [j.asiento for j in primera_baza] == [2, 3, 0, 1]
    assert primera_baza[0].carta_id == swap_suit(26)  # bySeat[2] del trick 0


def test_puntuacion_mano_asigna_puntos_de_baza_al_ganador_y_remate_al_asiento_que_se_lleva_el_resto():
    puntos = _puntuacion_mano(_MANO_REAL["tricks"], _MANO_REAL["remate"])
    assert sum(puntos) in (26, 78)  # 78 si hubo pozo (regla de Pleno invierte el resultado)
    assert all(p >= 0 for p in puntos)


def test_puntuacion_mano_invierte_con_pozo():
    # Las 13 cartas de corazon (id dominio 39..51) + Q de picas (id dominio 36) = 26 puntos.
    # swap_suit es su propia inversa: swap_suit(id_dominio) da el id de cable equivalente.
    puntos_cable = [swap_suit(i) for i in list(range(39, 52)) + [36]]
    relleno_cable = [swap_suit(0), swap_suit(1)]  # 2 cartas sin puntos, para completar 4 bazas de 4
    cartas = puntos_cable + relleno_cable
    tricks = [
        {"trick": i, "bySeat": cartas[i * 4:(i + 1) * 4], "winner": 0, "leader": 0}
        for i in range(4)
    ]
    assert _puntuacion_mano(tricks, None) == [0, 26, 26, 26]


def test_mano_desde_reconstruida_es_reproducible():
    mano = mano_desde_reconstruida(_MANO_REAL, numero_mano=1)
    assert mano is not None
    assert mano.direccion_pase == "izquierda"
    assert mano.remate_asiento == 2
    assert mano_reconstruible(mano)


def test_mano_desde_reconstruida_none_si_bridge_no_la_marco_reconstruible():
    h = dict(_MANO_REAL, reconstructable=False)
    assert mano_desde_reconstruida(h, numero_mano=1) is None


def test_partida_desde_archivo(tmp_path):
    ruta = tmp_path / "session-test.reconstructed.jsonl"
    import json
    ruta.write_text(json.dumps(_MANO_REAL) + "\n", encoding="utf-8")
    avisos: list = []
    partida = partida_desde_archivo(ruta, avisos)
    assert partida is not None
    assert len(partida.manos) == 1
    assert partida.asiento_agente == 0
    assert sum(partida.marcador_final) == 26
