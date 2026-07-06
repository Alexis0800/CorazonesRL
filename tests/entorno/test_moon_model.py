"""Tests de src/entorno/moon_model.py."""
from __future__ import annotations

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.moon_model import (
    DIM_PROPIO,
    DIM_RIVAL,
    EntradaBaza,
    _alguien_mas_tiene_puntos,
    _ganador_parcial,
    _ratio_bazas_con_puntos,
    _tasa_lidero_corazon_dama,
    features_propio,
)


def _carta(palo, valor):
    return Carta(palo, valor)


class TestAlguienMasTienePuntos:
    def test_falso_si_nadie_ha_capturado_puntos(self):
        m = MotorCorazones()
        m.repartir()
        assert _alguien_mas_tiene_puntos(m, 0) is False

    def test_verdadero_si_otro_jugador_capturo_un_corazon(self):
        m = MotorCorazones()
        m.repartir()
        m.jugadores[1].bazas_ganadas = [_carta(3, 5)]  # 5 de corazones
        assert _alguien_mas_tiene_puntos(m, 0) is True

    def test_ignora_los_puntos_propios_del_objetivo(self):
        m = MotorCorazones()
        m.repartir()
        m.jugadores[0].bazas_ganadas = [_carta(3, 5)]
        assert _alguien_mas_tiene_puntos(m, 0) is False


class TestGanadorParcial:
    def test_none_si_mesa_vacia(self):
        m = MotorCorazones()
        m.repartir()
        assert _ganador_parcial(m) is None

    def test_gana_la_carta_mas_alta_del_palo_de_salida(self):
        m = MotorCorazones()
        m.repartir()
        m.mesa = [(0, _carta(0, 5)), (1, _carta(0, 10)), (2, _carta(3, 14))]
        m.palo_de_salida = 0
        assert _ganador_parcial(m) == 1  # el As de corazones no sigue el palo

    def test_no_seguir_el_palo_no_gana(self):
        m = MotorCorazones()
        m.repartir()
        m.mesa = [(0, _carta(0, 5)), (1, _carta(3, 14))]
        m.palo_de_salida = 0
        assert _ganador_parcial(m) == 0


class TestRatioBazasConPuntos:
    def test_cero_sin_historial(self):
        assert _ratio_bazas_con_puntos([], 0) == 0.0

    def test_ignora_bazas_sin_puntos(self):
        historial = [
            EntradaBaza(lider=0, ganador=0, tenia_puntos=False, lidero_corazon_o_dama=False),
        ]
        assert _ratio_bazas_con_puntos(historial, 0) == 0.0

    def test_calcula_la_razon_correcta(self):
        historial = [
            EntradaBaza(lider=0, ganador=1, tenia_puntos=True, lidero_corazon_o_dama=False),
            EntradaBaza(lider=1, ganador=1, tenia_puntos=True, lidero_corazon_o_dama=False),
            EntradaBaza(lider=2, ganador=2, tenia_puntos=False, lidero_corazon_o_dama=False),
        ]
        assert _ratio_bazas_con_puntos(historial, 1) == 1.0
        assert _ratio_bazas_con_puntos(historial, 0) == 0.0


class TestTasaLideroCorazonDama:
    def test_cero_si_nunca_lidero(self):
        historial = [
            EntradaBaza(lider=1, ganador=1, tenia_puntos=True, lidero_corazon_o_dama=True),
        ]
        assert _tasa_lidero_corazon_dama(historial, 0) == 0.0

    def test_calcula_la_tasa_correcta(self):
        historial = [
            EntradaBaza(lider=0, ganador=0, tenia_puntos=True, lidero_corazon_o_dama=True),
            EntradaBaza(lider=0, ganador=1, tenia_puntos=False, lidero_corazon_o_dama=False),
        ]
        assert _tasa_lidero_corazon_dama(historial, 0) == 0.5


def test_dimensiones_publicadas():
    assert DIM_PROPIO == 333
    assert DIM_RIVAL == 272


class TestFeaturesPropio:
    def test_forma_y_rango(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        historial = []
        feats = features_propio(
            m, 0, vacios, historial, [], [],
            [0, 0, 0, 0], [0, 0, 0, 0], None,
        )
        assert feats.shape == (DIM_PROPIO,)
        assert feats.dtype == np.float32
        # los 2 slots de moon_prob del vector v13 reutilizado deben quedar en 0
        assert feats[187] == 0.0
        assert feats[188] == 0.0

    def test_mano_propia_se_refleja_en_el_one_hot(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        feats = features_propio(m, 0, vacios, [], [], [], [0, 0, 0, 0], [0, 0, 0, 0], None)
        for c in m.jugadores[0].mano:
            assert feats[c.id] == 1.0

    def test_ratio_bazas_con_puntos_al_final_del_vector(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        historial = [
            EntradaBaza(lider=0, ganador=0, tenia_puntos=True, lidero_corazon_o_dama=False),
        ]
        feats = features_propio(m, 0, vacios, historial, [], [], [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert feats[-1] == 1.0  # gané la única baza-con-puntos jugada
