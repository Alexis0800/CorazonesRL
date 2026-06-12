"""
Tests para jugar_contra_modelo.py — parseo, política modelo, integración.
"""

from src.motor import MotorCorazones
from src.carta import Carta
from jugar_contra_modelo import (
    carta_a_str,
    parsear_carta,
    _crear_politica_modelo,
)
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCartaAStr:
    """Pruebas para conversión Carta → string."""

    def test_as_corazones(self) -> None:
        c = Carta(3, 14)
        assert carta_a_str(c) == "A♥"

    def test_dos_treboles(self) -> None:
        c = Carta(0, 2)
        assert carta_a_str(c) == "2♣"

    def test_dama_picas(self) -> None:
        c = Carta(2, 12)
        assert carta_a_str(c) == "Q♠"

    def test_diez_diamantes(self) -> None:
        c = Carta(1, 10)
        assert carta_a_str(c) == "10♦"


class TestParsearCartaJugar:
    """Pruebas para parseo de texto a ID (versión jugar_contra_modelo)."""

    def test_parsear_valido(self) -> None:
        assert parsear_carta("A♥") == 51
        assert parsear_carta("2♣") == 0
        assert parsear_carta("Q♠") == 36

    def test_parsear_vacio(self) -> None:
        with pytest.raises(ValueError):
            parsear_carta("")

    def test_parsear_palo_invalido(self) -> None:
        with pytest.raises(ValueError, match="Palo desconocido"):
            parsear_carta("A★")


class TestCrearPoliticaModelo:
    """Pruebas para la fábrica de política RL."""

    def test_politica_devuelve_carta_legal(self) -> None:
        """Verifica que la política devuelve una carta de entre las legales."""
        # Crear un motor con estado conocido
        motor = MotorCorazones()
        motor.repartir()

        # Mock del modelo: siempre elige acción 0
        class MockModel:
            def predict(self, obs, action_masks=None, deterministic=True):
                # Encontrar primera acción legal
                if action_masks is not None and np.any(action_masks):
                    return np.array([int(np.argmax(action_masks))]), None
                return np.array([0]), None

        politica = _crear_politica_modelo(MockModel(), None, 1)

        # Obtener legales y llamar
        idx = 1
        legales = motor.obtener_jugadas_legales(idx)
        carta = politica(motor, idx, legales)

        assert carta in legales

    def test_politica_sin_vecnorm_no_crashea(self) -> None:
        """Sin VecNormalize, la política debe funcionar igual."""
        motor = MotorCorazones()
        motor.repartir()

        class MockModel:
            def predict(self, obs, action_masks=None, deterministic=True):
                if action_masks is not None and np.any(action_masks):
                    return np.array([int(np.argmax(action_masks))]), None
                return np.array([0]), None

        politica = _crear_politica_modelo(MockModel(), None, 0)

        legales = motor.obtener_jugadas_legales(0)
        carta = politica(motor, 0, legales)
        assert carta in legales

    def test_politica_respeta_action_mask(self) -> None:
        """La política solo debe devolver cartas legales (respetando la máscara)."""
        motor = MotorCorazones()
        motor.repartir()

        # Mock que elige la primera acción legal usando la máscara
        class MockModel:
            def predict(self, obs, action_masks=None, deterministic=True):
                if action_masks is not None and np.any(action_masks):
                    return np.array([int(np.argmax(action_masks))]), None
                return np.array([0]), None

        politica = _crear_politica_modelo(MockModel(), None, 0)

        legales = motor.obtener_jugadas_legales(0)
        carta = politica(motor, 0, legales)
        assert carta in legales
