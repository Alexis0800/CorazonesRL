"""
Tests del pase (v10b) en MotorCorazones.

Cubre la rotación de dirección, los receptores, el intercambio de 3 cartas
(conservando 13 por mano) y el recálculo del portador del 2♣ tras el pase.
"""
from __future__ import annotations

from src.dominio.motor import MotorCorazones, receptor_y_dador_por_numero_mano
from src.dominio.carta import Carta


def _dos_treboles() -> Carta:
    return Carta._TODAS[0]  # id 0 = 2♣


class TestDireccionPase:
    def test_rotacion(self):
        motor = MotorCorazones()
        motor.numero_mano = 1
        assert motor.direccion_pase() == "izquierda"
        motor.numero_mano = 2
        assert motor.direccion_pase() == "derecha"
        motor.numero_mano = 3
        assert motor.direccion_pase() == "enfrente"
        motor.numero_mano = 4
        assert motor.direccion_pase() is None
        motor.numero_mano = 5
        assert motor.direccion_pase() == "izquierda"

    def test_receptor_segun_direccion(self):
        motor = MotorCorazones()
        motor.numero_mano = 1  # izquierda → +1
        assert motor.receptor_pase(0) == 1
        assert motor.receptor_pase(3) == 0
        motor.numero_mano = 2  # derecha → +3
        assert motor.receptor_pase(0) == 3
        motor.numero_mano = 3  # enfrente → +2
        assert motor.receptor_pase(0) == 2
        motor.numero_mano = 4  # sin pase
        assert motor.receptor_pase(0) is None


class TestReceptorYDadorPorNumeroMano:
    def test_izquierda(self):
        receptor, dador = receptor_y_dador_por_numero_mano(1, 0)
        assert receptor == 1  # seat+1 = izquierda
        assert dador == 3     # seat-1 me pasó a mí

    def test_derecha(self):
        receptor, dador = receptor_y_dador_por_numero_mano(2, 0)
        assert receptor == 3
        assert dador == 1

    def test_enfrente(self):
        receptor, dador = receptor_y_dador_por_numero_mano(3, 0)
        assert receptor == 2
        assert dador == 2

    def test_sin_pase(self):
        assert receptor_y_dador_por_numero_mano(4, 0) == (None, None)

    def test_coincide_con_receptor_pase_para_los_4_asientos(self):
        motor = MotorCorazones()
        motor.numero_mano = 1
        for seat in range(4):
            receptor, dador = receptor_y_dador_por_numero_mano(1, seat)
            assert receptor == motor.receptor_pase(seat)
            assert motor.receptor_pase(dador) == seat


class TestNumeroMano:
    def test_nueva_partida_arranca_en_mano_1(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        assert motor.numero_mano == 1

    def test_repartir_incrementa_mano(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        motor.repartir()
        assert motor.numero_mano == 2


class TestEjecutarPase:
    def test_intercambio_mueve_cartas_y_conserva_13(self):
        motor = MotorCorazones()
        motor.nueva_partida()  # mano 1 = izquierda (+1)
        # Cada jugador pasa sus primeras 3 cartas.
        selecciones = {i: list(motor.jugadores[i].mano[:3]) for i in range(4)}
        esperado_receptor = {i: (i + 1) % 4 for i in range(4)}

        motor.ejecutar_pase(selecciones)

        # Cada jugador conserva 13 cartas.
        assert all(len(j.mano) == 13 for j in motor.jugadores)
        # Las cartas pasadas por i están ahora en la mano del receptor.
        for i, cartas in selecciones.items():
            receptor = esperado_receptor[i]
            for c in cartas:
                assert c in motor.jugadores[receptor].mano

    def test_mano_sin_pase_es_noop(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        motor.numero_mano = 4  # sin pase
        manos_antes = [list(j.mano) for j in motor.jugadores]
        motor.ejecutar_pase({i: list(motor.jugadores[i].mano[:3]) for i in range(4)})
        manos_despues = [list(j.mano) for j in motor.jugadores]
        assert manos_antes == manos_despues

    def test_pasar_3_cartas_obligatorio(self):
        import pytest
        motor = MotorCorazones()
        motor.nueva_partida()
        with pytest.raises(ValueError):
            motor.ejecutar_pase({0: list(motor.jugadores[0].mano[:2]),
                                 1: [], 2: [], 3: []})

    def test_dos_treboles_cambia_jugador_inicial(self):
        motor = MotorCorazones()
        motor.nueva_partida()  # izquierda (+1)
        # Localizar quién tiene el 2♣ y forzar que lo pase.
        dueno = next(i for i in range(4)
                     if any(c.es_dos_de_treboles for c in motor.jugadores[i].mano))
        # Selección del dueño incluye el 2♣; los demás pasan 3 cualquiera.
        sel = {}
        for i in range(4):
            mano = motor.jugadores[i].mano
            if i == dueno:
                otras = [c for c in mano if not c.es_dos_de_treboles][:2]
                sel[i] = [_dos_treboles()] + otras
            else:
                sel[i] = list(mano[:3])
        motor.ejecutar_pase(sel)
        # El 2♣ pasó al receptor → ahora él abre.
        assert motor.indice_jugador_inicial == (dueno + 1) % 4
        assert any(c.es_dos_de_treboles
                   for c in motor.jugadores[(dueno + 1) % 4].mano)
