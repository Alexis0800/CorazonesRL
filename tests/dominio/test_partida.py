"""
Tests del nivel de partida completa en MotorCorazones (v10).

Cubre nueva_partida, puntuaciones_historicas, partida_terminada y ranking_partida,
y que el marcador persiste entre manos dentro de una partida.
"""
from __future__ import annotations

from src.dominio.motor import MotorCorazones


class TestNuevaPartida:
    def test_nueva_partida_pone_marcador_a_cero(self):
        motor = MotorCorazones()
        for j in motor.jugadores:
            j.puntuacion_historica = 50
        motor.nueva_partida()
        assert motor.puntuaciones_historicas() == [0, 0, 0, 0]

    def test_nueva_partida_reparte_mano(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        assert all(len(j.mano) == 13 for j in motor.jugadores)
        assert motor.numero_baza == 1


class TestMarcadorPersistente:
    def test_repartir_conserva_marcador(self):
        """repartir() (nueva mano) NO debe borrar la puntuación acumulada."""
        motor = MotorCorazones()
        motor.nueva_partida()
        motor.jugadores[0].puntuacion_historica = 17
        motor.jugadores[2].puntuacion_historica = 5
        motor.repartir()
        assert motor.jugadores[0].puntuacion_historica == 17
        assert motor.jugadores[2].puntuacion_historica == 5

    def test_jugar_varias_manos_acumula(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        total_por_mano = []
        for _ in range(3):
            puntos = motor.jugar_mano()  # reparte internamente, conserva acumulado
            total_por_mano.append(sum(puntos))
        # Cada mano reparte 26 puntos (salvo pozo: 26*3=78). Acumulado coherente.
        suma_acumulada = sum(motor.puntuaciones_historicas())
        assert suma_acumulada == sum(total_por_mano)


class TestPartidaTerminada:
    def test_no_terminada_al_inicio(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        assert not motor.partida_terminada(100)

    def test_terminada_cuando_alguien_llega_al_limite(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        motor.jugadores[1].puntuacion_historica = 100
        assert motor.partida_terminada(100)

    def test_limite_configurable(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        motor.jugadores[1].puntuacion_historica = 55
        assert not motor.partida_terminada(100)
        assert motor.partida_terminada(50)


class TestRanking:
    def test_ranking_ordena_por_puntuacion_ascendente(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        motor.jugadores[0].puntuacion_historica = 40
        motor.jugadores[1].puntuacion_historica = 10
        motor.jugadores[2].puntuacion_historica = 99
        motor.jugadores[3].puntuacion_historica = 25
        assert motor.ranking_partida() == [1, 3, 0, 2]

    def test_ganador_es_el_de_menos_puntos(self):
        motor = MotorCorazones()
        motor.nueva_partida()
        motor.jugadores[0].puntuacion_historica = 80
        motor.jugadores[1].puntuacion_historica = 30
        motor.jugadores[2].puntuacion_historica = 5
        motor.jugadores[3].puntuacion_historica = 60
        ganador = motor.ranking_partida()[0]
        assert ganador == 2
