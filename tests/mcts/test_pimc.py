"""
Tests TDD para el módulo PIMC (Perfect Information Monte Carlo).

Orden TDD: los tests definen el contrato de la API antes de la implementación.
"""

import pytest
import numpy as np

from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_evasivo


# ============================================================
# Helpers compartidos
# ============================================================

def _jugar_n_bazas(motor: MotorCorazones, n: int) -> None:
    """Juega N bazas completas usando BotExperto para todos los jugadores."""
    bots = [BotExperto() for _ in range(4)]
    for _ in range(n):
        for _ in range(4):
            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)
            carta = bots[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)
        motor.resolver_baza()


def motor_a_mitad_mano(seed: int = 42, bazas_jugadas: int = 4) -> MotorCorazones:
    """Motor tras N bazas completas, reproducible."""
    import random
    random.seed(seed)
    motor = MotorCorazones()
    motor.repartir()
    _jugar_n_bazas(motor, bazas_jugadas)
    return motor


def agente_actual(motor: MotorCorazones) -> int:
    """Retorna el índice del jugador cuyo turno es el actual."""
    return motor.obtener_jugador_actual()


# ============================================================
# TestDeterminizar
# ============================================================

class TestDeterminizar:
    """Verifica que determinizar() crea copias válidas del estado del motor."""

    def test_retorna_nuevo_objeto(self):
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano()
        clon = determinizar(motor, 0)
        assert clon is not motor

    def test_suma_cartas_52(self):
        """Todas las 52 cartas están presentes en el clon."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=5)
        clon = determinizar(motor, 1)
        total = (
            sum(len(j.mano) for j in clon.jugadores)
            + sum(len(j.bazas_ganadas) for j in clon.jugadores)
            + len(clon.mesa)
        )
        assert total == 52

    def test_cartas_unicas_sin_duplicados(self):
        """No hay cartas repetidas en el clon."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=4)
        clon = determinizar(motor, 0, rng=np.random.default_rng(7))
        todas = []
        for j in clon.jugadores:
            todas.extend(j.mano)
            todas.extend(j.bazas_ganadas)
        todas.extend(c for _, c in clon.mesa)
        assert len(todas) == len(set(c.id for c in todas)), "Cartas duplicadas detectadas"

    def test_mano_agente_preservada_exactamente(self):
        """Las cartas del agente no cambian tras determinizar."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=3)
        for agente in range(4):
            mano_original = frozenset(c.id for c in motor.jugadores[agente].mano)
            clon = determinizar(motor, agente)
            mano_clon = frozenset(c.id for c in clon.jugadores[agente].mano)
            assert mano_clon == mano_original, f"Mano del agente {agente} modificada"

    def test_bazas_ganadas_preservadas(self):
        """El cementerio (bazas ganadas) no cambia."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=5)
        clon = determinizar(motor, 2)
        for i in range(4):
            orig = frozenset(c.id for c in motor.jugadores[i].bazas_ganadas)
            clon_set = frozenset(c.id for c in clon.jugadores[i].bazas_ganadas)
            assert orig == clon_set, f"Bazas ganadas del jugador {i} modificadas"

    def test_numero_cartas_oponentes_correcto(self):
        """Cada oponente recibe exactamente las mismas N cartas que tenía."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=6)
        clon = determinizar(motor, 0)
        for i in range(4):
            assert len(clon.jugadores[i].mano) == len(motor.jugadores[i].mano)

    def test_estado_publico_preservado(self):
        """corazones_rotos, numero_baza, palo_de_salida, mesa se preservan."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=4)
        clon = determinizar(motor, 0)
        assert clon.corazones_rotos == motor.corazones_rotos
        assert clon.numero_baza == motor.numero_baza
        assert clon.palo_de_salida == motor.palo_de_salida
        assert clon.indice_jugador_inicial == motor.indice_jugador_inicial
        # Mesa debe tener las mismas cartas en el mismo orden
        assert len(clon.mesa) == len(motor.mesa)
        for (idx_c, carta_c), (idx_o, carta_o) in zip(clon.mesa, motor.mesa):
            assert idx_c == idx_o
            assert carta_c.id == carta_o.id

    def test_voids_respetados_cuando_se_proveen(self):
        """Si oponente X es void en palo P, el clon no le asigna cartas de P."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=3)
        agente = agente_actual(motor)
        oponente = (agente + 1) % 4
        if len(motor.jugadores[oponente].mano) == 0:
            pytest.skip("Oponente sin cartas")
        vacios = {oponente: {3}}  # oponente void en corazones
        for _ in range(30):
            clon = determinizar(motor, agente, vacios=vacios, rng=np.random.default_rng())
            for carta in clon.jugadores[oponente].mano:
                assert carta.palo != 3, "Void en corazones violado"

    def test_determinizacion_reproducible_con_semilla(self):
        """Con la misma semilla, la distribución es idéntica."""
        from src.mcts.pimc import determinizar
        motor = motor_a_mitad_mano(bazas_jugadas=4)
        clon1 = determinizar(motor, 0, rng=np.random.default_rng(42))
        clon2 = determinizar(motor, 0, rng=np.random.default_rng(42))
        for i in range(4):
            ids1 = sorted(c.id for c in clon1.jugadores[i].mano)
            ids2 = sorted(c.id for c in clon2.jugadores[i].mano)
            assert ids1 == ids2

    def test_mesa_actual_no_modificada(self):
        """Las cartas en la mesa actual se respetan tal cual."""
        from src.mcts.pimc import determinizar
        # Motor con 1 baza incompleta (2 cartas ya jugadas)
        import random
        random.seed(5)
        motor = MotorCorazones()
        motor.repartir()
        # Jugar 2 cartas en la baza actual
        for _ in range(2):
            idx = motor.obtener_jugador_actual()
            legales = motor.obtener_jugadas_legales(idx)
            motor.jugar_carta(idx, legales[0])
        mesa_ids = [(idx, c.id) for idx, c in motor.mesa]
        clon = determinizar(motor, motor.obtener_jugador_actual())
        clon_mesa_ids = [(idx, c.id) for idx, c in clon.mesa]
        assert clon_mesa_ids == mesa_ids


# ============================================================
# TestSimularRestoMano
# ============================================================

class TestSimularRestoMano:
    """Verifica que simular_resto_mano() completa la mano correctamente.

    IMPORTANTE: simular_resto_mano() debe llamarse solo cuando es el turno
    del agente (motor.obtener_jugador_actual() == agente_idx). Los tests
    usan agente_actual() para garantizar esto.
    """

    def _crear_bots(self):
        return {i: bot_evasivo for i in range(4)}

    def test_retorna_entero_en_rango_valido(self):
        """La puntuación del agente está entre 0 y 26."""
        from src.mcts.pimc import determinizar, simular_resto_mano
        motor = motor_a_mitad_mano(bazas_jugadas=5)
        agente = agente_actual(motor)  # usa el jugador cuyo turno es
        clon = determinizar(motor, agente, rng=np.random.default_rng(1))
        legales = clon.obtener_jugadas_legales(agente)
        puntos = simular_resto_mano(clon, agente, legales[0], self._crear_bots())
        assert isinstance(puntos, int)
        assert 0 <= puntos <= 26

    def test_manos_vacias_al_terminar(self):
        """Tras el rollout, todos los jugadores tienen la mano vacía."""
        from src.mcts.pimc import determinizar, simular_resto_mano
        motor = motor_a_mitad_mano(bazas_jugadas=6)
        agente = agente_actual(motor)
        clon = determinizar(motor, agente, rng=np.random.default_rng(2))
        legales = clon.obtener_jugadas_legales(agente)
        simular_resto_mano(clon, agente, legales[0], self._crear_bots())
        for j in clon.jugadores:
            assert len(j.mano) == 0

    def test_mesa_vacia_al_terminar(self):
        """La mesa queda vacía al terminar."""
        from src.mcts.pimc import determinizar, simular_resto_mano
        motor = motor_a_mitad_mano(bazas_jugadas=7)
        agente = agente_actual(motor)
        clon = determinizar(motor, agente, rng=np.random.default_rng(3))
        legales = clon.obtener_jugadas_legales(agente)
        simular_resto_mano(clon, agente, legales[0], self._crear_bots())
        assert len(clon.mesa) == 0

    def test_suma_puntos_consistente(self):
        """La suma de puntos es 26 sin pleno o 78 con pleno (3×26+0)."""
        from src.mcts.pimc import determinizar, simular_resto_mano
        motor = motor_a_mitad_mano(bazas_jugadas=4)
        agente = agente_actual(motor)
        clon = determinizar(motor, agente, rng=np.random.default_rng(9))
        legales = clon.obtener_jugadas_legales(agente)
        simular_resto_mano(clon, agente, legales[0], {i: bot_evasivo for i in range(4)})
        puntos_finales = clon.calcular_puntuacion_mano()
        # Pleno: 3×26 = 78; normal: 26
        assert sum(puntos_finales) in {26, 78}

    def test_primera_carta_debe_ser_legal(self):
        """Jugar una carta ilegal lanza ValueError."""
        from src.mcts.pimc import determinizar, simular_resto_mano
        motor = motor_a_mitad_mano(bazas_jugadas=3)
        agente = agente_actual(motor)
        clon = determinizar(motor, agente)
        legales = clon.obtener_jugadas_legales(agente)
        ilegales = [c for c in clon.jugadores[agente].mano if c not in legales]
        if not ilegales:
            pytest.skip("No hay cartas ilegales disponibles para este estado")
        with pytest.raises(ValueError):
            simular_resto_mano(clon, agente, ilegales[0], self._crear_bots())

    def test_funciona_en_ultima_baza(self):
        """Funciona correctamente cuando quedan solo 4 cartas (última baza)."""
        from src.mcts.pimc import determinizar, simular_resto_mano
        motor = motor_a_mitad_mano(bazas_jugadas=12)  # quedan 4 cartas total
        agente = agente_actual(motor)
        clon = determinizar(motor, agente)
        legales = clon.obtener_jugadas_legales(agente)
        puntos = simular_resto_mano(clon, agente, legales[0], self._crear_bots())
        assert 0 <= puntos <= 26


# ============================================================
# TestPIMCMejorJugada
# ============================================================

class TestPIMCMejorJugada:
    """Verifica el algoritmo PIMC completo.

    Todos los tests usan agente_actual(motor) para garantizar que
    se llama PIMC cuando es el turno correcto del agente.
    """

    def test_retorna_carta_de_las_legales(self):
        """La carta retornada siempre está entre las jugadas legales."""
        from src.mcts.pimc import pimc_mejor_jugada
        motor = motor_a_mitad_mano(bazas_jugadas=5)
        agente = agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)
        mejor = pimc_mejor_jugada(motor, agente, legales, num_mundos=5)
        assert mejor in legales

    def test_unica_carta_legal_retornada_sin_simulacion(self):
        """Con 1 carta legal, la retorna directamente (sin MCTS)."""
        from src.mcts.pimc import pimc_mejor_jugada
        import random
        for seed in range(200):
            random.seed(seed)
            motor = MotorCorazones()
            motor.repartir()
            for _ in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)
                if len(legales) == 1:
                    mejor = pimc_mejor_jugada(motor, idx, legales, num_mundos=0)
                    assert mejor == legales[0]
                    return
                motor.jugar_carta(idx, legales[0])
                if len(motor.mesa) == 4:
                    motor.resolver_baza()
        pytest.skip("No se encontró estado con 1 carta legal en 200 intentos")

    def test_motor_original_no_modificado(self):
        """El motor original NO se modifica durante el PIMC."""
        from src.mcts.pimc import pimc_mejor_jugada
        motor = motor_a_mitad_mano(bazas_jugadas=4)
        agente = agente_actual(motor)
        mano_antes = [c.id for c in motor.jugadores[agente].mano]
        baza_antes = motor.numero_baza
        legales = motor.obtener_jugadas_legales(agente)
        pimc_mejor_jugada(motor, agente, legales, num_mundos=10)
        assert [c.id for c in motor.jugadores[agente].mano] == mano_antes
        assert motor.numero_baza == baza_antes

    def test_reproducible_con_semilla(self):
        """Con la misma semilla, la carta elegida es siempre la misma."""
        from src.mcts.pimc import pimc_mejor_jugada
        motor = motor_a_mitad_mano(bazas_jugadas=5)
        agente = agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)
        if len(legales) < 2:
            pytest.skip("Solo 1 carta legal")
        mejor1 = pimc_mejor_jugada(motor, agente, legales, num_mundos=20,
                                   rng=np.random.default_rng(999))
        mejor2 = pimc_mejor_jugada(motor, agente, legales, num_mundos=20,
                                   rng=np.random.default_rng(999))
        assert mejor1.id == mejor2.id

    def test_no_tomar_q_spades_innecesariamente(self):
        """Cuando el agente tiene Q♠ y alternativas sin picas, PIMC no lidera con Q♠."""
        from src.mcts.pimc import pimc_mejor_jugada
        import random
        for seed in range(500):
            random.seed(seed)
            motor = MotorCorazones()
            motor.repartir()
            _jugar_n_bazas(motor, 3)
            agente = agente_actual(motor)
            if motor.mesa:  # solo cuando el agente lidera (mesa vacía)
                continue
            mano = motor.jugadores[agente].mano
            tiene_q = any(c.es_dama_de_picas for c in mano)
            legales = motor.obtener_jugadas_legales(agente)
            tiene_alternativas = any(not c.es_dama_de_picas and c.palo != 2
                                     for c in legales)
            if tiene_q and tiene_alternativas:
                mejor = pimc_mejor_jugada(motor, agente, legales, num_mundos=20,
                                          rng=np.random.default_rng(seed))
                assert not mejor.es_dama_de_picas, (
                    f"PIMC eligio liderar con Q picas (seed={seed}), lo cual es suboptimo"
                )
                return
        pytest.skip("No se encontro estado apropiado en 500 intentos")

    def test_scores_validos_por_carta(self):
        """Los scores calculados por carta estan en [0, 26]."""
        from src.mcts.pimc import _puntaje_esperado_por_carta
        motor = motor_a_mitad_mano(bazas_jugadas=5)
        agente = agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)
        if len(legales) < 2:
            pytest.skip("Solo 1 carta legal")
        scores = _puntaje_esperado_por_carta(motor, agente, legales,
                                             num_mundos=20,
                                             rng=np.random.default_rng(77))
        assert len(scores) == len(legales)
        assert all(0 <= s <= 26 for s in scores.values())
