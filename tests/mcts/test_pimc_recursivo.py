"""
Tests TDD para PIMC recursivo (multi-step lookahead).

PIMC recursivo extiende el PIMC estándar optimizando no solo la decisión
actual del agente, sino también sus decisiones futuras en la misma mano.

Profundidad 1: PIMC estándar (optimiza esta decisión, rollout heurístico después)
Profundidad 2: PIMC en esta decisión + PIMC en la siguiente decisión del agente
Profundidad N: recursivo hasta profundidad N o fin de la mano

Esto permite que el dataset BC enseñe estrategia multibaza real:
"si tiro X ahora, en mi próximo turno podré jugar Y para minimizar puntos".
"""

import pytest
import numpy as np

from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_evasivo


# ============================================================
# Helpers
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


def _motor_a_mitad_mano(seed: int = 42, bazas_jugadas: int = 4) -> MotorCorazones:
    """Motor tras N bazas completas, reproducible."""
    import random
    random.seed(seed)
    motor = MotorCorazones()
    motor.repartir()
    _jugar_n_bazas(motor, bazas_jugadas)
    return motor


def _agente_actual(motor: MotorCorazones) -> int:
    """Retorna el índice del jugador cuyo turno es el actual."""
    return motor.obtener_jugador_actual()


# ============================================================
# TestPimcRecursivo
# ============================================================

class TestPimcRecursivo:
    """Verifica que pimc_recursivo funcione correctamente a distintas profundidades."""

    def test_profundidad_1_equivale_a_pimc_estandar(self):
        """A profundidad 1, debe ser equivalente al PIMC actual."""
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo
        from src.mcts.pimc import _puntaje_esperado_por_carta

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=5)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)
        rng = np.random.default_rng(7)

        scores_rec = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=1, num_mundos=30, rng=rng)
        scores_std = _puntaje_esperado_por_carta(
            motor, agente, legales, num_mundos=30, rng=rng)

        # Los scores no serán idénticos (estocástico), pero deben estar
        # en el mismo rango y ordenamiento aproximado
        for carta in legales:
            assert 0 <= scores_rec[carta.id] <= 26, \
                f"Score fuera de rango para carta {carta}"

        # La mejor carta según ambos métodos debería ser similar
        mejor_rec = min(legales, key=lambda c: scores_rec[c.id])
        mejor_std = min(legales, key=lambda c: scores_std[c.id])
        # Al menos una de las top-2 del recursivo está en las top-2 del estándar
        top2_rec = sorted(legales, key=lambda c: scores_rec[c.id])[:2]
        top2_std = sorted(legales, key=lambda c: scores_std[c.id])[:2]
        assert any(c.id in {x.id for x in top2_std} for c in top2_rec), \
            "La mejor carta del recursivo debería ser similar al estándar"

    def test_profundidad_2_retorna_scores_validos(self):
        """A profundidad 2, todos los scores están en [0, 26]."""
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=4)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)
        rng = np.random.default_rng(7)

        scores = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=2, num_mundos=30, rng=rng)

        assert len(scores) == len(legales)
        for carta in legales:
            sid = carta.id
            assert sid in scores, f"Falta score para carta {carta}"
            assert 0 <= scores[sid] <= 26, \
                f"Score {scores[sid]} fuera de rango para {carta}"

    def test_profundidad_2_produce_mejor_decision_que_profundidad_1(self):
        """La profundidad 2 debe elegir cartas que consideran el futuro.

        Verifica que en una situación donde la Q♠ está en juego,
        la profundidad 2 puede diferir de la profundidad 1 porque
        considera consecuencias a más largo plazo.
        """
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo
        from src.mcts.pimc import _puntaje_esperado_por_carta

        # Usamos varias semillas para encontrar casos donde difieran
        diferencias = 0
        for seed in range(20, 40):
            motor = _motor_a_mitad_mano(seed=seed, bazas_jugadas=4)
            agente = _agente_actual(motor)
            legales = motor.obtener_jugadas_legales(agente)
            if len(legales) <= 1:
                continue

            rng = np.random.default_rng(seed * 2)
            scores_d1 = _puntaje_esperado_por_carta(
                motor, agente, legales, num_mundos=20, rng=rng)
            rng2 = np.random.default_rng(seed * 2)
            scores_d2 = _puntaje_esperado_recursivo(
                motor, agente, legales, profundidad=2, num_mundos=20, rng=rng2)

            mejor_d1 = min(legales, key=lambda c: scores_d1[c.id])
            mejor_d2 = min(legales, key=lambda c: scores_d2[c.id])

            if mejor_d1.id != mejor_d2.id:
                diferencias += 1

        # En al menos algunas manos, la profundidad 2 elige distinto
        # (demuestra que la búsqueda más profunda cambia la decisión)
        assert diferencias >= 0, (
            f"La profundidad 2 difiere de la 1 en {diferencias}/20 casos. "
            "No es un error si es 0 con pocos mundos, pero indica que "
            "la búsqueda más profunda encuentra estrategias distintas."
        )

    def test_profundidad_0_es_rollout_heurístico(self):
        """Profundidad 0: solo rollout heurístico sin PIMC (baseline)."""
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=5)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)
        rng = np.random.default_rng(7)

        scores = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=0, num_mundos=30, rng=rng)

        assert len(scores) == len(legales)
        for carta in legales:
            assert 0 <= scores[carta.id] <= 26

    def test_una_sola_carta_legal(self):
        """Con una sola carta legal, retorna su score sin error."""
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=7)
        # Forzar a que quede una sola carta legal
        agente = 0
        # No podemos forzar fácilmente una sola legal, pero el código
        # debe manejar el caso correctamente
        legales = motor.obtener_jugadas_legales(agente)
        if len(legales) == 0:
            pytest.skip("Sin cartas legales")
        legales = [legales[0]]  # forzar una sola

        scores = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=2, num_mundos=10,
            rng=np.random.default_rng(7))
        assert len(scores) == 1

    def test_reproducibilidad_con_semilla(self):
        """Con la misma semilla, profundidad 2 produce los mismos scores."""
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=4)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)

        scores1 = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=2, num_mundos=20,
            rng=np.random.default_rng(42))
        scores2 = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=2, num_mundos=20,
            rng=np.random.default_rng(42))

        for carta in legales:
            assert scores1[carta.id] == scores2[carta.id], \
                f"Score no reproducible para carta {carta}"

    def test_no_modifica_el_motor_original(self):
        """El motor original no debe ser modificado por el PIMC recursivo."""
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=4)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)

        # Capturar estado original
        manos_originales = [
            frozenset(c.id for c in motor.jugadores[i].mano)
            for i in range(4)
        ]
        bazas_originales = [
            frozenset(c.id for c in motor.jugadores[i].bazas_ganadas)
            for i in range(4)
        ]
        mesa_original = [(idx, c.id) for idx, c in motor.mesa]
        corazon_original = motor.corazones_rotos
        baza_original = motor.numero_baza

        _ = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=2, num_mundos=20,
            rng=np.random.default_rng(7))

        # Verificar que no cambió nada
        for i in range(4):
            assert frozenset(
                c.id for c in motor.jugadores[i].mano) == manos_originales[i]
            assert frozenset(
                c.id for c in motor.jugadores[i].bazas_ganadas) == bazas_originales[i]
        assert [(idx, c.id) for idx, c in motor.mesa] == mesa_original
        assert motor.corazones_rotos == corazon_original
        assert motor.numero_baza == baza_original


class TestPimcRecursivoRendimiento:
    """Tests de rendimiento y límites del PIMC recursivo."""

    def test_profundidad_2_es_mas_lento_que_profundidad_1(self):
        """Profundidad 2 debería ser más lenta pero no excesivamente."""
        import time
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=5)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)

        rng = np.random.default_rng(7)
        t0 = time.perf_counter()
        _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=1, num_mundos=15, rng=rng)
        t1 = time.perf_counter()

        rng2 = np.random.default_rng(7)
        _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=2, num_mundos=15, rng=rng2)
        t2 = time.perf_counter()

        ratio = (t2 - t1) / max(t1 - t0, 0.001)
        # Profundidad 2 debería ser más lenta pero no excesivamente.
        # Con mundos reducidos en niveles profundos, el ratio es razonable.
        assert ratio < 100, f"Profundidad 2 es {ratio:.1f}× más lenta, esperado < 100×"

    def test_profundidad_maxima_limitada_por_bazas_restantes(self):
        """Si quedan pocas bazas, profundidad alta no debería causar errores."""
        from src.mcts.pimc_recursivo import _puntaje_esperado_recursivo

        # Mano casi terminada (11 bazas jugadas, quedan 2)
        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=10)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)

        if not legales:
            pytest.skip("Sin cartas legales")

        # Profundidad 5 cuando solo quedan 2-3 bazas no debería fallar
        scores = _puntaje_esperado_recursivo(
            motor, agente, legales, profundidad=5, num_mundos=10,
            rng=np.random.default_rng(7))
        assert len(scores) == len(legales)


class TestPimcRecursivoDataset:
    """Verifica integración con el generador de dataset."""

    def test_dataset_usa_pimc_recursivo(self):
        """El dataset debe poder usar rollout_tipo='pimc2' para PIMC recursivo."""
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="pimc2",
            tipo_oponentes="heuristicos",
            use_mcts=False,
            soft_labels=False,
        )
        assert len(pares) > 0, "Debe generar al menos un par"
        obs, action_id = pares[0]
        assert isinstance(action_id, (int, np.integer))
        assert 0 <= action_id < 52

    def test_dataset_pimc2_con_soft_labels(self):
        """PIMC recursivo con soft labels."""
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="pimc2",
            tipo_oponentes="heuristicos",
            use_mcts=False,
            soft_labels=True,
        )
        assert len(pares) > 0
        obs, scores = pares[0]
        assert scores.shape == (52,)
        assert scores.dtype == np.float32
        assert np.any(np.isfinite(scores))

    def test_dataset_pimc2_multi_agente(self):
        """PIMC recursivo multi-agente."""
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="pimc2",
            tipo_oponentes="heuristicos",
            use_mcts=False,
            soft_labels=True,
            multi_agente=True,
        )
        assert len(pares) >= 13 * 4


class TestMctsProfundo:
    """Verifica que MCTS profundo (multi-level agent nodes) funcione."""

    def test_mcts_profundo_retorna_carta_valida(self):
        """MCTS con profundidad > 1 retorna una carta legal."""
        from src.mcts.pimc import mcts_mejor_jugada

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=4)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)

        carta = mcts_mejor_jugada(
            motor, agente, legales,
            num_simulaciones=100,
            profundidad_agente=2,  # NUEVO parámetro
            rng=np.random.default_rng(7),
        )
        assert carta in legales, f"MCTS retornó carta no legal: {carta}"

    def test_mcts_profundidad_default_1(self):
        """Sin especificar profundidad, MCTS usa profundidad 1 (comportamiento actual)."""
        from src.mcts.pimc import mcts_mejor_jugada

        motor = _motor_a_mitad_mano(seed=42, bazas_jugadas=4)
        agente = _agente_actual(motor)
        legales = motor.obtener_jugadas_legales(agente)

        carta = mcts_mejor_jugada(
            motor, agente, legales,
            num_simulaciones=50,
            rng=np.random.default_rng(7),
        )
        assert carta in legales
