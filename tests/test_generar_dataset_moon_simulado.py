"""Tests de scripts/generar_dataset_moon_simulado.py."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generar_dataset_moon_simulado import simular_mano, simular_partida
from src.captura.replay import mano_reconstruible
from src.dominio.motor import MotorCorazones


def _policias_fijas():
    from src.agentes.heuristicos import bot_agresivo, bot_conservador, bot_evasivo
    return [bot_conservador, bot_agresivo, bot_evasivo, bot_conservador]


def test_simular_mano_produce_52_jugadas_y_puntuacion_valida():
    motor = MotorCorazones()
    mano = simular_mano(motor, _policias_fijas())
    assert len(mano.jugadas) == 52
    assert sum(mano.puntuacion_mano) in (26, 78)
    assert mano_reconstruible(mano)


def test_simular_mano_pase_dado_y_recibido_son_3_cartas_distintas_de_la_mano_inicial():
    motor = MotorCorazones()
    mano = simular_mano(motor, _policias_fijas())
    assert len(mano.mano_inicial_agente) == 13
    if mano.direccion_pase is not None:
        assert len(mano.pase_dado) == 3
        assert len(mano.pase_recibido) == 3
        assert set(mano.pase_dado) <= set(mano.mano_inicial_agente)
        assert set(mano.pase_recibido).isdisjoint(mano.pase_dado)


def test_direccion_pase_rota_con_numero_de_mano_dentro_de_la_partida():
    motor = MotorCorazones()
    direcciones = [simular_mano(motor, _policias_fijas()).direccion_pase for _ in range(4)]
    assert direcciones == ["izquierda", "derecha", "enfrente", None]


def test_simular_partida_marcador_final_es_suma_de_las_manos():
    rng = random.Random(0)
    partida = simular_partida(rng, "test_1", manos_por_partida=4, prob_forzar_lunatico=0.0)
    assert len(partida.manos) == 4
    esperado = [0, 0, 0, 0]
    for m in partida.manos:
        for i in range(4):
            esperado[i] += m.puntuacion_mano[i]
    assert partida.marcador_final == esperado
    assert sorted(partida.ranking_final) == [0, 1, 2, 3]


def test_forzar_lunatico_aumenta_la_tasa_de_pozo():
    """BotLunatico persigue el pozo de verdad -- forzarlo en un asiento debe
    producir notablemente más manos con pozo que sin forzarlo, sobre muchas
    partidas (con semilla fija para que el test sea determinista)."""
    def tasa_pozo(prob_forzar, n=60, seed=1):
        rng = random.Random(seed)
        pozo = manos = 0
        for i in range(n):
            partida = simular_partida(rng, f"p_{i}", manos_por_partida=4, prob_forzar_lunatico=prob_forzar)
            for m in partida.manos:
                manos += 1
                if sum(m.puntuacion_mano) == 78:
                    pozo += 1
        return pozo / manos

    sin_forzar = tasa_pozo(prob_forzar=0.0)
    con_forzar = tasa_pozo(prob_forzar=1.0)
    assert con_forzar > sin_forzar


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
