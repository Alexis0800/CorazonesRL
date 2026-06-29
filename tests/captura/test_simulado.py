"""El AdaptadorSimulado debe producir un stream válido y re-jugable."""
from __future__ import annotations

from src.captura.recolector import RecolectorPartidas
from src.captura.replay import ejemplos_de_mano
from src.captura.simulado import AdaptadorSimulado
from src.entorno.dimensiones import DIM_ENTORNO
from src.entorno.observacion import ObservacionBuilder


def _capturar(**kw):
    rec = RecolectorPartidas(adaptador=AdaptadorSimulado(**kw))
    return rec.ejecutar()


def test_genera_una_partida_bien_formada():
    partidas = _capturar(max_manos=2, seed=1)
    assert len(partidas) == 1
    p = partidas[0]
    assert len(p.manos) == 2
    for m in p.manos:
        assert len(m.jugadas) == 52
        assert len(m.mano_inicial_agente) == 13
    # mano 1 tiene pase (izquierda) registrado
    assert len(p.manos[0].pase_dado) == 3
    assert len(p.manos[0].pase_recibido) == 3


def test_marcador_coherente():
    p = _capturar(max_manos=3, seed=2)[0]
    suma_manos = [0, 0, 0, 0]
    for m in p.manos:
        for i in range(4):
            suma_manos[i] += m.puntuacion_mano[i]
    assert p.marcador_final == suma_manos


def test_es_rejugable_por_el_encoder():
    p = _capturar(max_manos=2, seed=3)[0]
    builder = ObservacionBuilder(dim=DIM_ENTORNO)
    for m in p.manos:
        ejemplos = ejemplos_de_mano(m, p.asiento_agente, builder)
        assert len(ejemplos) == 13
        for obs, accion in ejemplos:
            assert obs.shape == (DIM_ENTORNO,)
            assert 0 <= accion < 52
