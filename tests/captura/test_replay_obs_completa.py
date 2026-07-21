"""Regresión: el encoder de replay debe emitir la obs COMPLETA (alineada con el
env), no la versión mínima con las features estratégicas en cero.

El bug histórico (ver CLAUDE.md, TODO de replay.py): `ejemplos_de_mano` usaba
`construir_desde_motor` (obs mínima), así que el BC de datos humanos no estaba
alineado con un modelo entrenado en el env. Este test falla si se regresa a eso.
"""
import os
import json

import numpy as np
import pytest

from src.captura import replay
from src.captura.modelos import Jugada, RegistroMano, RegistroPartida
from src.entorno.observacion import ObservacionBuilder

_DATA = "data/partidas_bridge_full.jsonl"


def _cargar(path, n):
    out = []
    for linea in open(path, encoding="utf-8"):
        d = json.loads(linea)
        manos = []
        for m in d["manos"]:
            jug = [Jugada(**j) for j in m["jugadas"]]
            resto = {k: v for k, v in m.items() if k != "jugadas"}
            manos.append(RegistroMano(jugadas=jug, **resto))
        resto = {k: v for k, v in d.items() if k != "manos"}
        out.append(RegistroPartida(manos=manos, **resto))
        if len(out) >= n:
            break
    return out


@pytest.mark.skipif(not os.path.exists(_DATA), reason="requiere data/partidas_bridge_full.jsonl")
def test_obs_completa_poblada_y_marcador_hilado():
    partidas = _cargar(_DATA, 15)
    dim = 228
    builder = ObservacionBuilder(dim=dim)

    ejemplos = []
    for p in partidas:
        ejemplos += replay.ejemplos_de_partida(p, builder, dim, "agente")
    assert ejemplos, "no se generaron ejemplos"
    X = np.array([o for o, _ in ejemplos])

    assert X.shape[1] == dim
    # Región estratégica [187:224]: con la obs mínima esto era TODO cero.
    assert (np.abs(X[:, 187:224]).sum(axis=1) > 0).all(), "features estratégicas en cero (obs mínima)"
    # Bloque de pase [224:228] debe ser 0 durante el juego (no en fase de pase).
    assert X[:, 224:228].max() == 0.0
    # Mano [0:52] one-hot: cada decisión tiene entre 1 y 13 cartas.
    sumas = X[:, 0:52].sum(axis=1)
    assert sumas.min() >= 1 and sumas.max() <= 13
    # Marcador histórico [172:176] hila entre manos → alguna mano tardía tiene score>0.
    assert X[:, 172:176].max() > 0.0


@pytest.mark.skipif(not os.path.exists(_DATA), reason="requiere data/partidas_bridge_full.jsonl")
def test_perspectiva_rivales_emite_los_tres_asientos():
    partidas = _cargar(_DATA, 10)
    dim = 228
    builder = ObservacionBuilder(dim=dim)
    ag = ri = 0
    for p in partidas:
        ag += len(replay.ejemplos_de_partida(p, builder, dim, "agente"))
        ri += len(replay.ejemplos_de_partida(p, builder, dim, "rivales"))
    # 3 asientos rivales vs 1 del agente → ~3x (no exacto: mismas manos, 3 perspectivas).
    assert ri > 2.5 * ag
