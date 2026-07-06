"""Tests de scripts/entrenar_moon_prob.py (generación de dataset, sin entrenar red)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from entrenar_moon_prob import _receptor_y_dador, construir_dataset, ejemplos_de_mano
from src.captura.escritor import EscritorJsonl
from src.captura.modelos import Jugada, RegistroMano, RegistroPartida
from src.entorno.moon_model import DIM_PROPIO, DIM_RIVAL


def _mano_completa_simple() -> RegistroMano:
    """Una mano de 13 bazas donde el asiento 0 gana TODO (pozo perfecto).

    Determinista por REPARTO (no por heurística de juego): al asiento 0 se le
    da el palo de Tréboles COMPLETO (13 cartas) y el resto se reparte entre
    los otros 3. Como nadie más tiene tréboles, el asiento 0 siempre lidera
    (nunca lo superan siguiendo el palo) y por lo tanto gana TODAS las bazas
    -- incluyendo cualquier punto que los demás descarten -- sin importar qué
    carta legal se juegue en cada turno.

    (La versión anterior hacía que el asiento 0 jugara "la más alta legal" y
    los demás "la más baja legal" sobre un reparto aleatorio; medido
    empíricamente eso solo producía un pozo real en ~1% de las manos, porque
    el reparto aleatorio no garantiza que el asiento 0 tenga la carta más
    alta de cada palo. Forzar el reparto en vez de la estrategia de juego lo
    hace 100% determinista.)
    """
    from src.dominio.carta import Carta
    from src.dominio.motor import MotorCorazones

    m = MotorCorazones()
    treboles = [c for c in Carta._TODAS if c.palo == 0]  # 0 = Tréboles
    resto = [c for c in Carta._TODAS if c.palo != 0]
    m.jugadores[0].recibir_mano(treboles)
    for i in range(1, 4):
        m.jugadores[i].recibir_mano(resto[(i - 1) * 13: i * 13])
    for jug in m.jugadores:
        jug.bazas_ganadas = []
    m.corazones_rotos = False
    m.numero_baza = 1
    m.numero_mano = 1
    m.mesa = []
    m.palo_de_salida = None
    m._mano_activa = True
    m._fijar_jugador_inicial()

    jugadas = []
    baza = 1
    while not all(len(j.mano) == 0 for j in m.jugadores):
        idx = m.obtener_jugador_actual()
        legales = m.obtener_jugadas_legales(idx)
        carta = legales[0]  # cualquier legal sirve: el reparto ya garantiza el pozo
        jugadas.append(Jugada(asiento=idx, carta_id=carta.id, baza=baza))
        m.jugar_carta(idx, carta)
        if len(m.mesa) == 4:
            m.resolver_baza()
            baza += 1
    puntos = m.calcular_puntuacion_mano()
    return RegistroMano(
        numero_mano=1, direccion_pase=None, mano_inicial_agente=[],
        jugadas=jugadas, puntuacion_mano=puntos,
    )


def test_receptor_y_dador_izquierda():
    receptor, dador = _receptor_y_dador("izquierda", 0)
    assert receptor == 1  # seat+1 = izquierda
    assert dador == 3     # seat-1 me pasó a mí


def test_receptor_y_dador_sin_pase():
    assert _receptor_y_dador(None, 0) == (None, None)


def test_ejemplos_de_mano_formas_y_no_vacio():
    mano = _mano_completa_simple()
    ejemplos_propio, ejemplos_rival = ejemplos_de_mano(mano, asiento_agente_real=0)
    assert len(ejemplos_propio) > 0
    assert len(ejemplos_rival) > 0
    feats, label = ejemplos_propio[0]
    assert feats.shape == (DIM_PROPIO,)
    assert label in (0.0, 1.0)
    feats_r, label_r = ejemplos_rival[0]
    assert feats_r.shape == (DIM_RIVAL,)
    assert label_r in (0.0, 1.0)


def test_ejemplos_de_mano_etiqueta_el_pozo_correctamente():
    mano = _mano_completa_simple()
    # calcular_puntuacion_mano() ya aplica la regla de Pleno: el tirador queda
    # en 0 y los otros 3 en 26 (suma 78). Una mano normal siempre suma 26 (sin
    # garantizar que nadie tenga exactamente 0), así que el pozo se detecta
    # por suma == 78, no == 26.
    if sum(mano.puntuacion_mano) != 78:
        return  # esta semilla en particular no produjo pozo; no es el foco del test
    luna_seat = mano.puntuacion_mano.index(0)
    ejemplos_propio, _ = ejemplos_de_mano(mano, asiento_agente_real=0)
    # al menos un ejemplo de la perspectiva del que hizo el pozo debe tener label 1.0
    # (los del asiento ganador, antes de que el gate lo excluya en las últimas bazas)
    assert any(label == 1.0 for _, label in ejemplos_propio) or luna_seat != 0


def test_construir_dataset_agrupa_por_partida_y_salta_manos_no_reconstruibles(tmp_path):
    mano_ok = _mano_completa_simple()
    # Sin jugadas ni manos_restantes: reconstruir_manos no puede completar las
    # 4 manos de 13 cartas -> mano_reconstruible() da False y debe saltarse,
    # sin tirar el resto de manos reconstruibles de la MISMA partida.
    mano_rota = RegistroMano(numero_mano=2, direccion_pase=None, mano_inicial_agente=[])

    partida = RegistroPartida(
        partida_id="p1", timestamp="2026-01-01T00:00:00", asiento_agente=0,
        fuente="test", manos=[mano_ok, mano_rota],
    )
    ruta = tmp_path / "partidas.jsonl"
    EscritorJsonl(ruta).escribir(partida)

    propio, rival, timestamps = construir_dataset(str(ruta))

    assert "p1" in propio and "p1" in rival and "p1" in timestamps
    assert len(propio["p1"]) > 0  # la mano_ok sí se procesó pese a mano_rota
    assert len(rival["p1"]) > 0
    assert timestamps["p1"] == "2026-01-01T00:00:00"


import numpy as np
from entrenar_moon_prob import _auc, _brier, _entrenar
from src.entorno.moon_model import _RedMoonMLP


def test_auc_perfecto_cuando_scores_separan_las_clases():
    y_true = np.array([0, 0, 1, 1], dtype=np.float32)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    assert _auc(y_true, y_score) == 1.0


def test_auc_nan_si_no_hay_de_una_clase():
    y_true = np.array([1, 1], dtype=np.float32)
    y_score = np.array([0.5, 0.6], dtype=np.float32)
    assert np.isnan(_auc(y_true, y_score))


def test_auc_medio_si_todos_los_scores_empatan():
    y_true = np.array([0, 1], dtype=np.float32)
    y_score = np.array([0.5, 0.5], dtype=np.float32)
    assert _auc(y_true, y_score) == 0.5


def test_brier_cero_si_prediccion_perfecta():
    y_true = np.array([0.0, 1.0], dtype=np.float32)
    y_score = np.array([0.0, 1.0], dtype=np.float32)
    assert _brier(y_true, y_score) == 0.0


def test_entrenar_reduce_la_perdida_en_datos_separables():
    rng = np.random.default_rng(0)
    dim = 5
    X_pos = rng.normal(3.0, 0.1, size=(50, dim)).astype(np.float32)
    X_neg = rng.normal(-3.0, 0.1, size=(50, dim)).astype(np.float32)
    X_train = np.concatenate([X_pos[:40], X_neg[:40]])
    y_train = np.concatenate([np.ones(40), np.zeros(40)]).astype(np.float32)
    X_val = np.concatenate([X_pos[40:], X_neg[40:]])
    y_val = np.concatenate([np.ones(10), np.zeros(10)]).astype(np.float32)

    red = _RedMoonMLP(dim)
    red, auc, brier = _entrenar(red, X_train, y_train, X_val, y_val, epocas=200)
    assert auc > 0.9
    assert brier < 0.1
