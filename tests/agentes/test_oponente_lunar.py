"""
Tests de OponenteLunar — rival de pool que compone ModoLunar sobre BotExperto.

Se inyecta un ModoLunar con estimador stub (P fija) directamente en `_modo`
(el slot lazy), así no se cargan pesos torch: aquí se prueba el CABLEADO
(delegación, reset, serialización), no el modelo.

El uso de pickle es el requisito real del pool: Ray picklea los oponentes
hacia los workers (contenido propio, no untrusted).
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

from src.agentes.bot_experto import BotExperto
from src.agentes.modo_lunar import ModoLunar
from src.agentes.oponente_lunar import OponenteLunar
from tests.agentes.test_modo_lunar import _EstimadorStub, _motor_13


def _lunero(p: float) -> OponenteLunar:
    opp = OponenteLunar()
    opp._modo = ModoLunar(_EstimadorStub(p), umbral_juego=0.30)  # stub, sin torch
    return opp


def test_p_alta_persigue():
    """Con P alta el gate se compromete y juega la lógica moon (lidera alto)."""
    opp = _lunero(0.90)
    m = _motor_13()
    legales = list(m.jugadores[0].mano)
    carta = opp(m, 0, legales)
    assert opp._modo.comprometida
    assert carta == max(legales, key=lambda c: c.valor)  # _jugar_moon lidera alto


def test_p_baja_delega_a_base():
    """Con P baja ModoLunar devuelve None y decide la base (BotExperto)."""
    opp = _lunero(0.01)
    m, m2 = _motor_13(), _motor_13()
    carta = opp(m, 0, list(m.jugadores[0].mano))
    assert not opp._modo.comprometida
    esperado = BotExperto()(m2, 0, list(m2.jugadores[0].mano))
    assert carta == esperado


def test_pase_delega_segun_p():
    m = _motor_13()
    # P alta: pase constructivo de luna (BotLunatico: suelta las 3 más bajas).
    assert _lunero(0.90).pasar(m, 0) is not None
    # P baja: pase del experto (mismo resultado que un BotExperto fresco).
    assert _lunero(0.01).pasar(m, 0) == BotExperto().pasar(_motor_13(), 0)


def test_reset_por_mano():
    opp = _lunero(0.90)
    m = _motor_13()
    opp(m, 0, list(m.jugadores[0].mano))
    assert opp._modo.comprometida
    opp.reset()
    assert not opp._modo.comprometida


def test_pickle_descarta_modo():
    """El estado serializado nunca lleva ModoLunar (contiene redes torch)."""
    opp = _lunero(0.90)  # _modo construido (stub)
    opp2 = pickle.loads(pickle.dumps(opp))
    assert opp2._modo is None  # se reconstruye lazy en el worker
    # Fresco también funciona y sigue lazy.
    assert pickle.loads(pickle.dumps(OponenteLunar()))._modo is None


def test_pickle_sin_importar_torch():
    """Round-trip de pickle en un proceso limpio SIN que torch se importe
    (garantía de serialización barata hacia los workers de Ray)."""
    raiz = Path(__file__).resolve().parents[2]
    codigo = (
        "import pickle, sys\n"
        "from src.agentes.oponente_lunar import OponenteLunar\n"
        "pickle.loads(pickle.dumps(OponenteLunar()))\n"
        "assert 'torch' not in sys.modules, 'torch se importo en el pickle'\n"
    )
    r = subprocess.run([sys.executable, "-c", codigo], cwd=str(raiz),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
