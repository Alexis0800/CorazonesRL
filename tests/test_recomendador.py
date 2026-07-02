"""Test del reparto estimado en Recomendador._motor() (sin cargar checkpoint)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recomendador import Recomendador, parse_cartas
from src.dominio.carta import Carta


def _recomendador_sin_modelo(mi_idx: int = 0) -> Recomendador:
    r = Recomendador.__new__(Recomendador)
    r.me = mi_idx
    r.scores = [0, 0, 0, 0]
    r.reset_mano([])
    return r


def test_motor_no_asigna_carta_a_rival_void_en_ese_palo():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")  # 13 picas
    # Rival 1 (izquierda) es void en corazones.
    r.vacios[1].add(3)

    m = r._motor(mesa=[])

    for c in m.jugadores[1].mano:
        assert c.palo != 3, f"rival void en corazones recibió {c.id}"


def test_motor_reparte_todas_las_desconocidas():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    m = r._motor(mesa=[])
    repartidas = sum(len(m.jugadores[i].mano) for i in range(4) if i != r.me)
    assert repartidas == len(Carta._TODAS) - len(r.mano)
