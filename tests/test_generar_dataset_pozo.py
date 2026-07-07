"""Tests de scripts/generar_dataset_pozo.py (solo la extracción de etiquetas)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generar_dataset_pozo import cargar_etiquetas
from src.captura.modelos import RegistroMano, RegistroPartida


def _partida_con_manos(puntuaciones):
    manos = [
        RegistroMano(numero_mano=i + 1, direccion_pase=None, mano_inicial_agente=[],
                     puntuacion_mano=pm)
        for i, pm in enumerate(puntuaciones)
    ]
    return RegistroPartida(partida_id="p1", timestamp="t", asiento_agente=0,
                            fuente="test", manos=manos)


def test_filtra_solo_manos_con_pozo(tmp_path):
    # mano 1: pozo (suma 78); mano 2: normal (suma 26)
    partida = _partida_con_manos([[0, 26, 26, 26], [10, 5, 6, 5]])
    volcado = tmp_path / "volcado.jsonl"
    with open(volcado, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "partida_id": "p1", "mano": 1, "baza": 3,
            "carta_elegida": "5T", "legales": ["5T", "AP"],
            "scores": {"5T": 1.0, "AP": 9.0}, "regret": 0.0,
        }) + "\n")
        f.write(json.dumps({
            "partida_id": "p1", "mano": 2, "baza": 4,
            "carta_elegida": "QC", "legales": ["QC", "2D"],
            "scores": {"QC": 3.0, "2D": 8.0}, "regret": 0.0,
        }) + "\n")

    etiquetas = cargar_etiquetas(str(volcado), [partida], filtro="pozo")

    assert (("p1", 1, 3)) in etiquetas
    assert (("p1", 2, 4)) not in etiquetas  # mano sin pozo, descartada


def test_filtro_todos_no_descarta_manos_normales(tmp_path):
    partida = _partida_con_manos([[0, 26, 26, 26], [10, 5, 6, 5]])
    volcado = tmp_path / "volcado.jsonl"
    with open(volcado, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "partida_id": "p1", "mano": 2, "baza": 4,
            "carta_elegida": "QC", "legales": ["QC", "2D"],
            "scores": {"QC": 3.0, "2D": 8.0}, "regret": 0.0,
        }) + "\n")

    etiquetas = cargar_etiquetas(str(volcado), [partida], filtro="todos")
    assert ("p1", 2, 4) in etiquetas


def test_etiqueta_es_la_carta_de_menor_score(tmp_path):
    partida = _partida_con_manos([[0, 26, 26, 26]])  # seat 0 shot the moon (regla de Pleno)
    volcado = tmp_path / "volcado.jsonl"
    with open(volcado, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "partida_id": "p1", "mano": 1, "baza": 5,
            "carta_elegida": "3T", "legales": ["3T", "10C", "JC"],
            "scores": {"3T": 26.0, "10C": 2.2, "JC": 2.2}, "regret": 23.8,
        }) + "\n")

    from src.captura.modelos import str_a_carta_id
    etiquetas = cargar_etiquetas(str(volcado), [partida], filtro="pozo")
    assert etiquetas[("p1", 1, 5)] in (str_a_carta_id("10C"), str_a_carta_id("JC"))
