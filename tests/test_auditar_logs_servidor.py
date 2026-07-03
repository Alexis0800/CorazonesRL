import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from auditar_logs_servidor import _auditar_partida


def _escribir(tmp_path: Path, eventos: list[dict]) -> Path:
    p = tmp_path / "servidor_inferencia_test.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in eventos), encoding="utf-8")
    return p


def test_detecta_error_y_suma_de_puntos_invalida(tmp_path, capsys):
    eventos = [
        {"evento": "reset_mano", "entrada": {"cartas": list(range(13))},
         "salida": {"ok": True}, "estado": {"scores": [0, 0, 0, 0]}},
        {"evento": "registrar_baza", "entrada": {"jugadas": [], "ganador": 0},
         "salida": {"ok": True, "puntos_mano": [5, 5, 5, 5], "scores": [5, 5, 5, 5]},
         "estado": {"scores": [5, 5, 5, 5]}},
        {"evento": "error:/recomendar_jugada", "entrada": {}, "salida": {"error": "boom"},
         "estado": {"scores": [5, 5, 5, 5]}},
    ]
    ruta = _escribir(tmp_path, eventos)
    _auditar_partida(ruta)
    salida = capsys.readouterr().out
    assert "ERROR en error:/recomendar_jugada" in salida
    assert "puntos de mano suman 20" in salida


def test_linea_corrupta_no_interrumpe_la_auditoria(tmp_path, capsys):
    p = tmp_path / "servidor_inferencia_test.jsonl"
    lineas = [
        json.dumps({"evento": "reset_mano", "entrada": {"cartas": list(range(13))},
                    "salida": {"ok": True}, "estado": {"scores": [0, 0, 0, 0]}}),
        '{"a": 1}{"b": 2}',  # línea corrupta simulada (entrelazado de 2 hilos)
        json.dumps({"evento": "registrar_resto", "entrada": {"ganador": 0, "cartas_restantes": []},
                    "salida": {"ok": True, "puntos_mano": [26, 0, 0, 0], "scores": [26, 0, 0, 0]},
                    "estado": {"scores": [26, 0, 0, 0]}}),
    ]
    p.write_text("\n".join(lineas), encoding="utf-8")

    _auditar_partida(p)
    salida = capsys.readouterr().out
    assert "línea 2 corrupta" in salida
    assert "2 eventos, 1 corruptas" in salida


def test_partida_limpia_no_reporta_advertencias(tmp_path, capsys):
    eventos = [
        {"evento": "reset_mano", "entrada": {"cartas": list(range(13))},
         "salida": {"ok": True}, "estado": {"scores": [0, 0, 0, 0]}},
        {"evento": "registrar_resto", "entrada": {"ganador": 0, "cartas_restantes": []},
         "salida": {"ok": True, "puntos_mano": [26, 0, 0, 0], "scores": [26, 0, 0, 0]},
         "estado": {"scores": [26, 0, 0, 0]}},
    ]
    ruta = _escribir(tmp_path, eventos)
    _auditar_partida(ruta)
    salida = capsys.readouterr().out
    assert "⚠" not in salida
    assert "puesto agente (asiento 0): 4" in salida
