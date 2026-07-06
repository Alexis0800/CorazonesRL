import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from auditar_logs_servidor import _auditar_partida, _imprimir_resumen


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


def test_auditar_partida_devuelve_puesto_y_marcador(tmp_path):
    eventos = [
        {"evento": "reset_mano", "entrada": {"cartas": list(range(13))},
         "salida": {"ok": True}, "estado": {"scores": [0, 0, 0, 0]}},
        {"evento": "registrar_resto", "entrada": {"ganador": 0, "cartas_restantes": []},
         "salida": {"ok": True, "puntos_mano": [26, 0, 0, 0], "scores": [26, 0, 0, 0]},
         "estado": {"scores": [26, 0, 0, 0]}},
    ]
    ruta = _escribir(tmp_path, eventos)
    resultado = _auditar_partida(ruta)
    assert resultado["scores"] == [26, 0, 0, 0]
    assert resultado["puesto"] == 4
    assert resultado["n_errores"] == 0
    assert resultado["n_corruptas"] == 0


def test_resumen_calcula_distribucion_de_puestos(capsys):
    resultados = [
        {"archivo": "a", "n_manos": 5, "n_errores": 0, "n_corruptas": 0, "scores": [0, 5, 10, 15], "puesto": 1},
        {"archivo": "b", "n_manos": 5, "n_errores": 1, "n_corruptas": 0, "scores": [15, 5, 10, 0], "puesto": 4},
        {"archivo": "c", "n_manos": 5, "n_errores": 0, "n_corruptas": 0, "scores": None, "puesto": None},
    ]
    _imprimir_resumen(resultados)
    salida = capsys.readouterr().out
    assert "3 partidas, 2 con marcador final" in salida
    assert "puesto 1:   1/2 ( 50.0%)" in salida
    assert "puesto 4:   1/2 ( 50.0%)" in salida
    assert "puesto promedio: 2.50" in salida
    assert "1 errores y 0 líneas corruptas en total" in salida
