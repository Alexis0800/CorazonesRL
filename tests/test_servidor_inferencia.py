"""Test del servidor de inferencia: logging JSONL, cálculo de puntos y
/terminar_partida (sin cargar checkpoint)."""
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recomendador import Recomendador, parse_cartas
import servidor_inferencia as si


def _servidor_sin_modelo(tmp_path: Path):
    r = Recomendador.__new__(Recomendador)
    r.me = 0
    r.scores = [0, 0, 0, 0]
    r.ultima_mano_puntos = None
    r.con_pase = False
    r.reset_mano([])

    si.Handler.recomendador = r
    si.Handler.log_path = tmp_path / "log.jsonl"
    server = ThreadingHTTPServer(("127.0.0.1", 0), si.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, r


def _post(puerto: int, ruta: str, payload: dict) -> dict:
    datos = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{puerto}{ruta}", data=datos,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(puerto: int, ruta: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{puerto}{ruta}") as resp:
        return json.loads(resp.read())


def test_reset_mano_registra_evento_con_estado(tmp_path):
    server, _ = _servidor_sin_modelo(tmp_path)
    try:
        puerto = server.server_address[1]
        cartas = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
        salida = _post(puerto, "/reset_mano", {"cartas": [c.id for c in cartas]})
        assert salida == {"ok": True}

        lineas = si.Handler.log_path.read_text(encoding="utf-8").splitlines()
        assert len(lineas) == 1
        evento = json.loads(lineas[0])
        assert evento["evento"] == "reset_mano"
        assert evento["estado"]["mano"] == sorted(c.id for c in cartas)
    finally:
        server.shutdown()


def test_registrar_resto_calcula_puntos_y_acumula_marcador(tmp_path):
    """El servidor calcula los puntos de la mano desde las cartas, el bridge
    no manda ningún número de puntaje: cierra el flujo que antes dependía de
    que el bridge acertara la cuenta (ahí vivía el bug de "suma 25")."""
    server, r = _servidor_sin_modelo(tmp_path)
    try:
        puerto = server.server_address[1]
        cartas_restantes = parse_cartas("AC 2C QP")  # A♥ + 2♥ + Q♠ = 1+1+13
        salida = _post(puerto, "/registrar_resto",
                        {"ganador": 2, "cartas_restantes": [c.id for c in cartas_restantes]})
        assert salida == {"ok": True, "puntos_mano": [0, 0, 15, 0], "scores": [0, 0, 15, 0]}
        assert r.scores == [0, 0, 15, 0]

        # segunda mano: se acumula sobre el marcador previo
        r.reset_mano([])
        cartas_restantes2 = parse_cartas("3C")  # 1 punto para el jugador 1
        salida2 = _post(puerto, "/registrar_resto",
                         {"ganador": 1, "cartas_restantes": [c.id for c in cartas_restantes2]})
        assert salida2 == {"ok": True, "puntos_mano": [0, 1, 0, 0], "scores": [0, 1, 15, 0]}
    finally:
        server.shutdown()


def test_puntos_endpoint_devuelve_marcador_actual(tmp_path):
    server, r = _servidor_sin_modelo(tmp_path)
    try:
        puerto = server.server_address[1]
        assert _get(puerto, "/puntos") == {"scores": [0, 0, 0, 0], "ultima_mano_puntos": None}

        cartas_restantes = parse_cartas("QP")  # 13 puntos al jugador 3
        _post(puerto, "/registrar_resto",
              {"ganador": 3, "cartas_restantes": [c.id for c in cartas_restantes]})
        assert _get(puerto, "/puntos") == {
            "scores": [0, 0, 0, 13], "ultima_mano_puntos": [0, 0, 0, 13]}
    finally:
        server.shutdown()


def test_terminar_partida_reinicia_marcador_y_devuelve_scores_finales(tmp_path):
    server, r = _servidor_sin_modelo(tmp_path)
    try:
        puerto = server.server_address[1]
        cartas = parse_cartas("AP KP QP")
        _post(puerto, "/reset_mano", {"cartas": [c.id for c in cartas]})
        r.scores = [30, 10, 0, 5]

        salida = _post(puerto, "/terminar_partida", {})
        assert salida == {"ok": True, "scores_finales": [30, 10, 0, 5]}
        assert r.scores == [0, 0, 0, 0]
        assert r.mano == []
    finally:
        server.shutdown()
