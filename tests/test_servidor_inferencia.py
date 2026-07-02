"""Test del servidor de inferencia: logging JSONL y /terminar_partida (sin cargar checkpoint)."""
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


def test_terminar_partida_reinicia_marcador_y_mano(tmp_path):
    server, r = _servidor_sin_modelo(tmp_path)
    try:
        puerto = server.server_address[1]
        cartas = parse_cartas("AP KP QP")
        _post(puerto, "/reset_mano", {"cartas": [c.id for c in cartas]})
        r.scores = [30, 10, 0, 5]

        salida = _post(puerto, "/terminar_partida", {})
        assert salida == {"ok": True}
        assert r.scores == [0, 0, 0, 0]
        assert r.mano == []
    finally:
        server.shutdown()
