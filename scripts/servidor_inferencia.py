"""
Servidor de inferencia local para el puente SFS2X (proyecto Node en
D:\\Github\\Personal\\hearts-sfs-bridge). Envuelve `Recomendador`
(scripts/recomendador.py) — reutiliza toda su reconstrucción de estado y
carga de modelo — detrás de una API HTTP/JSON minimalista para que un
proceso Node.js pueda pedir jugadas sin reimplementar nada del lado Python.

Habla en `carta.id` (0-51) en toda la API, igual que el resto del proyecto
(ver `src/captura/modelos.py`).

Uso:
    python scripts/servidor_inferencia.py --modelo models/produccion --asiento 0 --puerto 8765

Endpoints (todos POST, cuerpo y respuesta JSON):
    /reset_mano       {"cartas": [id, ...]}                        -> {"ok": true}
    /recomendar_pase  {"direccion": "izquierda|derecha|frente|sin"} -> {"cartas": [id, id, id]}
    /recomendar_jugada {"mesa_antes": [[asiento, cartaId], ...]}    -> {"carta": id}
    /registrar_baza   {"jugadas": [[asiento, cartaId], ...], "ganador": asiento} -> {"ok": true}
    /registrar_puntos_mano {"puntos": [p0, p1, p2, p3]}             -> {"ok": true, "scores": [...]}
    /terminar_partida (sin cuerpo)                                  -> {"ok": true}
    /salud            (cualquier método)                           -> {"ok": true, "obs_dim": N}

Cada llamada se registra en `--log` (JSONL) con la entrada recibida del bridge
y el estado que el modelo cree tener (mano/cementerio/marcador), para poder
comparar ambos lados si se desincronizan. `/registrar_puntos_mano` acumula el
puntaje de la mano que acaba de terminar (incluida la de alguien "llevándose
el resto") al marcador persistente — sin esto el modelo nunca se entera del
marcador real y juega cada mano como si la partida siguiera 0-0-0-0. Llamar
justo cuando la mano termina (13 bazas o remate del resto), antes o después
de mandar las cartas de la mano siguiente a `/reset_mano` — el orden entre
ambos no importa, son estados independientes.
`/terminar_partida` reinicia el marcador y el estado de mano sin tener que
reiniciar el proceso.
"""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
import json
import argparse

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

from scripts.recomendador import Recomendador
from src.dominio.carta import Carta


_LOOKUP_POR_ID = {c.id: c for c in Carta._TODAS}


def _carta(id_: int) -> Carta:
    return _LOOKUP_POR_ID[id_]


class Handler(BaseHTTPRequestHandler):
    recomendador: Recomendador  # inyectado por main()
    log_path: _Path  # inyectado por main()
    _pendiente_nuevo_log: bool = False  # classvar: rotar log en el próximo endpoint

    def log_message(self, fmt, *args):
        print("[servidor_inferencia]", fmt % args)

    def _leer_json(self) -> dict:
        largo = int(self.headers.get("Content-Length", 0))
        crudo = self.rfile.read(largo) if largo else b"{}"
        return json.loads(crudo or b"{}")

    def _responder(self, status: int, payload: dict):
        cuerpo = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _rotar_log(self):
        """Crea un nuevo archivo de log con timestamp fresco."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Handler.log_path = Handler.log_path.parent / \
            f"servidor_inferencia_{ts}.jsonl"
        Handler._pendiente_nuevo_log = False
        print(
            f"[servidor_inferencia] Nuevo log de partida: {Handler.log_path}")

    def _log_evento(self, evento: str, entrada: dict, salida: dict):
        # Si se pidió rotar después de /terminar_partida, estrenar archivo
        if Handler._pendiente_nuevo_log:
            self._rotar_log()

        linea = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "evento": evento,
            "entrada": entrada,
            "salida": salida,
            "estado": self.recomendador.estado_actual(),
        }
        with open(Handler.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(linea, ensure_ascii=False) + "\n")

    def do_POST(self):
        try:
            datos = self._leer_json()
            r = self.recomendador
            if self.path == "/reset_mano":
                r.reset_mano([_carta(i) for i in datos["cartas"]])
                salida = {"ok": True}
                self._log_evento("reset_mano", datos, salida)
                self._responder(200, salida)
            elif self.path == "/recomendar_pase":
                cartas = r.recomendar_pase(datos["direccion"])
                salida = {"cartas": [c.id for c in cartas]}
                self._log_evento("recomendar_pase", datos, salida)
                self._responder(200, salida)
            elif self.path == "/recomendar_jugada":
                mesa_antes = [(idx, _carta(cid))
                              for idx, cid in datos.get("mesa_antes", [])]
                carta = r.recomendar_jugada(mesa_antes)
                salida = {"carta": carta.id}
                self._log_evento("recomendar_jugada", datos, salida)
                self._responder(200, salida)
            elif self.path == "/registrar_baza":
                jugadas = [(idx, _carta(cid)) for idx, cid in datos["jugadas"]]
                r.registrar_baza(jugadas, datos["ganador"])
                salida = {"ok": True}
                self._log_evento("registrar_baza", datos, salida)
                self._responder(200, salida)
            elif self.path == "/registrar_puntos_mano":
                r.scores = [r.scores[i] + datos["puntos"][i] for i in range(4)]
                salida = {"ok": True, "scores": list(r.scores)}
                self._log_evento("registrar_puntos_mano", datos, salida)
                self._responder(200, salida)
            elif self.path == "/terminar_partida":
                r.scores = [0, 0, 0, 0]
                r.reset_mano([])
                salida = {"ok": True}
                self._log_evento("terminar_partida", datos, salida)
                self._responder(200, salida)
                Handler._pendiente_nuevo_log = True  # próximo endpoint → archivo nuevo
            else:
                self._responder(
                    404, {"error": f"ruta desconocida: {self.path}"})
        except Exception as e:
            # Sin esto, una excepción (p.ej. "Cannot choose from an empty sequence" cuando el
            # bridge pide una jugada con la mano ya vacía) no deja NINGÚN rastro en el JSONL --
            # _log_evento solo se llama del lado del éxito arriba. El bridge sí registra el fallo
            # en su propio log ("modelo no disponible"), pero desde este lado parecía que la
            # petición nunca había llegado. Envuelto en su propio try: si el estado también está
            # roto, preferimos perder la línea de log a perder la respuesta HTTP.
            try:
                self._log_evento(f"error:{self.path}", locals().get(
                    "datos", {}), {"error": str(e)})
            except Exception:
                pass
            self._responder(400, {"error": str(e)})

    def do_GET(self):
        if self.path == "/salud":
            self._responder(
                200, {"ok": True, "obs_dim": self.recomendador.obs_dim})
        else:
            self._responder(
                404, {"error": "usa POST para los endpoints de juego"})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Servidor de inferencia para el puente SFS2X")
    p.add_argument("--modelo", required=True,
                   help="Ruta al checkpoint (igual que recomendador.py --modelo)")
    p.add_argument("--asiento", type=int, default=0,
                   help="Asiento (0-3) del agente en la mesa")
    p.add_argument("--puerto", type=int, default=8765)
    p.add_argument("--log", default="",
                   help="Ruta al JSONL de log. Si se omite, se genera "
                   "logs/servidor_inferencia_YYYYMMDD_HHMMSS.jsonl automáticamente")
    args = p.parse_args()

    if not args.log:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = f"logs/servidor_inferencia_{ts}.jsonl"

    print(f"Cargando modelo desde {args.modelo} (asiento {args.asiento})...")
    Handler.recomendador = Recomendador(args.modelo, mi_idx=args.asiento)
    Handler.log_path = _Path(args.log)
    Handler.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Modelo cargado (obs_dim={Handler.recomendador.obs_dim}). "
          f"Escuchando en http://127.0.0.1:{args.puerto} (log: {Handler.log_path})")

    server = ThreadingHTTPServer(("127.0.0.1", args.puerto), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo servidor de inferencia.")


if __name__ == "__main__":
    main()
