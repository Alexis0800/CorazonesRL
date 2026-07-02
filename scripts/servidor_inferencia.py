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
    /salud            (cualquier método)                           -> {"ok": true, "obs_dim": N}
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.dominio.carta import Carta
from scripts.recomendador import Recomendador

_LOOKUP_POR_ID = {c.id: c for c in Carta._TODAS}


def _carta(id_: int) -> Carta:
    return _LOOKUP_POR_ID[id_]


class Handler(BaseHTTPRequestHandler):
    recomendador: Recomendador  # inyectado por main()

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

    def do_POST(self):
        try:
            datos = self._leer_json()
            r = self.recomendador
            if self.path == "/reset_mano":
                r.reset_mano([_carta(i) for i in datos["cartas"]])
                self._responder(200, {"ok": True})
            elif self.path == "/recomendar_pase":
                cartas = r.recomendar_pase(datos["direccion"])
                self._responder(200, {"cartas": [c.id for c in cartas]})
            elif self.path == "/recomendar_jugada":
                mesa_antes = [(idx, _carta(cid)) for idx, cid in datos.get("mesa_antes", [])]
                carta = r.recomendar_jugada(mesa_antes)
                self._responder(200, {"carta": carta.id})
            elif self.path == "/registrar_baza":
                jugadas = [(idx, _carta(cid)) for idx, cid in datos["jugadas"]]
                r.registrar_baza(jugadas, datos["ganador"])
                self._responder(200, {"ok": True})
            else:
                self._responder(404, {"error": f"ruta desconocida: {self.path}"})
        except Exception as e:
            self._responder(400, {"error": str(e)})

    def do_GET(self):
        if self.path == "/salud":
            self._responder(200, {"ok": True, "obs_dim": self.recomendador.obs_dim})
        else:
            self._responder(404, {"error": "usa POST para los endpoints de juego"})


def main() -> None:
    p = argparse.ArgumentParser(description="Servidor de inferencia para el puente SFS2X")
    p.add_argument("--modelo", required=True, help="Ruta al checkpoint (igual que recomendador.py --modelo)")
    p.add_argument("--asiento", type=int, default=0, help="Asiento (0-3) del agente en la mesa")
    p.add_argument("--puerto", type=int, default=8765)
    args = p.parse_args()

    print(f"Cargando modelo desde {args.modelo} (asiento {args.asiento})...")
    Handler.recomendador = Recomendador(args.modelo, mi_idx=args.asiento)
    print(f"Modelo cargado (obs_dim={Handler.recomendador.obs_dim}). "
          f"Escuchando en http://127.0.0.1:{args.puerto}")

    server = ThreadingHTTPServer(("127.0.0.1", args.puerto), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo servidor de inferencia.")


if __name__ == "__main__":
    main()
