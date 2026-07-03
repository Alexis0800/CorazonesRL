"""
Convierte los `logs/session-*.jsonl` del puente SFS2X (proyecto Node en
D:\\Github\\Personal\\hearts-sfs-bridge) al formato `RegistroPartida` de este
repo (`src/captura/modelos.py`), para poder reproducirlos con
`src/captura/replay.py` igual que cualquier otra captura.

Traducción de ids: el bridge registra SIEMPRE ids "de cable" (convención
SFS2X: 0♠ 1♥ 2♣ 3♦); el dominio de este repo usa `carta.id` (0♣ 1♦ 2♠ 3♥).
La permutación ♣↔♠ ♦↔♥ es su propia inversa (ver
`hearts-sfs-bridge/src/inference-client.js`, SUIT_SWAP) y se aplica aquí a
TODO id que llega del bridge.

Una mano es reproducible por `replay.py` solo si se vieron sus 52 jugadas
(13 bazas × 4 asientos). Cuando la app resuelve el resto automáticamente
("remate") las cartas del barrido nunca cruzan el cable individualmente
(ver FINDINGS.md del bridge) y el bridge tampoco registra qué asiento tenía
cada una — por diseño, ni falta ni sobra dato: la mano se guarda igual (con
sus bazas reales previas al remate) pero queda con `manos_restantes` vacío,
así que `mano_reconstruible()` la excluye del replay hoja por hoja. Su
puntuación de mano sigue siendo válida (la calcula MotorCorazones del lado
del servidor de inferencia) y queda guardada en `puntuacion_mano`.

Uso:
    python scripts/importar_sesiones_bridge.py --dir D:/Github/Personal/hearts-sfs-bridge/logs --out data/partidas_bridge.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

from src.captura.escritor import EscritorJsonl
from src.captura.modelos import Jugada, RegistroMano, RegistroPartida
from src.captura.replay import mano_reconstruible

_SUIT_SWAP = [2, 3, 0, 1]  # cable <-> modelo (♠<->♣, ♥<->♦), su propia inversa


def swap_suit(carta_id_cable: int) -> int:
    """`carta.id` de cable (0♠1♥2♣3♦) -> `carta.id` del modelo (0♣1♦2♠3♥)."""
    palo, resto = divmod(carta_id_cable, 13)
    return _SUIT_SWAP[palo] * 13 + resto


class _Acumulador:
    """Estado de la mano en curso mientras se recorre un session-*.jsonl."""

    def __init__(self, numero_mano: int, seat: int, direccion_pase: Optional[str], cartas_cable: List[int]):
        self.numero_mano = numero_mano
        self.seat = seat
        self.direccion_pase = direccion_pase
        self.mano_inicial_agente = [swap_suit(c) for c in cartas_cable]
        self.pase_dado: List[int] = []
        self.pase_recibido: List[int] = []
        self.jugadas: List[Jugada] = []
        self._bazas_vistas: Dict[tuple, int] = {}  # (asiento, baza) -> índice en jugadas, para deduplicar ecos

    def agregar_jugada(self, asiento: int, carta_id_cable: int, trick: int, avisos: List[str], archivo: str) -> None:
        baza = trick + 1
        clave = (asiento, baza)
        if clave in self._bazas_vistas:
            avisos.append(
                f"{archivo}: mano {self.numero_mano}, asiento {asiento} baza {baza}: "
                f"cardPlayed duplicado (eco tardío), se ignora la repetición")
            return
        self._bazas_vistas[clave] = len(self.jugadas)
        self.jugadas.append(Jugada(asiento=asiento, carta_id=swap_suit(carta_id_cable), baza=baza))

    def cerrar(self, puntuacion_mano: List[int]) -> RegistroMano:
        return RegistroMano(
            numero_mano=self.numero_mano,
            direccion_pase=self.direccion_pase,
            mano_inicial_agente=self.mano_inicial_agente,
            pase_dado=self.pase_dado,
            pase_recibido=self.pase_recibido,
            jugadas=self.jugadas,
            puntuacion_mano=puntuacion_mano,
            # El bridge no registra el reparto por asiento de un remate (ver docstring
            # del módulo) -- se deja vacío a propósito; mano_reconstruible() lo detecta.
            remate_asiento=None,
            manos_restantes=[],
        )


def partida_desde_lineas(lineas: List[dict], partida_id: str, avisos: Optional[List[str]] = None) -> Optional[RegistroPartida]:
    """Recorre los eventos (ya parseados) de UN `session-*.jsonl` y arma su `RegistroPartida`.

    Devuelve None si el archivo no contiene ninguna mano repartida (p.ej. una
    sesión que se cayó antes de conectar). `avisos` (si se pasa) acumula texto
    humano sobre manos incompletas/descartadas, para reportar sin truncar en silencio.
    """
    avisos = avisos if avisos is not None else []
    manos: List[RegistroMano] = []
    actual: Optional[_Acumulador] = None
    asiento_agente: Optional[int] = None
    timestamp = lineas[0]["t"] if lineas else ""
    marcador_final: List[int] = []

    def cerrar_actual(puntuacion_mano: List[int]) -> None:
        nonlocal actual
        if actual is None:
            return
        manos.append(actual.cerrar(puntuacion_mano))
        actual = None

    for linea in lineas:
        tipo = linea.get("type")
        if tipo in ("handDealt", "holdHandDealt"):
            if actual is not None:
                # La mano anterior nunca cerró con handFinal (sesión cortada a media
                # mano) -- se guarda igual, sin puntuación, para no perder sus jugadas.
                avisos.append(
                    f"{partida_id}: mano {actual.numero_mano} nunca recibió handFinal "
                    f"(sesión cortada); se guarda sin puntuación")
                cerrar_actual([])
            if asiento_agente is None:
                asiento_agente = linea["seat"]
            # "sin pase" (holdHandDealt) => None, la misma convención que motor.direccion_pase();
            # una mano CON pase se corrige abajo al ver su evento 'pass'.
            actual = _Acumulador(
                numero_mano=len(manos) + 1, seat=linea["seat"],
                direccion_pase=None, cartas_cable=linea["cartas"])
        elif tipo == "pass" and actual is not None:
            actual.direccion_pase = linea["dir"]
            actual.pase_dado = [swap_suit(c) for c in linea["ids"]]
        elif tipo == "passReceived" and actual is not None:
            actual.pase_recibido = [swap_suit(c) for c in linea["cards"]]
        elif tipo == "cardPlayed" and actual is not None:
            actual.agregar_jugada(linea["seat"], linea["cardId"], linea["trick"], avisos, partida_id)
        elif tipo == "handFinal":
            cerrar_actual(list(linea.get("puntos", [])))
            if linea.get("scores"):
                marcador_final = list(linea["scores"])  # marcador corriente, por si no hay matchEnded
        elif tipo == "matchEnded":
            marcador_final = list(linea.get("marcadorFinal", []))
            cerrar_actual([])  # por si acaso (normalmente ya cerrada por su propio handFinal)
            break

    cerrar_actual([])  # sesión sin matchEnded (p.ej. proceso detenido a mano): cierra lo que haya

    if not manos:
        return None

    orden_peor_a_mejor = sorted(range(4), key=lambda s: -marcador_final[s]) if len(marcador_final) == 4 else []

    return RegistroPartida(
        partida_id=partida_id,
        timestamp=timestamp,
        asiento_agente=asiento_agente if asiento_agente is not None else 0,
        fuente="sfs:hearts-sfs-bridge",
        manos=manos,
        marcador_final=marcador_final,
        ranking_final=orden_peor_a_mejor,
    )


def leer_sesion(path: Path, avisos: List[str]) -> Optional[RegistroPartida]:
    lineas = []
    for n, l in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not l.strip():
            continue
        try:
            lineas.append(json.loads(l))
        except json.JSONDecodeError:
            avisos.append(f"{path.name}: línea {n} corrupta, se ignora")
    return partida_desde_lineas(lineas, partida_id=path.stem, avisos=avisos)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="Carpeta con session-*.jsonl (busca recursivo)")
    p.add_argument("--out", required=True, help="Ruta del .jsonl de salida (RegistroPartida, append)")
    args = p.parse_args()

    rutas = sorted(Path(args.dir).glob("**/session-*.jsonl"))
    if not rutas:
        p.error(f"no se encontraron session-*.jsonl bajo {args.dir}")

    escritor = EscritorJsonl(args.out)
    avisos: List[str] = []
    n_partidas = 0
    n_manos = 0
    n_manos_reconstruibles = 0

    for ruta in rutas:
        partida = leer_sesion(ruta, avisos)
        if partida is None:
            avisos.append(f"{ruta.name}: sin manos repartidas, se omite")
            continue
        n_partidas += 1
        n_manos += len(partida.manos)
        n_manos_reconstruibles += sum(1 for m in partida.manos if mano_reconstruible(m))
        escritor.escribir(partida)

    print(f"Partidas importadas: {n_partidas}  (de {len(rutas)} archivos)")
    print(f"Manos totales: {n_manos}  |  reproducibles jugada-a-jugada: {n_manos_reconstruibles} "
          f"({100 * n_manos_reconstruibles / n_manos:.0f}%)" if n_manos else "Manos totales: 0")
    if avisos:
        print(f"\nAvisos ({len(avisos)}):")
        for a in avisos:
            print(f"  - {a}")


if __name__ == "__main__":
    main()
