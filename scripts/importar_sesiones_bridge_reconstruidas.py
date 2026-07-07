"""
Convierte `logs/reconstructed/session-*.reconstructed.jsonl` del puente SFS2X
(proyecto Node en D:\\Github\\Personal\\hearts-sfs-bridge, generados por su
script `reconstruct-partidas.js`) al formato `RegistroPartida` de este repo
(`src/captura/modelos.py`), igual que `importar_sesiones_bridge.py` pero para
este formato distinto.

Diferencia clave con `importar_sesiones_bridge.py`: ESE script consume el
`session-*.jsonl` crudo (eventos del wire) y, para una mano que terminó en
concesión ("se llevará el resto"), deja `manos_restantes=[]` a propósito
porque el wire nunca registraba qué asiento tenía cada carta del remate. El
formato `*.reconstructed.jsonl` SÍ lo captura (`remate.seats`), así que esas
manos también son reproducibles jugada-a-jugada aquí (ver
`mano_reconstruible()` en `src/captura/replay.py`).

Traducción de ids: mismo `SUIT_SWAP` que `importar_sesiones_bridge.py` (el
wire usa la convención 0♠1♥2♣3♦; el dominio usa 0♣1♦2♠3♥).

Puntuación de mano: este formato no trae el score final por asiento (el wire
nunca lo expuso a este nivel de reconstrucción), así que se calcula sumando
los puntos de las 4 cartas de cada baza al ganador, más los puntos de las
cartas del remate al asiento que se lleva el resto.

Dirección del pase: `pass.direction` en este formato es el entero crudo del
protocolo SFS2X (`d[1]` del opcode 11 en game-state-tracker.js), que ningún
punto de este pipeline decodifica a izquierda/derecha/enfrente. En vez de
adivinar su significado, se deriva la dirección real: se busca en qué
`seatsFinalHands` de OTRO asiento aparecen las 3 cartas de `pass.cardsOut` —
ese asiento es el receptor real, y su offset (+1/+3/+2) da la dirección
según `MotorCorazones._OFFSET_PASE`.

Uso:
    python scripts/importar_sesiones_bridge_reconstruidas.py \
        --dir D:/Github/Personal/hearts-sfs-bridge/logs/reconstructed \
        --out data/partidas_bridge.jsonl
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
from src.dominio.carta import Carta

_SUIT_SWAP = [2, 3, 0, 1]  # cable <-> modelo (♠<->♣, ♥<->♦), su propia inversa
_OFFSET_A_DIRECCION = {1: "izquierda", 3: "derecha", 2: "enfrente"}


def swap_suit(carta_id_cable: int) -> int:
    """`carta.id` de cable (0♠1♥2♣3♦) -> `carta.id` del modelo (0♣1♦2♠3♥)."""
    palo, resto = divmod(carta_id_cable, 13)
    return _SUIT_SWAP[palo] * 13 + resto


def _direccion_pase(
    seat: int, cardsOut: List[int], seats_final_hands_cable: List[List[int]],
) -> Optional[str]:
    """Dirección real del pase de `seat`, detectada por dónde terminaron sus 3
    cartas dadas (`cardsOut`, ids de cable) -- ver docstring del módulo.

    None si no se encuentra un asiento cuya mano final contenga las 3 cartas
    (dato inconsistente: se descarta la dirección, no la mano completa).
    """
    salida = set(cardsOut)
    for otro in range(4):
        if otro == seat:
            continue
        if salida and salida <= set(seats_final_hands_cable[otro]):
            return _OFFSET_A_DIRECCION.get((otro - seat) % 4)
    return None


def _jugadas_de_mano(tricks: List[dict]) -> List[Jugada]:
    """Bazas en orden de juego real (líder primero), no en orden de asiento."""
    jugadas: List[Jugada] = []
    for t in tricks:
        orden = [(t["leader"] + i) % 4 for i in range(4)]
        for asiento in orden:
            jugadas.append(Jugada(
                asiento=asiento, carta_id=swap_suit(t["bySeat"][asiento]),
                baza=t["trick"] + 1,
            ))
    return jugadas


def _puntuacion_mano(tricks: List[dict], remate: Optional[dict]) -> List[int]:
    """Puntos por asiento, aplicando la regla de Pleno (ver
    `MotorCorazones.calcular_puntuacion_mano`): el ganador de cada baza se
    lleva los puntos crudos de sus 4 cartas (el del remate, los de todas las
    cartas reveladas); si un asiento junta los 26 puntos, el resultado se
    invierte (ese asiento queda en 0, los otros 3 en 26)."""
    puntos = [0, 0, 0, 0]
    for t in tricks:
        puntos[t["winner"]] += sum(
            Carta._TODAS[swap_suit(c)].puntos for c in t["bySeat"]
        )
    if remate is not None:
        puntos[remate["winner"]] += sum(
            Carta._TODAS[swap_suit(c)].puntos
            for mano_restante in remate["seats"] for c in mano_restante
        )
    for i, pts in enumerate(puntos):
        if pts == 26:
            resultado = [26, 26, 26, 26]
            resultado[i] = 0
            return resultado
    return puntos


def mano_desde_reconstruida(h: dict, numero_mano: int) -> Optional[RegistroMano]:
    """`RegistroMano` a partir de UNA línea de `*.reconstructed.jsonl`, o None
    si esa mano no vino marcada como reconstruible por el bridge."""
    if not h.get("reconstructable"):
        return None

    seat = h["seat"]
    direccion = None
    pase_dado: List[int] = []
    pase_recibido: List[int] = []
    if h.get("viaPass") and h.get("pass"):
        cardsOut = h["pass"].get("cardsOut") or []
        cardsIn = h["pass"].get("cardsIn") or []
        direccion = _direccion_pase(seat, cardsOut, h["seatsFinalHands"])
        pase_dado = [swap_suit(c) for c in cardsOut]
        pase_recibido = [swap_suit(c) for c in cardsIn]

    remate = h.get("remate")
    return RegistroMano(
        numero_mano=numero_mano,
        direccion_pase=direccion,
        mano_inicial_agente=[swap_suit(c) for c in h["dealtHand"]],
        pase_dado=pase_dado,
        pase_recibido=pase_recibido,
        jugadas=_jugadas_de_mano(h["tricks"]),
        puntuacion_mano=_puntuacion_mano(h["tricks"], remate),
        remate_asiento=remate["winner"] if remate else None,
        manos_restantes=(
            [[swap_suit(c) for c in cartas] for cartas in remate["seats"]]
            if remate else []
        ),
    )


def partida_desde_archivo(ruta: Path, avisos: List[str]) -> Optional[RegistroPartida]:
    manos: List[RegistroMano] = []
    asiento_agente: Optional[int] = None

    for n, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
        linea = linea.strip()
        if not linea:
            continue
        try:
            h = json.loads(linea)
        except json.JSONDecodeError:
            avisos.append(f"{ruta.name}: línea {n} corrupta, se ignora")
            continue

        if asiento_agente is None:
            asiento_agente = h.get("seat", 0)

        mano = mano_desde_reconstruida(h, numero_mano=len(manos) + 1)
        if mano is None:
            avisos.append(f"{ruta.name}: línea {n} no marcada reconstruible por el bridge, se omite")
            continue
        if h.get("viaPass") and mano.direccion_pase is None:
            avisos.append(
                f"{ruta.name}: mano {mano.numero_mano} con pase pero dirección no "
                f"identificable (cardsOut no aparece en ninguna mano final ajena)")
        if not mano_reconstruible(mano):
            avisos.append(f"{ruta.name}: mano {mano.numero_mano} inconsistente tras reconstruir, se omite")
            continue
        manos.append(mano)

    if not manos:
        return None

    marcador = [0, 0, 0, 0]
    for m in manos:
        for i in range(4):
            marcador[i] += m.puntuacion_mano[i]
    ranking_peor_a_mejor = sorted(range(4), key=lambda s: -marcador[s])

    return RegistroPartida(
        partida_id=ruta.stem,
        timestamp=ruta.stem.replace(".reconstructed", ""),
        asiento_agente=asiento_agente if asiento_agente is not None else 0,
        fuente="sfs:hearts-sfs-bridge-reconstructed",
        manos=manos,
        marcador_final=marcador,
        ranking_final=ranking_peor_a_mejor,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="Carpeta con session-*.reconstructed.jsonl (busca recursivo)")
    p.add_argument("--out", required=True, help="Ruta del .jsonl de salida (RegistroPartida, append)")
    args = p.parse_args()

    rutas = sorted(Path(args.dir).glob("**/*.reconstructed.jsonl"))
    if not rutas:
        p.error(f"no se encontraron *.reconstructed.jsonl bajo {args.dir}")

    escritor = EscritorJsonl(args.out)
    avisos: List[str] = []
    n_partidas = 0
    n_manos = 0
    n_manos_reconstruibles = 0

    for ruta in rutas:
        partida = partida_desde_archivo(ruta, avisos)
        if partida is None:
            avisos.append(f"{ruta.name}: sin manos reconstruibles, se omite")
            continue
        n_partidas += 1
        n_manos += len(partida.manos)
        n_manos_reconstruibles += sum(1 for m in partida.manos if mano_reconstruible(m))
        escritor.escribir(partida)

    print(f"Partidas importadas: {n_partidas}  (de {len(rutas)} archivos)")
    if n_manos:
        print(f"Manos totales: {n_manos}  |  reproducibles jugada-a-jugada: {n_manos_reconstruibles} "
              f"({100 * n_manos_reconstruibles / n_manos:.0f}%)")
    else:
        print("Manos totales: 0")
    if avisos:
        print(f"\nAvisos ({len(avisos)}):")
        for a in avisos:
            print(f"  - {a}")


if __name__ == "__main__":
    main()
