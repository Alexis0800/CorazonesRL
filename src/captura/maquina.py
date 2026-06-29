"""
Maquina de estados de captura: convierte una secuencia de `EstadoVisual`
(banner + mesa, leidos de cada fotograma) en el stream de eventos del puerto
(`InicioMano`, `JugadaObservada`, `RemateResto`, `FinMano`, `FinPartida`).

Es PURA: no sabe de ADB ni de ficheros. La alimenta cualquier fuente de
fotogramas (ADB en vivo, una carpeta del video, un test). Asi el mismo cerebro
sirve para validar offline sobre el video y para capturar en vivo.

Claves de diseño:
  - La app fija al agente abajo. Cada posicion de pantalla -> asiento absoluto
    segun la rotacion de turno (configurable; por defecto horario en pantalla:
    abajo -> izquierda -> arriba -> derecha). El replay valida la rotacion: si
    estuviera al reves, las jugadas saldrian fuera de orden y se detecta.
  - Las 4 cartas de una baza se BUFFERIZAN y se emiten en ORDEN DE TURNO (desde
    el lider de la baza), no en orden de deteccion. Lider de la baza 1 = quien
    juega el 2 de treboles; de las siguientes = el ganador de la baza previa
    (banner "X recoge la baza").
  - El banner narra fase/turno/ganador; la mesa aporta las cartas. La
    combinacion es robusta a fotogramas perdidos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from src.dominio.carta import Carta
from src.captura.modelos import str_a_carta_id
from src.captura.puerto import (
    Evento, FinMano, FinPartida, InicioMano, JugadaObservada, RemateResto,
)
from src.captura.vision_hearts import EstadoVisual

# posicion de pantalla -> asiento absoluto (rotacion horaria, agente abajo=0).
ROTACION_HORARIA = {"abajo": 0, "izquierda": 1, "arriba": 2, "derecha": 3}
ROTACION_ANTIHORARIA = {"abajo": 0, "derecha": 1, "arriba": 2, "izquierda": 3}

_DOS_TREBOLES = str_a_carta_id("2T")
_DIRECCIONES = {"izquierda", "derecha", "enfrente"}


@dataclass
class MaquinaCaptura:
    """Consume estados y produce eventos. Llama `procesar(estado)` por fotograma
    y `finalizar()` al agotar la fuente."""

    asiento_agente: int = 0
    seat_de_posicion: Dict[str, int] = field(
        default_factory=lambda: dict(ROTACION_HORARIA))
    min_confirmaciones: int = 2           # frames estables para fiarse de una carta
    # --- estado interno ---
    _cand: Dict[str, list] = field(default_factory=dict)   # pos -> [carta_id, n]
    _en_mano: bool = False
    _numero_mano: int = 0
    _direccion: Optional[str] = None
    _baza: int = 1
    _lider: Optional[int] = None          # lider de la baza en curso
    _buf: Dict[int, int] = field(default_factory=dict)   # asiento -> carta_id
    _ultima_cat: str = ""
    _esperando_limpieza: bool = False     # baza volcada; ignorar mesa hasta vaciarse
    _avisos: List[str] = field(default_factory=list)      # diagnostico

    # --- API ---
    def procesar(self, estado: EstadoVisual) -> Iterator[Evento]:
        cat = estado.banner.categoria
        dato = estado.banner.dato
        mesa_vacia = all(v is None for v in estado.mesa.values())
        if mesa_vacia:
            self._esperando_limpieza = False

        # 1) Inicio de mano por banner de pase (solo en el FLANCO de subida).
        if cat == "pase" and dato in _DIRECCIONES and self._ultima_cat != "pase":
            if self._en_mano:                   # no se cerro la anterior: cerrar
                yield from self._fin_mano()
            yield from self._iniciar_mano(direccion=dato)

        # 2) Mano SIN pase: abrir solo con cartas en mesa Y contexto de juego real
        #    (turno/baza), no en parpadeos de animacion (banner vacio/popup).
        elif (not self._en_mano and not mesa_vacia and cat in ("turno", "baza")):
            yield from self._iniciar_mano(direccion=None)

        # 3) Cartas nuevas en la mesa -> buffer de la baza (en orden de turno).
        #    Solo en estados ASENTADOS (turno/baza): en animaciones (vacio/
        #    desconocido/pase) las cartas deslizan y se leen mal.
        if mesa_vacia:
            self._cand.clear()
        # Solo leemos en estados ASENTADOS (turno/baza): en pase, popups y
        # animaciones (vacio/desconocido) las cartas deslizan y se leen mal.
        if self._en_mano and not self._esperando_limpieza and cat in ("turno", "baza"):
            self._acumular_mesa(estado.mesa)
            if len(self._buf) == 4:
                yield from self._volcar_baza()
                self._esperando_limpieza = True

        # 4) Banner "X recoge la baza": solo marca FRONTERA (volcar resto y, si
        #    procede, cerrar la mano). El GANADOR/lider se calcula de las cartas
        #    en `_volcar_baza`, no del nombre del banner -> robusto a cualquier
        #    nombre de jugador.
        if cat == "baza":
            if self._buf:                       # baza incompleta no volcada
                yield from self._volcar_baza(lider=self._lider)
            self._esperando_limpieza = True
            if self._baza > 13:
                yield from self._fin_mano()

        self._ultima_cat = cat

    def finalizar(self, marcador: Optional[List[int]] = None) -> Iterator[Evento]:
        """Cierra mano/partida pendientes al agotar la fuente."""
        if self._en_mano:
            yield from self._fin_mano()
        yield FinPartida(marcador=list(marcador) if marcador else [0, 0, 0, 0])

    # --- internos ---
    def _iniciar_mano(self, direccion: Optional[str]) -> Iterator[Evento]:
        self._numero_mano += 1
        self._en_mano = True
        self._direccion = direccion
        self._baza = 1
        self._lider = None
        self._buf = {}
        yield InicioMano(numero_mano=self._numero_mano,
                         direccion_pase=direccion, mano_agente=[])

    def _acumular_mesa(self, mesa: Dict[str, Optional[int]]) -> None:
        for pos, cid in mesa.items():
            asiento = self.seat_de_posicion.get(pos)
            if asiento is None or asiento in self._buf:
                continue
            # confirmacion temporal: la carta debe repetirse en el slot
            cand = self._cand.get(pos)
            if cid is None:
                continue
            if cand and cand[0] == cid:
                cand[1] += 1
            else:
                self._cand[pos] = [cid, 1]
                cand = self._cand[pos]
            if cand[1] < self.min_confirmaciones:
                continue
            if cid in self._buf.values():       # misma carta en dos slots: ruido
                continue
            self._buf[asiento] = cid

    def _volcar_baza(self, lider: Optional[int] = None) -> Iterator[Evento]:
        if not self._buf:
            return
        if lider is None:
            lider = self._lider
        if lider is None:                       # baza 1: lidera quien jugo 2♣
            lider = next((a for a, c in self._buf.items() if c == _DOS_TREBOLES),
                         min(self._buf))
            self._lider = lider
        orden = [(lider + k) % 4 for k in range(4)]
        jugadas = [(a, self._buf[a]) for a in orden if a in self._buf]
        for asiento, cid in jugadas:
            yield JugadaObservada(asiento=asiento, carta_id=cid, baza=self._baza)
        # Ganador = carta mas alta del palo de salida -> lider de la baza siguiente.
        if len(jugadas) == 4:
            self._lider = self._ganador_baza(jugadas)
        self._buf = {}
        self._baza += 1

    @staticmethod
    def _ganador_baza(jugadas: List) -> int:
        """jugadas = [(asiento, carta_id)] en orden de turno. Devuelve el asiento
        que gana (carta mas alta del palo de salida)."""
        palo_salida = Carta._TODAS[jugadas[0][1]].palo
        mejor_asiento, mejor_valor = jugadas[0][0], -1
        for asiento, cid in jugadas:
            c = Carta._TODAS[cid]
            if c.palo == palo_salida and c.valor > mejor_valor:
                mejor_asiento, mejor_valor = asiento, c.valor
        return mejor_asiento

    def _fin_mano(self) -> Iterator[Evento]:
        if self._buf:                           # restos sin volcar
            yield from self._volcar_baza()
        self._en_mano = False
        yield FinMano(puntuacion=[0, 0, 0, 0])


__all__ = [
    "MaquinaCaptura", "ROTACION_HORARIA", "ROTACION_ANTIHORARIA",
]
