"""
Captura automática vía ADB sobre una app de Corazones del móvil.

Tres piezas, separadas por responsabilidad (SRP):

- `ClienteADB`     : plumbing ADB real (screencap, tap, swipe) por subprocess.
                     NO depende de OpenCV; sí de tener `adb` en el PATH.
- `ParserPantalla` : ABC que convierte un screenshot en `EstadoVisto`. ES la
                     parte FRÁGIL y específica de cada app → aislada tras esta
                     interfaz (DIP). `ParserPlantillas` la implementa por
                     template-matching y requiere CALIBRACIÓN (regiones +
                     plantillas de las 52 cartas). Ver `scripts/calibrar_captura.py`.
- `AdaptadorADB`   : `AdaptadorJuego` que poll-ea la pantalla, emite eventos y,
                     en el turno del agente, usa una `politica` para elegir carta
                     y la toca con el `ClienteADB`.

OpenCV se importa de forma perezosa: este módulo se puede importar sin tener
`opencv-python` instalado (solo se necesita al usar `ParserPlantillas`).
Dependencias opcionales en `requirements-captura.txt`.
"""
from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.captura.puerto import (
    AdaptadorJuego, Evento, FinMano, FinPartida, InicioMano, InicioPartida,
    JugadaObservada, PaseAgente,
)

# Política: dado lo visible, decide qué carta (id) juega el agente.
Politica = Callable[["EstadoVisto"], int]


def politica_primera_legal(estado: "EstadoVisto") -> int:
    """Placeholder: juega la primera carta de la mano. Sustituir por el modelo."""
    if not estado.mano_agente:
        raise RuntimeError("No hay cartas en la mano del agente para jugar.")
    return estado.mano_agente[0]


# --- ClienteADB -----------------------------------------------------------

class ClienteADB:
    """Wrapper mínimo de la CLI `adb`. Requiere Android Platform Tools."""

    def __init__(self, serial: Optional[str] = None, adb: str = "adb") -> None:
        self.serial = serial
        self.adb = adb

    def _base(self) -> List[str]:
        cmd = [self.adb]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def _run(self, *args: str, binario: bool = False) -> bytes:
        try:
            res = subprocess.run(
                self._base() + list(args),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No se encontro el ejecutable '{self.adb}'. Instala Android "
                "Platform Tools y ponlo en el PATH, o pasa la ruta con --adb "
                "(p.ej. --adb C:/platform-tools/adb.exe)."
            ) from e
        return res.stdout

    def dispositivos(self) -> List[str]:
        out = self._run("devices").decode("utf-8", "replace").splitlines()
        return [l.split("\t")[0] for l in out[1:] if "\tdevice" in l]

    def captura(self) -> np.ndarray:
        """Screenshot actual como array BGR (requiere OpenCV)."""
        import cv2  # import perezoso (dep opcional)
        import time

        for intento in range(3):
            png = self._run("exec-out", "screencap", "-p")
            arr = np.frombuffer(png, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            time.sleep(0.1)
        raise RuntimeError("No se pudo decodificar el screenshot de ADB "
                           "tras 3 intentos.")

    def tap(self, x: int, y: int) -> None:
        self._run("shell", "input", "tap", str(int(x)), str(int(y)))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 200) -> None:
        self._run("shell", "input", "swipe",
                  str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(ms))


# --- Parser de pantalla ---------------------------------------------------

@dataclass
class EstadoVisto:
    """Lo que el parser logra leer de un screenshot (parcial por naturaleza)."""
    fase: str = "desconocido"   # pase | jugar | fin_mano | fin_partida | desconocido
    numero_mano: Optional[int] = None
    direccion_pase: Optional[str] = None
    mano_agente: List[int] = field(default_factory=list)
    # carta_id -> (x, y) en píxeles, para poder tocarla
    posiciones_mano: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    mesa: List[Tuple[int, int]] = field(
        default_factory=list)  # (asiento, carta_id)
    turno_de: Optional[int] = None
    marcador: List[int] = field(default_factory=lambda: [0, 0, 0, 0])


class ParserPantalla(ABC):
    @abstractmethod
    def parsear(self, img: np.ndarray) -> EstadoVisto:
        """Convierte un screenshot BGR en `EstadoVisto`."""
        raise NotImplementedError


class ParserPlantillas(ParserPantalla):
    """Parser por template-matching (OpenCV).

    Requiere CALIBRACIÓN específica de la app objetivo:
      - `plantillas_dir`: 52 imágenes `<id>.png` (recortes de cada carta).
      - `regiones`: dict con cajas (x, y, w, h) de mano/mesa/marcador.

    Sin esos assets, `parsear` lanza un error claro: la visión es lo único que
    NO se puede dar hecho sin ver la app concreta. La arquitectura sí queda lista.
    """

    def __init__(
        self,
        plantillas_dir: str | Path,
        regiones: Optional[dict] = None,
        umbral: float = 0.85,
    ) -> None:
        self.plantillas_dir = Path(plantillas_dir)
        self.regiones = regiones or {}
        self.umbral = umbral
        self._plantillas: Dict[int, np.ndarray] = {}

    def _cargar_plantillas(self) -> None:
        import cv2  # perezoso

        if self._plantillas:
            return
        if not self.plantillas_dir.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio de plantillas: {self.plantillas_dir}. "
                "Genera los recortes con scripts/calibrar_captura.py."
            )
        for f in self.plantillas_dir.glob("*.png"):
            try:
                cid = int(f.stem)
            except ValueError:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                self._plantillas[cid] = img
        if not self._plantillas:
            raise FileNotFoundError(
                f"No se cargó ninguna plantilla <id>.png en {self.plantillas_dir}."
            )

    def _match_en_region(self, img: np.ndarray, caja: Tuple[int, int, int, int]):
        """Devuelve [(carta_id, (cx, cy))] de las cartas detectadas en una caja."""
        import cv2  # perezoso

        self._cargar_plantillas()
        x, y, w, h = caja
        roi = img[y:y + h, x:x + w]
        encontrados: List[Tuple[int, Tuple[int, int]]] = []
        for cid, tpl in self._plantillas.items():
            if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                continue
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv >= self.umbral:
                cx = x + maxloc[0] + tpl.shape[1] // 2
                cy = y + maxloc[1] + tpl.shape[0] // 2
                encontrados.append((cid, (cx, cy)))
        return encontrados

    def parsear(self, img: np.ndarray) -> EstadoVisto:
        if not self.regiones:
            raise NotImplementedError(
                "ParserPlantillas sin 'regiones' calibradas para la app. "
                "Define las cajas (mano/mesa/marcador) y las plantillas; ver "
                "scripts/calibrar_captura.py. La visión es específica de cada app."
            )
        estado = EstadoVisto()
        if "mano" in self.regiones:
            for cid, (cx, cy) in self._match_en_region(img, self.regiones["mano"]):
                estado.mano_agente.append(cid)
                estado.posiciones_mano[cid] = (cx, cy)
        # mesa / turno / marcador: dependen del layout de la app → calibrar.
        return estado


# --- AdaptadorADB ---------------------------------------------------------

class AdaptadorADB(AdaptadorJuego):
    """Poll-ea la pantalla, emite eventos y juega en el turno del agente.

    El bucle de polling es genérico DADO un `ParserPantalla` que funcione; la
    detección fina de transiciones (inicio de mano, pase, fin) depende de lo que
    el parser logre leer de la app concreta y suele requerir ajuste. Las partes
    app-específicas están marcadas con TODO.
    """

    def __init__(
        self,
        parser: ParserPantalla,
        politica: Politica = politica_primera_legal,
        cliente: Optional[ClienteADB] = None,
        asiento_agente: int = 0,
        app: str = "desconocida",
        poll_s: float = 1.0,
        max_partidas: int = 1,
    ) -> None:
        self.parser = parser
        self.politica = politica
        self.cliente = cliente or ClienteADB()
        self.asiento_agente = asiento_agente
        self.app = app
        self.poll_s = poll_s
        self.max_partidas = max_partidas

    def eventos(self) -> Iterator[Evento]:
        yield InicioPartida(asiento_agente=self.asiento_agente,
                            fuente=f"adb:{self.app}")
        partidas = 0
        emitidas: set = set()         # (numero_mano, baza, asiento, carta_id)
        mano_actual: Optional[int] = None
        marcador = [0, 0, 0, 0]

        while partidas < self.max_partidas:
            estado = self.parser.parsear(self.cliente.captura())

            # TODO(app): detectar inicio de mano de forma robusta para tu app.
            if estado.numero_mano is not None and estado.numero_mano != mano_actual:
                mano_actual = estado.numero_mano
                yield InicioMano(numero_mano=mano_actual,
                                 direccion_pase=estado.direccion_pase,
                                 mano_agente=list(estado.mano_agente))

            # Fase de pase: la política elige 3 (TODO: política de pase real).
            if estado.fase == "pase" and estado.mano_agente:
                dadas = estado.mano_agente[:3]
                for cid in dadas:
                    if cid in estado.posiciones_mano:
                        self.cliente.tap(*estado.posiciones_mano[cid])
                yield PaseAgente(dadas=dadas)

            # Jugadas visibles en la mesa → emitir las nuevas.
            for asiento, cid in estado.mesa:
                clave = (mano_actual, len(emitidas), asiento, cid)
                if clave not in emitidas:
                    emitidas.add(clave)
                    baza = estado.numero_mano or 0
                    yield JugadaObservada(asiento=asiento, carta_id=cid, baza=baza)

            # Turno del agente: decidir y tocar.
            if estado.fase == "jugar" and estado.turno_de == self.asiento_agente:
                cid = self.politica(estado)
                if cid in estado.posiciones_mano:
                    self.cliente.tap(*estado.posiciones_mano[cid])

            if estado.fase == "fin_partida":
                marcador = list(estado.marcador)
                partidas += 1
                yield FinPartida(marcador=marcador)
                break

            time.sleep(self.poll_s)


__all__ = [
    "ClienteADB", "ParserPantalla", "ParserPlantillas", "EstadoVisto",
    "AdaptadorADB", "Politica", "politica_primera_legal",
]
