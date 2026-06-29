"""
Vision especifica de la app de Corazones (es) capturada por ADB/video.

Responsabilidad: dado un screenshot BGR, leer el ESTADO de alto nivel sin OCR
de sistema (solo OpenCV + plantillas calibradas). Se apoya en dos hechos de
esta app concreta:

  1. La app NARRA el estado con un banner de texto de vocabulario finito
     ("Pasar 3 cartas a la izquierda", "El turno de Juan", "Pablo recoge la
     baza", "Tu turno", "Cartas pasadas para ti", ...). Lo clasificamos por
     plantillas (ver `calibracion/hearts_app/banners/` y `agrupar_banners.py`).
  2. La mesa son hasta 4 cartas en cruz; la mano del agente va abajo en rejilla.

Las puntuaciones NO se leen aqui: se recalculan con el motor en `replay.py` a
partir de las cartas jugadas. El banner basta para fase, direccion de pase,
turno, ganador de baza y fronteras.

Las regiones viven en `calibracion/hearts_app/regiones.json` (fracciones 0..1,
resolucion-independientes). El reconocimiento de cartas (hibrido) vive en
`vision_cartas.py` (Fase 2).

OpenCV se importa de forma perezosa (dependencia opcional, requirements-captura).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Posiciones en pantalla (NO asientos): la app fija al agente abajo.
POSICIONES = ("arriba", "izquierda", "derecha", "abajo")

# Tag de banner -> (categoria, dato). dato = posicion en pantalla o direccion.
# categoria: pase | pase_fin | turno | baza | vacio
_BANNER_SEMANTICA: Dict[str, Tuple[str, Optional[str]]] = {
    "vacio": ("vacio", None),
    "pase_izquierda": ("pase", "izquierda"),
    "pase_derecha": ("pase", "derecha"),
    "pase_enfrente": ("pase", "enfrente"),
    "pase_recibido": ("pase_fin", None),
    "turno_agente": ("turno", "abajo"),
    "turno_arriba": ("turno", "arriba"),
    "turno_izquierda": ("turno", "izquierda"),
    "turno_derecha": ("turno", "derecha"),
    "baza_agente": ("baza", "abajo"),
    "baza_arriba": ("baza", "arriba"),
    "baza_izquierda": ("baza", "izquierda"),
    "baza_derecha": ("baza", "derecha"),
}

# Firma perceptual: misma rejilla que scripts/agrupar_banners.py (debe coincidir).
_GW, _GH = 32, 8


def _firma(roi: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (_GW, _GH), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).astype(np.uint8).ravel()


# --- Regiones --------------------------------------------------------------

@dataclass
class Regiones:
    """Cajas de la UI en fracciones [x, y, w, h] (0..1)."""
    banner: List[float]
    marcador: Dict[str, List[float]]
    mesa: Dict[str, List[float]]
    mano: List[float]
    # Zona donde aparecen las cartas seleccionadas para pasar / recibidas.
    # Opcional: solo se necesita para AUTO-PASE (lectura de cartas recibidas).
    pases: Optional[List[float]] = None
    # Boton de confirmar el pase (circulo con check). Opcional: solo se necesita
    # para AUTO-PASE (tap por ADB). Si no esta calibrado, queda None.
    confirmar: Optional[List[float]] = None

    @staticmethod
    def cargar(path: str | Path) -> "Regiones":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return Regiones(banner=d["banner"], marcador=d["marcador"],
                        mesa=d["mesa"], mano=d["mano"],
                        pases=d.get("pases"),
                        confirmar=d.get("confirmar"))

    @staticmethod
    def recortar(img: np.ndarray, caja_frac: List[float]) -> np.ndarray:
        h, w = img.shape[:2]
        x, y, cw, ch = caja_frac
        x0, y0 = int(x * w), int(y * h)
        return img[y0:y0 + int(ch * h), x0:x0 + int(cw * w)]


# --- Clasificador de banner ------------------------------------------------

@dataclass
class ResultadoBanner:
    tag: str                 # etiqueta de plantilla (p.ej. "turno_arriba")
    distancia: int           # Hamming a la plantilla mas cercana
    categoria: str           # pase | pase_fin | turno | baza | vacio | desconocido
    dato: Optional[str]      # posicion en pantalla o direccion de pase


class BannerClasificador:
    """Clasifica el recorte del banner contra la biblioteca de plantillas.

    `dir_plantillas` contiene PNGs nombrados `<tag>__<i>.png`. Se clasifica por
    distancia de Hamming de la firma perceptual; si supera `umbral`, devuelve
    categoria 'desconocido' (banner nuevo -> re-correr agrupar_banners.py).
    """

    def __init__(self, dir_plantillas: str | Path, umbral: int = 40) -> None:
        self.dir = Path(dir_plantillas)
        self.umbral = umbral
        self._tpl: List[Tuple[str, np.ndarray]] = []  # (tag, firma)

    def _cargar(self) -> None:
        import cv2

        if self._tpl:
            return
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"No existe la biblioteca de banners: {self.dir}. "
                "Genera plantillas con scripts/agrupar_banners.py y curalas."
            )
        for f in sorted(self.dir.glob("*.png")):
            tag = f.stem.split("__")[0]
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                self._tpl.append((tag, _firma(img)))
        if not self._tpl:
            raise FileNotFoundError(f"Biblioteca de banners vacia: {self.dir}")

    def clasificar(self, img_o_roi: np.ndarray, regiones: Optional[Regiones] = None
                   ) -> ResultadoBanner:
        """Clasifica. Si se pasa `regiones`, recorta el banner del screenshot;
        si no, asume que ya recibe el recorte del banner."""
        self._cargar()
        roi = (Regiones.recortar(img_o_roi, regiones.banner)
               if regiones is not None else img_o_roi)
        f = _firma(roi)
        mejor_tag, mejor_d = "desconocido", 10 ** 9
        for tag, tf in self._tpl:
            d = int(np.count_nonzero(f != tf))
            if d < mejor_d:
                mejor_tag, mejor_d = tag, d
        if mejor_d > self.umbral:
            return ResultadoBanner("desconocido", mejor_d, "desconocido", None)
        cat, dato = _BANNER_SEMANTICA.get(mejor_tag, ("desconocido", None))
        return ResultadoBanner(mejor_tag, mejor_d, cat, dato)


@dataclass
class EstadoVisual:
    """Lectura de alto nivel de un screenshot: banner + cartas de la mesa."""
    banner: ResultadoBanner
    mesa: Dict[str, Optional[int]]   # posicion_pantalla -> carta_id | None


def leer_estado(img: np.ndarray, regiones: Regiones,
                banner_clf: "BannerClasificador", reconocedor) -> EstadoVisual:
    """Visión completa de un fotograma: clasifica el banner y lee la mesa.

    `reconocedor` es un `vision_cartas.ReconocedorPlantilla` (matchTemplate
    contra los naipes completos del sprite de la APK)."""
    return EstadoVisual(banner=banner_clf.clasificar(img, regiones),
                        mesa=leer_mesa(img, regiones, reconocedor))


_ANCHO_CARTA_FRAC = 0.774   # ancho / alto de un naipe (sprite APK 168x217)


@dataclass
class CartaMano:
    """Una carta localizada en la mano: su id (o None) y el punto donde tocarla."""
    carta_id: Optional[int]
    centro: Tuple[int, int]   # (x, y) en PIXELES del screenshot completo


def leer_mano_posiciones(img: np.ndarray, regiones: Regiones, reconocedor,
                         umbral_rango: float = 0.45, umbral_palo: float = 0.45
                         ) -> List[CartaMano]:
    """Como `leer_mano`, pero ademas devuelve el PUNTO de toque de cada carta en
    coordenadas del screenshot completo (para auto-juego por ADB).

    El punto de toque es el centro de la franja VISIBLE de la carta (entre su
    borde izquierdo y la siguiente carta solapada; la ultima del bloque usa su
    ancho completo): asi un tap cae siempre dentro de la carta correcta, no en la
    de encima. Recalcula posiciones cada vez que se llama, asi que tolera el
    reordenamiento de bloques tras seleccionar una carta en el pase.
    """
    from src.captura.modelos import str_a_carta_id
    from src.captura.vision_cartas import _filas_de_cartas, localizar_cartas

    H, W = img.shape[:2]
    mx, my = regiones.mano[0], regiones.mano[1]
    mx0, my0 = int(mx * W), int(my * H)
    mano_roi = Regiones.recortar(img, regiones.mano)
    filas = sorted(_filas_de_cartas(mano_roi, 0.01),
                   key=lambda b: (b[1] // 50, b[0]))
    out: List[CartaMano] = []
    for (fx, fy, fw, fh) in filas:
        blob = mano_roi[fy:fy + fh, fx:fx + fw]
        cajas = localizar_cartas(blob, 0.01)
        if not cajas:
            continue
        cardw = int(_ANCHO_CARTA_FRAC * fh)
        # El paso entre cartas (abanico) es CONSTANTE: interpolar posiciones
        # uniformes entre la primera y la ultima detectadas corrige el ruido de
        # localizacion en palos "ocupados" (♣) y sube las correlaciones a ~1.0.
        xs = [c[0] for c in cajas]
        n = len(xs)
        pos = ([int(xs[0] + (xs[-1] - xs[0]) * i / (n - 1)) for i in range(n)]
               if n > 1 else xs)
        # palo del bloque: carta mas a la derecha (entera) -> matchTemplate completo
        bx = pos[-1]
        s_palo, name_palo = reconocedor.buscar_carta(
            blob[0:fh, bx:min(bx + int(1.20 * cardw), blob.shape[1])], fh)
        palo_blk = name_palo[-1] if (name_palo and s_palo >=
                                     umbral_palo) else None
        for i, cx in enumerate(pos):
            # ventana con holgura (multiescala)
            x0 = max(0, cx - int(0.12 * cardw))
            x1 = min(cx + int(0.40 * cardw), blob.shape[1])
            win = blob[0:int(0.52 * fh), x0:x1]
            s_rango, name_rango = reconocedor.buscar_rango(win, fh)
            # Centro de la franja visible: hasta la siguiente carta (o ancho total
            # si es la ultima, acotado al borde del bloque).
            visible = (pos[i + 1] - cx) if i + 1 < n else min(cardw, fw - cx)
            tx = mx0 + fx + cx + max(1, visible) // 2
            ty = my0 + fy + fh // 2
            cid = None
            if (name_rango is not None and s_rango >= umbral_rango
                    and palo_blk is not None):
                cid = str_a_carta_id(name_rango[:-1] + palo_blk)
            out.append(CartaMano(cid, (tx, ty)))
    return out


def leer_mano(img: np.ndarray, regiones: Regiones, reconocedor,
              umbral_rango: float = 0.45, umbral_palo: float = 0.45
              ) -> List[Optional[int]]:
    """Lee las cartas de la mano del agente (zona inferior), en orden de lectura.

    Devuelve una lista de `carta_id` (o `None` si la confianza es baja) por carta
    localizada. Aprovecha que la app AGRUPA la mano por palo en bloques blancos
    separados: el PALO de un bloque se decide una vez con su carta mas a la
    derecha (visible entera) y se aplica a todas las de ese bloque; el RANGO se lee
    por carta. Si el bloque no resuelve palo, se omite el palo (carta = None).

    `reconocedor` es un `vision_cartas.ReconocedorPlantilla` (matchTemplate contra
    los naipes completos del sprite de la APK): robusto al solapamiento porque la
    ventana de busqueda absorbe el desajuste de pocos pixeles. Las plantillas
    salen del sprite de ESTA app (`scripts/cartas_desde_sprite.py`), asi que no
    dependen de la resolucion del dispositivo.
    """
    return [c.carta_id for c in
            leer_mano_posiciones(img, regiones, reconocedor, umbral_rango, umbral_palo)]


def leer_mesa(img: np.ndarray, regiones: Regiones, reconocedor
              ) -> Dict[str, Optional[int]]:
    """Lee la carta de cada posicion de la mesa (en cruz). Devuelve
    {posicion_pantalla: carta_id o None}. `reconocedor` es un
    `vision_cartas.ReconocedorPlantilla` (matchTemplate contra los naipes
    completos del sprite de la APK)."""
    from src.captura.modelos import str_a_carta_id
    from src.captura.vision_cartas import detectar_cartas

    out: Dict[str, Optional[int]] = {}
    for pos, caja in regiones.mesa.items():
        roi = Regiones.recortar(img, caja)
        cartas = detectar_cartas(roi, min_area_frac=0.05)
        if not cartas:
            out[pos] = None
            continue
        cx, cy, cw, ch = max(cartas, key=lambda b: b[2] * b[3])
        card = roi[cy:cy + ch, cx:cx + cw]
        score, name = reconocedor.buscar_carta(card, ch)
        if name is not None and score >= reconocedor.umbral:
            out[pos] = str_a_carta_id(name)
        else:
            out[pos] = None
    return out


def leer_pases(img: np.ndarray, regiones: Regiones, reconocedor
               ) -> List[Optional[int]]:
    """Lee las cartas en la zona de pases (hasta 3, seleccionadas o recibidas).

    Devuelve una lista de `carta_id` (o `None` si la confianza es baja),
    ordenada de izquierda a derecha. Si `regiones.pases` no esta calibrado,
    devuelve lista vacia.

    `reconocedor` es un `vision_cartas.ReconocedorPlantilla`.
    """
    from src.captura.modelos import str_a_carta_id
    from src.captura.vision_cartas import detectar_cartas

    if regiones.pases is None:
        return []
    roi = Regiones.recortar(img, regiones.pases)
    cartas = detectar_cartas(roi, min_area_frac=0.02)
    out: List[Optional[int]] = []
    for cx, cy, cw, ch in sorted(cartas, key=lambda b: b[0]):
        card = roi[cy:cy + ch, cx:cx + cw]
        score, name = reconocedor.buscar_carta(card, ch)
        if name is not None and score >= reconocedor.umbral:
            out.append(str_a_carta_id(name))
        else:
            out.append(None)
    return out


__all__ = [
    "POSICIONES", "Regiones", "BannerClasificador", "ResultadoBanner",
    "EstadoVisual", "leer_estado", "leer_mesa", "leer_mano",
    "CartaMano", "leer_mano_posiciones", "leer_pases",
]
