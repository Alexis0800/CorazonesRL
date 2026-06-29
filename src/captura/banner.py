"""
Clasificador de banners — SOLID: interfaz abstracta + implementacion + captura.

Extraido de `vision_hearts.py` para poder probar el clasificador de banners
de forma aislada, capturar plantillas especificas del dispositivo, y cambiar
la implementacion sin tocar el resto del pipeline de vision.

Jerarquia:
  IBannerClasificador  (ABC)
  └── BannerClasificador  (distancia euclidea sobre firma grayscale)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Tag de banner → (categoria, dato) ──────────────────────────────────────
# dato = posicion en pantalla o direccion de pase.
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


# ── Resultado ──────────────────────────────────────────────────────────────

@dataclass
class ResultadoBanner:
    """Resultado de clasificar un banner."""
    tag: str          # etiqueta de plantilla (p.ej. "turno_arriba")
    distancia: float  # distancia euclidea (0 = identico, 2 = opuesto)
    categoria: str    # pase | pase_fin | turno | baza | vacio | desconocido
    dato: Optional[str]  # posicion en pantalla o direccion de pase


# ── Firma perceptual ───────────────────────────────────────────────────────
# Raw grayscale normalizado (NO binarizado): robusto a diferencias de brillo
# entre dispositivos y preserva las intensidades de los bordes de las letras.
# 64×16 = 1024 dimensiones.
_GW, _GH = 64, 16


def firma(roi: np.ndarray) -> np.ndarray:
    """Vector unitario de 1024 floats: raw grayscale reescalado."""
    import cv2

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (_GW, _GH), interpolation=cv2.INTER_AREA)
    g = g.ravel()
    g -= g.mean()
    norm = np.linalg.norm(g)
    if norm > 1e-6:
        g /= norm
    return g


# ── Interfaz ───────────────────────────────────────────────────────────────

class IBannerClasificador(ABC):
    """Interfaz SOLID: cualquier clasificador de banner debe implementar esto."""

    @abstractmethod
    def clasificar(self, roi: np.ndarray) -> ResultadoBanner:
        """Clasifica un recorte de banner (ya recortado, no el screenshot completo).

        Si necesitas pasar un screenshot completo, usa `BannerClasificador`
        directamente con el parametro `regiones`."""
        ...

    @abstractmethod
    def clasificar_verbose(self, roi: np.ndarray
                           ) -> Tuple[ResultadoBanner, List[Tuple[str, float]]]:
        """Como `clasificar`, pero devuelve ademas el top-5 de distancias."""
        ...


# ── Implementacion por plantillas ──────────────────────────────────────────

class BannerClasificador(IBannerClasificador):
    """Clasifica el banner por distancia euclidea contra una biblioteca de
    plantillas PNG.

    Las plantillas viven en un directorio con archivos `<tag>__<i>.png`.
    La firma perceptual es un vector grayscale normalizado de 1024 dims.
    Si la distancia minima supera `umbral`, devuelve 'desconocido'.
    """

    def __init__(self, dir_plantillas: str | Path, umbral: float = 1.5) -> None:
        self.dir = Path(dir_plantillas)
        self.umbral = umbral
        self._tpl: List[Tuple[str, np.ndarray]] = []  # (tag, firma)

    # ── carga ──

    def _cargar(self) -> None:
        import cv2

        if self._tpl:
            return
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"No existe la biblioteca de banners: {self.dir}. "
                "Genera plantillas con scripts/agrupar_banners.py y curalas, "
                "o captura banners del dispositivo con capturar_banner()."
            )
        for f in sorted(self.dir.glob("*.png")):
            tag = f.stem.split("__")[0]
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                self._tpl.append((tag, firma(img)))
        if not self._tpl:
            raise FileNotFoundError(f"Biblioteca de banners vacia: {self.dir}")

    @property
    def num_plantillas(self) -> int:
        self._cargar()
        return len(self._tpl)

    # ── clasificacion ──

    def clasificar(self, roi: np.ndarray) -> ResultadoBanner:
        """Clasifica un RECORTE de banner (no el screenshot completo).
        Para pasar un screenshot con regiones, usa `clasificar_con_regiones`."""
        resultado, _ = self._clasificar_impl(roi)
        return resultado

    def clasificar_verbose(self, roi: np.ndarray
                           ) -> Tuple[ResultadoBanner, List[Tuple[str, float]]]:
        """Como `clasificar`, pero devuelve ademas el top-5 de distancias."""
        return self._clasificar_impl(roi)

    def clasificar_con_regiones(self, img: np.ndarray, regiones
                                ) -> ResultadoBanner:
        """Recorta el banner del screenshot usando `regiones.banner` y clasifica."""
        roi = regiones.recortar(img, regiones.banner)
        return self.clasificar(roi)

    def clasificar_verbose_con_regiones(self, img: np.ndarray, regiones
                                        ) -> Tuple[ResultadoBanner,
                                                   List[Tuple[str, float]]]:
        """Como `clasificar_con_regiones`, pero devuelve top-5."""
        roi = regiones.recortar(img, regiones.banner)
        return self.clasificar_verbose(roi)

    def _clasificar_impl(self, roi: np.ndarray
                         ) -> Tuple[ResultadoBanner, List[Tuple[str, float]]]:
        self._cargar()
        f = firma(roi)  # vector unitario
        mejor_tag, mejor_d = "desconocido", float("inf")
        distancias: List[Tuple[str, float]] = []
        for tag, tf in self._tpl:
            corr = float(np.dot(f, tf))
            d = float(np.sqrt(max(0.0, 2.0 - 2.0 * corr)))
            distancias.append((tag, d))
            if d < mejor_d:
                mejor_tag, mejor_d = tag, d
        resultado = ResultadoBanner(
            "desconocido", mejor_d, "desconocido", None)
        if mejor_d <= self.umbral:
            cat, dato = _BANNER_SEMANTICA.get(mejor_tag, ("desconocido", None))
            resultado = ResultadoBanner(mejor_tag, mejor_d, cat, dato)
        distancias.sort(key=lambda x: x[1])
        return resultado, distancias[:5]


# ── Herramienta de captura de banners del dispositivo ──────────────────────

def capturar_banner(img: np.ndarray, regiones,
                    dir_salida: str | Path,
                    etiqueta: str = "") -> Path:
    """Guarda un recorte del banner en `dir_salida` para crear plantillas
    especificas del dispositivo actual.

    Usa `regiones.banner` para recortar. Si se da `etiqueta`, se usa como
    prefijo del nombre (ej. \"pase_enfrente\"). Ideal para correr en modo
    debug y luego renombrar los PNGs manualmente.

    Devuelve el path del archivo guardado.
    """
    import cv2

    d = Path(dir_salida)
    d.mkdir(parents=True, exist_ok=True)
    import time
    ts = int(time.time() * 1000)
    prefijo = f"{etiqueta}__" if etiqueta else ""
    nombre = d / f"{prefijo}banner_{ts}.png"
    roi = regiones.recortar(img, regiones.banner)
    cv2.imwrite(str(nombre), roi)
    return nombre
