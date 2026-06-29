"""
Vision de cartas de la app: detecta cartas en una region y las reconoce por
matchTemplate contra los naipes completos del sprite de la APK
(`ReconocedorPlantilla`).

Pipeline:
  1. `detectar_cartas(roi)` -> bounding boxes de cartas sueltas (mesa/pases).
  2. `localizar_cartas(roi)` -> cajas de cartas solapadas (mano en abanico).
  3. `ReconocedorPlantilla.buscar_carta()` -> carta_id por correlacion (carta entera).
  4. `ReconocedorPlantilla.buscar_rango()` -> rango por correlacion (esquina solapada).

Las plantillas se generan del sprite de la APK con
`scripts/cartas_desde_sprite.py` y viven en
`calibracion/hearts_app/cartas_completas/`.

`firma_esquina` se conserva solo para CLUSTERING (scripts/agrupar_cartas.py).

OpenCV perezoso (dependencia opcional).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Firma perceptual de la esquina (para firma_esquina, usado por agrupar_cartas.py).
_GW, _GH = 16, 16


def _mascara_blanco(roi: np.ndarray) -> np.ndarray:
    """Mascara de zonas claras (cuerpo de las cartas) en HSV."""
    import cv2

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s < 70) & (v > 160)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _mascara_tinta(roi: np.ndarray) -> np.ndarray:
    """Mascara de tinta (rango/palo): oscuro o muy saturado sobre blanco."""
    import cv2

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return ((v < 140) | (s > 110)).astype(np.uint8)


def _filas_de_cartas(roi: np.ndarray, min_area_frac: float
                     ) -> List[Tuple[int, int, int, int]]:
    """Blobs blancos grandes = filas de cartas (o carta suelta en la mesa)."""
    import cv2

    H, W = roi.shape[:2]
    mask = _mascara_blanco(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filas = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area_frac * H * W and h >= 0.25 * H:
            filas.append((x, y, w, h))
    filas.sort(key=lambda b: b[1])
    return filas


# Parametros de separacion de la mano (cartas solapadas en abanico). Calibrados
# sobre capturas reales: la franja superior debe ser FINA para ver solo el glifo
# del rango (no los pips del cuerpo, que estan mas abajo y fusionarian todo).
_FRANJA_FRAC = 0.08      # alto de la franja con el glifo de rango (frac. de la carta)
_GAP_FUSION_FRAC = 0.10  # runs mas cercanos que esto = misma carta (rango+pip esquina)
_MARGEN_FRAC = 0.04      # margen a la izquierda del glifo hasta el borde de la carta
_ANCHO_CARTA_FRAC = 0.774  # ancho de carta / alto (sprite APK 168x217)


def localizar_cartas(
    roi: np.ndarray,
    min_area_frac: float = 0.01,
) -> List[Tuple[int, int, int, int]]:
    """Cajas (x, y, w, h) de cada CARTA de la mano (abanico solapado) en `roi`,
    en orden de lectura (fila arriba->abajo, izquierda->derecha).

    Las cartas se solapan hacia la derecha, asi que de cada carta solo queda
    visible su franja izquierda con el indice (rango + pip de esquina). Se ubican
    detectando los grupos de tinta del RANGO en una franja superior fina (donde
    aun no hay pips de cuerpo). Cada caja se extiende `_ANCHO_CARTA_FRAC * alto`
    hacia la derecha (acotada a la siguiente carta / borde): la ultima de cada
    fila queda entera (tiene cuerpo -> sirve para leer el palo por forma).
    """
    cajas: List[Tuple[int, int, int, int]] = []
    for (fx, fy, fw, fh) in _filas_de_cartas(roi, min_area_frac):
        franja_h = max(1, int(_FRANJA_FRAC * fh))
        sub = roi[fy:fy + franja_h, fx:fx + fw]
        col = _mascara_tinta(sub).sum(axis=0)
        umbral = max(2.0, 0.06 * franja_h)
        activos = col > umbral
        # runs contiguos de columnas activas
        x = 0
        runs = []
        while x < len(activos):
            if activos[x]:
                x0 = x
                while x < len(activos) and activos[x]:
                    x += 1
                runs.append([x0, x])
            else:
                x += 1
        # fusionar runs cercanos: rango y pip de esquina de la MISMA carta van
        # pegados; entre cartas hay blanco mas ancho.
        gap_fusion = int(_GAP_FUSION_FRAC * fh)
        fusion = []
        for r in runs:
            if fusion and r[0] - fusion[-1][1] <= gap_fusion:
                fusion[-1][1] = r[1]
            else:
                fusion.append(list(r))
        grupos = [g for g in fusion if g[1] - g[0] >= max(6, int(0.05 * fh))]
        margen = int(_MARGEN_FRAC * fh)
        ancho = int(_ANCHO_CARTA_FRAC * fh)
        for i, (x0, _x1) in enumerate(grupos):
            cx = max(0, fx + x0 - margen)
            # ancho: hasta la siguiente carta (solapada) o, si es la ultima, el
            # ancho completo de naipe (acotado al borde del blob).
            if i + 1 < len(grupos):
                cw = (fx + grupos[i + 1][0] - margen) - cx
            else:
                cw = ancho
            cw = min(cw, fx + fw - cx)
            cajas.append((cx, fy, cw, fh))
    return cajas


def detectar_cartas(
    roi: np.ndarray,
    min_area_frac: float = 0.05,
) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes de carta SUELTA (mesa): el mayor blob blanco por slot."""
    filas = _filas_de_cartas(roi, min_area_frac)
    return filas


# palo -> es_rojo (usado por ReconocedorPlantilla._cargar para filtrar PNGs)
_PALO_ROJO = {"C": True, "D": True, "P": False, "T": False}


def firma_esquina(esquina: np.ndarray) -> np.ndarray:
    """Firma perceptual gruesa de la esquina (solo para CLUSTERING, no para
    reconocer: en gris no separa treboles de picas)."""
    import cv2

    g = cv2.cvtColor(esquina, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (_GW, _GH), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).astype(np.uint8).ravel()


# --- Reconocimiento por plantilla del sprite (matchTemplate) ----------------
#
# Para la MANO (cartas solapadas en abanico) el reconocimiento por firma de
# esquina (Reconocedor) es fragil: un desajuste de pocos pixeles en el recorte
# dispara el Hamming. Como el arte de la app es identico al sprite de la APK
# (`cartas_desde_sprite.py`), `matchTemplate` con correlacion normalizada y una
# ventana de busqueda pequena es mucho mas robusto (absorbe el offset): da
# correlaciones ~1.0 en cartas bien renderizadas.

# Recorte de esquina del template para leer el RANGO (frac. de la carta).
_TPL_RANK_WF, _TPL_RANK_HF = 0.24, 0.34
# Escalas a probar (multiescala): la mano levantada para pasar y la mano en la
# mesa se renderizan a tamanos algo distintos; probar varias escalas y quedarse
# con la mejor correlacion absorbe esa diferencia (sube de ~0.6 ambiguo a ~1.0).
_ESCALAS = (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)


class ReconocedorPlantilla:
    """Reconoce cartas por correlacion (matchTemplate) contra los naipes
    completos del sprite de la APK (`<carta>.png`, p.ej. de
    `calibracion/hearts_app/cartas_completas/`).

    - `buscar_carta`: empareja la CARTA ENTERA (para una carta visible completa,
      p.ej. la mas a la derecha de un bloque de la mano, o una carta de la mesa).
    - `buscar_rango`: empareja solo la esquina del rango (para cartas solapadas
      de las que solo se ve el indice). Devuelve la mejor correlacion; el llamante
      filtra por umbral.
    """

    def __init__(self, dir_completas: str | Path, umbral: float = 0.5) -> None:
        self.dir = Path(dir_completas)
        self.umbral = umbral
        self._tpl: List[Tuple[str, np.ndarray]] = []   # (carta_str, gris full)

    def _cargar(self) -> None:
        import cv2

        if self._tpl:
            return
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"No existe la biblioteca de cartas completas: {self.dir}. "
                "Genérala con scripts/cartas_desde_sprite.py."
            )
        for f in sorted(self.dir.glob("*.png")):
            nombre = f.stem
            if len(nombre) < 2 or nombre[-1] not in _PALO_ROJO:
                continue
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                self._tpl.append((nombre, im))
        if not self._tpl:
            raise FileNotFoundError(f"Biblioteca de cartas completas vacía: {self.dir}")

    @staticmethod
    def _escalar(tpl: np.ndarray, alto: int, wf: float, hf: float,
                 escala: float = 1.0) -> np.ndarray:
        import cv2

        H, W = tpl.shape[:2]
        sub = tpl[0:max(1, int(hf * H)), 0:max(1, int(wf * W))]
        h = max(1, int(alto * hf * escala))
        w = max(1, int(sub.shape[1] * h / sub.shape[0]))
        return cv2.resize(sub, (w, h))

    def _buscar(self, win_bgr: np.ndarray, alto_carta: int, wf: float, hf: float
                ) -> Tuple[float, Optional[str]]:
        import cv2

        self._cargar()
        win = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2GRAY)
        mejor: Tuple[float, Optional[str]] = (-2.0, None)
        for nombre, tpl in self._tpl:
            for esc in _ESCALAS:
                t = self._escalar(tpl, alto_carta, wf, hf, esc)
                if t.shape[0] > win.shape[0] or t.shape[1] > win.shape[1]:
                    continue
                s = float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max())
                if s > mejor[0]:
                    mejor = (s, nombre)
        return mejor

    def buscar_carta(self, win_bgr: np.ndarray, alto_carta: int
                     ) -> Tuple[float, Optional[str]]:
        """(score, carta_str) emparejando la carta entera (alto ~95% de la carta)."""
        return self._buscar(win_bgr, alto_carta, 1.0, 0.95)

    def buscar_rango(self, win_bgr: np.ndarray, alto_carta: int
                     ) -> Tuple[float, Optional[str]]:
        """(score, carta_str) emparejando solo la esquina del rango. Usa el RANGO
        del resultado; el palo de la esquina no es fiable (decídelo por bloque)."""
        return self._buscar(win_bgr, alto_carta, _TPL_RANK_WF, _TPL_RANK_HF)


__all__ = [
    "localizar_cartas", "detectar_cartas", "firma_esquina",
    "ReconocedorPlantilla",
]
