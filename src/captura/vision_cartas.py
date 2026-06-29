"""
Vision de cartas de la app (Fase 2, hibrida): detecta cartas en una region y
reconoce cada una por la ESQUINA (rango + palo), con plantillas calibradas.

Pipeline:
  1. `detectar_cartas(roi)`  -> bounding boxes de cartas (blancas sobre fondo).
  2. `recortar_esquina(carta)` -> recorte del indice superior-izquierdo.
  3. `Reconocedor.reconocer(esquina)` -> carta_id (0-51) por plantillas.

El reconocimiento es por plantillas de esquina (firma perceptual + Hamming),
descubiertas con `scripts/agrupar_cartas.py` y curadas en
`calibracion/hearts_app/cartas/<carta>__<i>.png` (nombre = str de carta, p.ej.
'QP' = Q de picas; ver modelos.carta_a_str / str_a_carta_id).

OpenCV perezoso (dependencia opcional).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.captura.modelos import str_a_carta_id

# Fraccion de la carta que ocupa el recorte de esquina (alto y ancho).
_ESQ_W, _ESQ_H = 0.42, 0.30
# Firma perceptual de la esquina.
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


def recortar_esquina(carta_bgr: np.ndarray) -> np.ndarray:
    """Recorte del indice superior-izquierdo (rango + palo) de una carta entera."""
    h, w = carta_bgr.shape[:2]
    return carta_bgr[0:int(_ESQ_H * h), 0:int(_ESQ_W * w)]


# --- Reconocimiento hibrido: color + rango (glifo) + pip (forma) -----------
#
# La firma en gris no separa treboles de picas (mismo negro; solo cambia la
# forma del pip). Por eso reconocemos en 3 piezas, dentro de la esquina canonica
# 70x96: el COLOR (rojo->corazon/diamante, negro->pica/trebol) acota a 2 palos;
# el RANGO se lee por el glifo (independiente del color); y el PIP desempata los
# 2 palos del color por su forma.

_CW, _CH = 70, 96
_RANK_BOX = (0, 40, 2, 58)    # x0, x1, y0, y1 en la esquina canonica (glifo rango)
_PIP_BOX = (2, 52, 50, 96)    # pip de cuerpo bajo el rango: el mas consistente
_RG_W, _RG_H = 20, 26         # rejilla de firma del rango
_PG_W, _PG_H = 28, 26         # rejilla de firma del pip (forma del palo)

# palo -> es_rojo
_PALO_ROJO = {"C": True, "D": True, "P": False, "T": False}


def firma_esquina(esquina: np.ndarray) -> np.ndarray:
    """Firma perceptual gruesa de la esquina (solo para CLUSTERING, no para
    reconocer: en gris no separa treboles de picas)."""
    import cv2

    g = cv2.cvtColor(esquina, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (_GW, _GH), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).astype(np.uint8).ravel()


def _canon(corner: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(corner, (_CW, _CH), interpolation=cv2.INTER_AREA)


def _bits(region: np.ndarray, gw: int, gh: int) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
    return (g < g.mean()).astype(np.uint8).ravel()   # tinta (oscuro) = 1


def _sub(corner_canon: np.ndarray, box) -> np.ndarray:
    x0, x1, y0, y1 = box
    return corner_canon[y0:y1, x0:x1]


def firma_rango(corner_canon: np.ndarray) -> np.ndarray:
    return _bits(_sub(corner_canon, _RANK_BOX), _RG_W, _RG_H)


def firma_pip(corner_canon: np.ndarray) -> np.ndarray:
    return _bits(_sub(corner_canon, _PIP_BOX), _PG_W, _PG_H)


def es_rojo(corner_canon: np.ndarray) -> bool:
    """True si la tinta del indice es roja (corazon/diamante)."""
    c = corner_canon.astype(np.int32)
    b, g, r = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    ink = (r + g + b) < 620                       # pixel no-blanco
    if int(ink.sum()) < 12:
        return False
    return (r[ink].mean() - np.maximum(g[ink].mean(), b[ink].mean())) > 28


# Region (fracciones de la carta) donde vive el pip de cuerpo bajo el indice.
_PIP_REG = (0.0, 0.30, 0.10, 0.46)   # x0,x1,y0,y1 fracciones de la carta
_SOLID_CLUB = 0.81                    # solidez < => trebol; >= => pica


def _palo_por_forma(card_bgr: np.ndarray, rojo: bool) -> Tuple[Optional[str], float]:
    """Desempata el palo dentro del color por la FORMA del pip de cuerpo.

    Reglas geometricas (invariantes a escala/posicion):
      - rojo:  corazon tiene 2 lobulos arriba; diamante 1 punta (y muy convexo).
      - negro: trebol tiene huecos entre lobulos (solidez baja); pica es un
               arrowhead mas solido (solidez alta).
    Devuelve (palo, solidez) o (None, 0) si no halla pip.
    """
    import cv2

    h, w = card_bgr.shape[:2]
    x0, x1, y0, y1 = _PIP_REG
    reg = card_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if reg.size == 0:
        return None, 0.0
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    ink = (g < 120).astype(np.uint8)
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    if bw * bh < 60:
        return None, 0.0
    area = float(cv2.contourArea(c))
    harea = float(cv2.contourArea(cv2.convexHull(c))) or 1.0
    solid = area / harea
    pip = ink[by:by + bh, bx:bx + bw]
    top = pip[0:max(1, int(0.30 * bh)), :]
    col = top.sum(axis=0) > 0
    runs, prev = 0, False
    for v in col:
        if v and not prev:
            runs += 1
        prev = v
    if rojo:
        palo = "C" if (runs >= 2 and solid < 0.88) else "D"
    else:
        palo = "T" if solid < _SOLID_CLUB else "P"
    return palo, round(solid, 3)


@dataclass
class ResultadoCarta:
    carta_id: Optional[int]
    distancia: int                # distancia del rango (menor = mejor)
    rojo: Optional[bool] = None
    solidez: float = 0.0


class Reconocedor:
    """Reconoce una carta: color (rojo/negro) + rango (glifo, por plantilla) +
    palo (forma del pip de cuerpo). El rango usa la biblioteca de esquinas
    `<carta>__<i>.png`; el palo es por reglas geometricas (sin plantilla).

    Devuelve `carta_id=None` si la confianza del rango es baja: mejor no emitir
    una carta equivocada (el naipe sigue en la mesa varios fotogramas y se
    reintenta). La validacion final la hace el motor en el replay.
    """

    def __init__(self, dir_plantillas: str | Path, umbral_rango: int = 130) -> None:
        self.dir = Path(dir_plantillas)
        self.umbral_rango = umbral_rango
        self._rango_tpl: List[Tuple[str, np.ndarray]] = []   # (rango, firma)

    def _cargar(self) -> None:
        import cv2

        if self._rango_tpl:
            return
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"No existe la biblioteca de cartas: {self.dir}. "
                "Genera plantillas con scripts/agrupar_cartas.py y curalas."
            )
        for f in sorted(self.dir.glob("*.png")):
            nombre = f.stem.split("__")[0]
            if len(nombre) < 2 or nombre[-1] not in _PALO_ROJO:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            self._rango_tpl.append((nombre[:-1], firma_rango(_canon(img))))
        if not self._rango_tpl:
            raise FileNotFoundError(f"Biblioteca de cartas vacia: {self.dir}")

    def reconocer_carta(self, card_bgr: np.ndarray) -> ResultadoCarta:
        """Reconoce a partir de la CARTA entera detectada (no solo la esquina):
        necesita el cuerpo para leer la forma del pip."""
        self._cargar()
        corner = recortar_esquina(card_bgr)
        c = _canon(corner)
        rojo = es_rojo(c)
        rb = firma_rango(c)
        rango, rd = None, 10 ** 9
        for rg, tf in self._rango_tpl:
            d = int(np.count_nonzero(rb != tf))
            if d < rd:
                rango, rd = rg, d
        palo, solid = _palo_por_forma(card_bgr, rojo)
        if rango is None or palo is None or rd > self.umbral_rango:
            return ResultadoCarta(None, rd, rojo, solid)
        return ResultadoCarta(str_a_carta_id(rango + palo), rd, rojo, solid)

    def reconocer(self, esquina: np.ndarray) -> ResultadoCarta:
        """Compat: reconoce desde un recorte de esquina (sin forma de pip ->
        palo solo por color). Prefiere `reconocer_carta` con la carta entera."""
        self._cargar()
        c = _canon(esquina)
        rojo = es_rojo(c)
        rb = firma_rango(c)
        rango, rd = None, 10 ** 9
        for rg, tf in self._rango_tpl:
            d = int(np.count_nonzero(rb != tf))
            if d < rd:
                rango, rd = rg, d
        if rango is None or rd > self.umbral_rango:
            return ResultadoCarta(None, rd, rojo)
        # sin cuerpo no se puede la forma: elige el palo "por defecto" del color
        palo = "C" if rojo else "P"
        return ResultadoCarta(str_a_carta_id(rango + palo), rd, rojo)


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
    "localizar_cartas", "detectar_cartas", "recortar_esquina",
    "firma_esquina", "firma_rango", "firma_pip", "es_rojo",
    "Reconocedor", "ResultadoCarta", "ReconocedorPlantilla",
]
