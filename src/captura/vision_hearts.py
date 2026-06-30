"""
Vision especifica de la app de Corazones (es) capturada por ADB/video.

Responsabilidad: dado un screenshot BGR, leer el ESTADO de alto nivel sin OCR
de sistema (solo OpenCV + plantillas calibradas).

El reconocimiento de banners ahora vive en `src/captura/banner.py`
(SOLID: IBannerClasificador + BannerClasificador + capturar_banner).
El reconocimiento de cartas vive en `vision_cartas.py`.

Las puntuaciones NO se leen aqui: se recalculan con el motor en `replay.py` a
partir de las cartas jugadas.

OpenCV se importa de forma perezosa (dependencia opcional, requirements-captura).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Re-exportar desde banner.py (compatibilidad hacia atras) ──
from src.captura.banner import (
    BannerClasificador,
    BannerClasificadorTexto,
    IBannerClasificador,
    ResultadoBanner,
    capturar_banner,
)

# Posiciones en pantalla (NO asientos): la app fija al agente abajo.
POSICIONES = ("arriba", "izquierda", "derecha", "abajo")

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

# --- Estado visual ----------------------------------------------------------


@dataclass
class EstadoVisual:
    """Lectura de alto nivel de un screenshot: banner + cartas de la mesa."""
    banner: ResultadoBanner
    mesa: Dict[str, Optional[int]]   # posicion_pantalla -> carta_id | None


def leer_estado(img: np.ndarray, regiones: Regiones,
                banner_clf, reconocedor) -> EstadoVisual:
    """Visión completa de un fotograma: clasifica el banner y lee la mesa.

    `banner_clf` es cualquier `IBannerClasificador` (normalmente
    `BannerClasificador`).
    `reconocedor` es un `vision_cartas.ReconocedorPlantilla`."""
    return EstadoVisual(
        banner=banner_clf.clasificar_con_regiones(img, regiones),
        mesa=leer_mesa(img, regiones, reconocedor),
    )


_ANCHO_CARTA_FRAC = 0.774   # ancho / alto de un naipe (sprite APK 168x217)


def _escalas_cerca(esc: Optional[float]):
    """Vecindario de escalas alrededor de `esc` (la mejor escala del bloque).

    Todas las cartas del abanico estan al MISMO tamano fisico, asi que una vez
    medida la escala del bloque solo hay que probar esa ±1 paso en vez de las 10
    de `_ESCALAS`. None → todas (no se pudo medir)."""
    from src.captura.vision_cartas import _ESCALAS
    if esc is None or esc not in _ESCALAS:
        return None
    i = _ESCALAS.index(esc)
    lo, hi = max(0, i - 1), min(len(_ESCALAS), i + 2)
    return _ESCALAS[lo:hi]


@dataclass
class CartaMano:
    """Una carta localizada en la mano: su id (o None) y el punto donde tocarla."""
    carta_id: Optional[int]
    centro: Tuple[int, int]   # (x, y) en PIXELES del screenshot completo
    # 'mitad' | 'carta' | 'rango' | '' (no reconocida)
    metodo: str = ""
    card_h: int = 0           # alto de la carta en px (alto del bloque)
    # ancho VISIBLE de ESTA carta en px (gap hasta la sig)
    visible_w: int = 0


def _leer_mano_filtrada(img: np.ndarray, regiones: Regiones, reconocedor,
                        umbral_rango: float, umbral_palo: float,
                        umbral_mitad: float, umbral_carta: float,
                        solo_palo: Optional[str] = None,
                        buscar_id: Optional[int] = None) -> List[CartaMano]:
    """Núcleo de lectura de la mano. `leer_mano_posiciones` y `localizar_carta`
    son envoltorios sobre esta.

    - `solo_palo`: si se da, SOLO reconoce las cartas de los bloques cuyo palo
      coincide (los demás se saltan tras detectar su palo). Ahorra el match de
      todas las cartas de los otros bloques cuando solo interesa uno.
    - `buscar_id`: si se da, devuelve [esa carta] en cuanto la reconoce (salida
      temprana); si no aparece, devuelve lista vacía.

    El palo de un bloque se determina con su carta más a la derecha (visible
    entera) y se reusa para todas las suyas. El punto de toque es el centro de la
    franja VISIBLE de cada carta (un tap cae siempre dentro de la carta correcta,
    no en la de encima). Recalcula posiciones en cada llamada → tolera el
    reordenamiento de bloques tras cada selección del pase."""
    from src.captura.modelos import str_a_carta_id, carta_a_str
    from src.captura.vision_cartas import _filas_de_cartas, localizar_cartas

    # Si buscamos una carta concreta, basta con reconocer su propio palo.
    if solo_palo is None and buscar_id is not None:
        solo_palo = carta_a_str(buscar_id)[-1]

    H, W = img.shape[:2]
    mx, my = regiones.mano[0], regiones.mano[1]
    mx0, my0 = int(mx * W), int(my * H)
    mano_roi = Regiones.recortar(img, regiones.mano)
    filas = sorted(_filas_de_cartas(mano_roi, 0.01),
                   key=lambda b: (b[1] // 50, b[0]))
    out: List[CartaMano] = []
    # Toda la mano se renderiza a UNA sola escala fisica: el primer bloque que la
    # mide siembra la escala de los siguientes (evita rebarrer las 10 escalas).
    esc_hint = None
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
        # palo del bloque desde la carta mas a la derecha (la mas visible).
        # Probamos carta entera > mitad > rango, quedandonos con el de mayor
        # score por encima de su umbral. Asi determinamos el palo incluso si
        # la carta no esta 100% visible o el template no encaja perfecto.
        bx = pos[-1]
        b_visible = min(cardw, fw - bx)
        palo_blk = None
        # mejor escala del bloque (la fijamos para las 13 cartas)
        esc_blk = None
        blk_x1 = min(bx + int(1.20 * cardw), blob.shape[1])

        escalas_hint = _escalas_cerca(esc_hint)

        # 1) carta completa
        s_c, name_c, e_c = reconocedor.buscar_carta(
            blob[0:fh, bx:blk_x1], fh, escalas=escalas_hint)
        if name_c and s_c >= umbral_palo:
            palo_blk = name_c[-1]
            esc_blk = e_c

        # 2) mitad (si sigue sin palo)
        if palo_blk is None and b_visible < 0.85 * cardw:
            b_x1_m = min(bx + int(0.48 * cardw), blob.shape[1])
            b_win_m = blob[0:int(0.95 * fh), bx:b_x1_m]
            s_bm, name_bm, e_bm = reconocedor.buscar_mitad(
                b_win_m, fh, escalas=escalas_hint)
            if name_bm and s_bm >= umbral_mitad:
                palo_blk = name_bm[-1]
                esc_blk = e_bm

        # 3) rango (ultimo recurso; palo de esquina es ruidoso pero
        #    score alto da confianza)
        if palo_blk is None:
            b_x1_r = min(bx + int(0.40 * cardw), blob.shape[1])
            b_win_r = blob[0:int(0.52 * fh), bx:b_x1_r]
            s_br, name_br, _ = reconocedor.buscar_rango(
                b_win_r, fh, escalas=escalas_hint)
            if name_br and s_br >= umbral_rango + 0.15:
                palo_blk = name_br[-1]

        # Escalas a probar por carta: si conocemos la escala del bloque, solo su
        # vecindario (todas las cartas estan al mismo tamano); si no, todas.
        if esc_blk is not None:
            esc_hint = esc_blk
        # ── filtro de bloque: si solo interesa un palo, saltar los bloques con
        #    palo CONOCIDO y distinto (ya sembramos esc_hint; nos ahorramos
        #    reconocer sus cartas). Un bloque con palo desconocido (None) NO se
        #    salta: no se puede descartar que contenga la carta buscada. ──
        if (solo_palo is not None and palo_blk is not None
                and palo_blk != solo_palo):
            continue
        escalas_blk = _escalas_cerca(esc_blk)
        for i, cx in enumerate(pos):
            # ── porción visible de ESTA carta ──
            visible = (pos[i + 1] - cx) if i + 1 < n else min(cardw, fw - cx)
            tx = mx0 + fx + cx + max(1, visible) // 2
            ty = my0 + fy + fh // 2
            cid = None
            metodo = ""

            # ¿Está apilada? → no se ve el ancho completo
            apilada = visible < 0.85 * cardw

            if apilada:
                # ── MITAD IZQUIERDA (~48% ancho × 95% alto) ──
                cap_w = min(int(0.48 * cardw),
                            max(1, visible + int(0.05 * cardw)))
                x0_m = max(0, cx - int(0.05 * cardw))
                x1_m = min(cx + cap_w, blob.shape[1])
                if x1_m > x0_m:
                    win_m = blob[0:int(0.95 * fh), x0_m:x1_m]
                    s_m, name_m, _ = reconocedor.buscar_mitad(
                        win_m, fh, palo=(palo_blk or ""), escalas=escalas_blk)
                    # Aceptar si: (a) el palo coincide con el del bloque, o
                    # (b) no se conoce el palo del bloque (mitad da palo propio).
                    if name_m is not None and s_m >= umbral_mitad:
                        if palo_blk is None or name_m[-1] == palo_blk:
                            cid = str_a_carta_id(name_m)
                            metodo = "mitad"
                # fallback: solo rango + palo del bloque
                if cid is None:
                    x0_r = max(0, cx - int(0.12 * cardw))
                    x1_r = min(cx + int(0.40 * cardw), blob.shape[1])
                    win_r = blob[0:int(0.52 * fh), x0_r:x1_r]
                    s_r, name_r, _ = reconocedor.buscar_rango(
                        win_r, fh, escalas=escalas_blk)
                    if palo_blk is not None:
                        # Modo normal: rango de la esquina + palo del bloque
                        if (name_r is not None and s_r >= umbral_rango):
                            cid = str_a_carta_id(name_r[:-1] + palo_blk)
                            metodo = "rango"
                    else:
                        # Sin palo de bloque: aceptar rango con su propio palo
                        # (umbral mas laxo), o mitad con umbral reducido.
                        if name_r is not None and s_r >= umbral_rango + 0.05:
                            cid = str_a_carta_id(name_r)
                            metodo = "rango"
                        elif win_m is not None and s_m >= umbral_mitad - 0.20:
                            cid = str_a_carta_id(name_m)
                            metodo = "mitad"
            else:
                # ── CARTA COMPLETA (poco solapada o ultima del bloque) ──
                x0_c = max(0, cx - int(0.05 * cardw))
                x1_c = min(cx + min(cardw, visible + int(0.10 * cardw)),
                           blob.shape[1])
                if x1_c > x0_c:
                    win_c = blob[0:int(0.95 * fh), x0_c:x1_c]
                    s_c, name_c, _ = reconocedor.buscar_carta(
                        win_c, fh, palo=(palo_blk or ""), escalas=escalas_blk)
                    if name_c is not None and s_c >= umbral_carta:
                        cid = str_a_carta_id(name_c)
                        metodo = "carta"
                # fallback: rango + palo del bloque
                if cid is None:
                    x0_r = max(0, cx - int(0.12 * cardw))
                    x1_r = min(cx + int(0.40 * cardw), blob.shape[1])
                    win_r = blob[0:int(0.52 * fh), x0_r:x1_r]
                    s_r, name_r, _ = reconocedor.buscar_rango(
                        win_r, fh, escalas=escalas_blk)
                    if (name_r is not None and s_r >= umbral_rango
                            and palo_blk is not None):
                        cid = str_a_carta_id(name_r[:-1] + palo_blk)
                        metodo = "rango"
            out.append(CartaMano(cid, (tx, ty), metodo,
                                 card_h=fh, visible_w=visible))
            # ── salida temprana: ya localizamos la carta buscada ──
            if buscar_id is not None and cid == buscar_id:
                return [out[-1]]
    return [] if buscar_id is not None else out


def leer_mano_posiciones(img: np.ndarray, regiones: Regiones, reconocedor,
                         umbral_rango: float = 0.45, umbral_palo: float = 0.45,
                         umbral_mitad: float = 0.55, umbral_carta: float = 0.50
                         ) -> List[CartaMano]:
    """Lee TODA la mano y devuelve el PUNTO de toque de cada carta en
    coordenadas del screenshot completo (para auto-juego por ADB). Ver
    `_leer_mano_filtrada` para la estrategia de reconocimiento."""
    return _leer_mano_filtrada(img, regiones, reconocedor,
                               umbral_rango, umbral_palo,
                               umbral_mitad, umbral_carta)


def localizar_carta(img: np.ndarray, regiones: Regiones, reconocedor,
                    carta_id: int,
                    umbral_rango: float = 0.45, umbral_palo: float = 0.45,
                    umbral_mitad: float = 0.55, umbral_carta: float = 0.50
                    ) -> Optional[CartaMano]:
    """Localiza UNA carta concreta en la mano leyendo SOLO el bloque de su palo
    (el resto de bloques se detectan pero no se reconocen carta a carta). Mucho
    más rápido que `leer_mano_posiciones` cuando solo se necesita una carta
    (p.ej. la siguiente a tocar en el pase), sin perder precisión: la carta se
    sigue verificando por plantilla. Devuelve la `CartaMano` o None si no la
    encuentra con confianza."""
    res = _leer_mano_filtrada(img, regiones, reconocedor,
                              umbral_rango, umbral_palo,
                              umbral_mitad, umbral_carta,
                              solo_palo=None, buscar_id=carta_id)
    return res[0] if res else None


def leer_mano(img: np.ndarray, regiones: Regiones, reconocedor,
              umbral_rango: float = 0.45, umbral_palo: float = 0.45,
              umbral_mitad: float = 0.55, umbral_carta: float = 0.50
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
            leer_mano_posiciones(img, regiones, reconocedor,
                                 umbral_rango, umbral_palo,
                                 umbral_mitad, umbral_carta)]


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
        score, name, _ = reconocedor.buscar_carta(card, ch)
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
        score, name, _ = reconocedor.buscar_carta(card, ch)
        if name is not None and score >= reconocedor.umbral:
            out.append(str_a_carta_id(name))
        else:
            out.append(None)
    return out


__all__ = [
    "POSICIONES", "Regiones",
    # Re-exportados desde banner.py (compatibilidad hacia atras)
    "BannerClasificador", "BannerClasificadorTexto",
    "IBannerClasificador", "ResultadoBanner",
    "capturar_banner",
    # Propios
    "EstadoVisual", "leer_estado", "leer_mesa", "leer_mano",
    "CartaMano", "leer_mano_posiciones", "localizar_carta", "leer_pases",
]
