"""
Auto-pase por ADB: ejecuta la FASE DE PASE tocando la pantalla con el modelo.

Este es el primer "actuador" del proyecto (hasta ahora la visión SOLO observaba).
Cierra el lazo: ve la mano (visión) -> pide al modelo qué 3 cartas pasar ->
las toca por ADB -> confirma -> lee qué cartas recibió.

El reto de esta app: al tocar una carta hay una animación y los bloques de la
mano (agrupados por palo) se REORDENAN. Por eso NO se cachean posiciones: antes
de cada toque se vuelve a leer la mano y se localiza la carta por su id. Así el
movimiento de los bloques es irrelevante.

Responsabilidad (SRP): solo CONDUCE el pase. La visión vive en `vision_hearts`,
el plumbing ADB en `adb.py`, y la decisión la inyecta el llamante como callback
(`recomendar`), normalmente envolviendo `scripts/recomendador.Recomendador`.

Lo que SÍ necesita calibrado para funcionar de punta a punta:
  - `regiones.confirmar` (caja del botón círculo-check), o una plantilla
    `confirmar.png` para localizarlo por correlación. Sin eso, deja las 3 cartas
    seleccionadas y avisa para que confirmes a mano.
Lo que NO necesita: una región especial para las cartas recibidas. Se deducen
releyendo la mano cuando el pase se resuelve (recibidas = mano_nueva − (mano_vieja
− pasadas)).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import cv2

from src.captura.adb import ClienteADB
from src.captura.vision_hearts import (BannerClasificador, CartaMano, Regiones,
                                       leer_mano, leer_mano_posiciones,
                                       leer_pases)
from src.captura.vision_cartas import localizar_cartas

# Callback de decisión: (mano_ids, direccion) -> 3 ids a pasar.
Recomendar = Callable[[List[int], str], List[int]]


@dataclass
class ConfigAutoPase:
    poll_s: float = 0.04           # espera entre capturas en bucles de sondeo
    settle_s: float = 0.05         # fallback si _esperar_estabilidad no converge
    intentos_lectura: int = 2      # reintentos para leer la mano completa (13)
    intentos_carta: int = 5        # reintentos para localizar+tocar una carta
    intentos_recibidas: int = 15   # sondeos esperando a que el pase se resuelva
    umbral_confirmar: float = 0.30  # correlación mínima de la plantilla del botón
    # ── estabilidad visual ──
    umbral_cambio: float = 0.008   # fracción de píxeles cambiados para "estable"
    intentos_estabilidad: int = 3  # reintentos esperando estabilidad
    # ── debug ──
    # si no es None, guarda screenshots de cada paso
    debug_dir: Optional[str] = None


@dataclass
class ResultadoPase:
    direccion: str
    pasadas: List[int] = field(default_factory=list)
    recibidas: List[int] = field(default_factory=list)
    confirmado: bool = False
    nota: str = ""


class ControladorPase:
    """Conduce la fase de pase tocando la pantalla por ADB."""

    def __init__(
        self,
        cliente: ClienteADB,
        regiones: Regiones,
        banner_clf: BannerClasificador,
        reconocedor_mano,
        recomendar: Recomendar,
        config: Optional[ConfigAutoPase] = None,
        plantilla_confirmar: Optional[str | Path] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cli = cliente
        self.reg = regiones
        self.clf = banner_clf
        self.rec = reconocedor_mano
        self.recomendar = recomendar
        self.cfg = config or ConfigAutoPase()
        self.log = log
        self._tpl_confirmar = (Path(plantilla_confirmar)
                               if plantilla_confirmar else None)
        self._debug_n = 0  # contador para screenshots de debug

    # ---- debug ------------------------------------------------------------

    def _debug_shot(self, etapa: str, punto: tuple[int, int] | None = None,
                    img: Optional[np.ndarray] = None) -> None:
        """Guarda un screenshot de debug.

        Si se pasa `punto=(x, y)`, dibuja una cruz roja y un círculo en ese
        punto ANTES de guardar, mostrando dónde se va a hacer el tap.
        """
        if not self.cfg.debug_dir:
            return
        import cv2 as _cv2
        shot = img.copy() if img is not None else self.cli.captura()
        if punto is not None:
            px, py = punto
            # ── cruz roja de 20px ──
            _cv2.line(shot, (px - 20, py), (px + 20, py), (0, 0, 255), 3)
            _cv2.line(shot, (px, py - 20), (px, py + 20), (0, 0, 255), 3)
            # ── círculo de mira ──
            _cv2.circle(shot, (px, py), 25, (0, 0, 255), 2)
            _cv2.circle(shot, (px, py), 5, (0, 0, 255), -1)
        name = f"{self.cfg.debug_dir}/debug_{self._debug_n:02d}_{etapa}.png"
        Path(self.cfg.debug_dir).mkdir(parents=True, exist_ok=True)
        _cv2.imwrite(name, shot)
        self._debug_n += 1

    # ---- estabilidad visual (animación terminada) ------------------------

    def _esperar_estabilidad(self, region_frac=None) -> bool:
        """Espera hasta que la región de la MANO deje de cambiar visualmente.

        Compara frames consecutivos de la zona de cartas del agente; cuando la
        fracción de píxeles que cambian cae por debajo de `umbral_cambio`,
        asume que la animación terminó. Si no se pasa región, usa self.reg.mano.
        """
        reg = region_frac or self.reg.mano
        prev_gray = None
        for i in range(self.cfg.intentos_estabilidad):
            time.sleep(self.cfg.poll_s)
            img = self.cli.captura()
            H, W = img.shape[:2]
            x0 = int(reg[0] * W)
            y0 = int(reg[1] * H)
            x1 = int((reg[0] + reg[2]) * W)
            y1 = int((reg[1] + reg[3]) * H)
            gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(np.float32)
                                      - prev_gray.astype(np.float32))) / 255.0
                if diff < self.cfg.umbral_cambio:
                    return True
            prev_gray = gray
        return False

    # ---- lectura ----------------------------------------------------------

    def _fase_pase(self, img: np.ndarray) -> Optional[str]:
        """Devuelve la dirección del pase si el banner es de pase, si no None."""
        r = self.clf.clasificar(img, self.reg)
        return r.dato if r.categoria == "pase" else None

    def _leer_mano_completa(self) -> tuple[np.ndarray, List[CartaMano]]:
        """Captura y lee la mano; reintenta hasta tener 13 cartas reconocidas."""
        mejor_img, mejor = None, []
        for _ in range(self.cfg.intentos_lectura):
            img = self.cli.captura()
            cartas = leer_mano_posiciones(img, self.reg, self.rec)
            ok = [c for c in cartas if c.carta_id is not None]
            if len(ok) > len(mejor):
                mejor_img, mejor = img, cartas
            if sum(1 for c in cartas if c.carta_id is not None) >= 13:
                return img, cartas
            time.sleep(self.cfg.poll_s)
        return mejor_img, mejor

    # ---- selección de una carta (con relectura por el reordenamiento) -----

    def _seleccionar(self, carta_id: int,
                     pos_cache: Optional[dict[int, tuple[int, int]]] = None
                     ) -> bool:
        """Localiza la carta por id y la toca.

        Si `pos_cache` contiene la carta, usa la posición directamente sin
        re-leer (la mano NO se reordena entre taps de selección — solo se
        iluminan las cartas elegidas). Si no, usa el pipeline rápido."""
        # ── cache: la mano no se refluye entre taps de selección ──
        if pos_cache is not None and carta_id in pos_cache:
            x, y = pos_cache[carta_id]
            self.cli.tap(int(x), int(y))
            time.sleep(0.08)
            return True

        # ── búsqueda rápida ──
        img = self.cli.captura()
        punto = self._localizar_carta_rapida(img, carta_id)
        if punto is not None:
            self._debug_shot(f"tap_{carta_id}", punto=punto, img=img)
            self.cli.tap(*punto)
            time.sleep(0.08)
            return True

        # ── fallback: lectura completa ──
        cartas = leer_mano_posiciones(img, self.reg, self.rec)
        objetivo = next((c for c in cartas if c.carta_id == carta_id), None)
        if objetivo is not None:
            x, y = objetivo.centro
            self._debug_shot(f"tap_fb_{carta_id}",
                             punto=(int(x), int(y)), img=img)
            self.cli.tap(x, y)
            time.sleep(0.08)
            return True
        return False

    # ---- búsqueda rápida de UNA carta (evita leer toda la mano) -----------

    # Umbral mínimo de correlación para aceptar un match de rango.
    _UMBRAL_RANGO = 0.50
    # Factor ancho/alto de carta (sprite APK ≈ 168/217).
    _ANCHO_CARTA_FRAC = 0.774

    def _localizar_carta_rapida(self, img: np.ndarray, carta_id: int
                                ) -> Optional[tuple[int, int]]:
        """Busca UNA carta en la mano con `buscar_rango` en cada caja.

        Recorre TODAS las cajas y devuelve la de MAYOR score que coincida
        con el rango buscado (no la primera), para evitar falsos positivos
        cuando dos cartas comparten rango (ej. JP vs JC)."""
        from src.captura.modelos import carta_a_str

        target_rank = carta_a_str(carta_id)[:-1]

        H, W = img.shape[:2]
        mx, my = self.reg.mano[0], self.reg.mano[1]
        mw, mh = self.reg.mano[2], self.reg.mano[3]
        x0, y0 = int(mx * W), int(my * H)
        x1, y1 = int((mx + mw) * W), int((my + mh) * H)
        roi = img[y0:y1, x0:x1]

        best = None   # (score, centro_x, centro_y)

        cajas = localizar_cartas(roi)
        for cx, cy, cw, ch in cajas:
            cardw = int(self._ANCHO_CARTA_FRAC * ch)
            wx0 = max(0, cx - int(0.12 * cardw))
            wx1 = min(cx + int(0.40 * cardw), roi.shape[1])
            wy0 = cy
            wy1 = cy + int(0.52 * ch)
            if wy1 > roi.shape[0] or wx0 >= wx1:
                continue
            win = roi[wy0:wy1, wx0:wx1]
            if win.size == 0:
                continue

            score, encontrada = self.rec.buscar_rango(win, int(ch))
            if encontrada is None or score < self._UMBRAL_RANGO:
                continue
            if encontrada[:-1] != target_rank:
                continue
            if best is None or score > best[0]:
                centro_x = x0 + cx + cw // 2
                centro_y = y0 + cy + ch // 2
                best = (score, centro_x, centro_y)

        if best is not None:
            return (best[1], best[2])
        return None

    # ---- verificación en zona de pases -----------------------------------

    def _verificar_pases(self, esperadas: List[int]) -> bool:
        """Lee la zona de pases (cartas seleccionadas) y verifica que
        coincidan con las que el modelo recomendó pasar.

        La zona de pases muestra las 3 cartas seleccionadas (antes de
        confirmar). Las lee con `leer_pases` (matchTemplate contra sprites
        completos de la APK) y comprueba que el conjunto coincida.
        """
        from src.captura.modelos import carta_a_str
        time.sleep(0.15)  # margen para animación de selección
        for intento in range(3):
            img = self.cli.captura()
            pases = [p for p in leer_pases(
                img, self.reg, self.rec) if p is not None]
            if len(pases) >= 3:
                break
            time.sleep(0.2)
        else:
            # Último intento
            img = self.cli.captura()
            pases = [p for p in leer_pases(
                img, self.reg, self.rec) if p is not None]

        if sorted(pases) == sorted(esperadas):
            self.log(f"  ✅ pases verificados: "
                     + " ".join(carta_a_str(c) for c in pases))
            return True
        else:
            self.log(f"  ⚠ pases en zona NO coinciden: leídos="
                     + " ".join(carta_a_str(c) for c in sorted(pases))
                     + "  esperados="
                     + " ".join(carta_a_str(c) for c in sorted(esperadas)))
            return False

    # ---- verificación post-pase -------------------------------------------

    def _verificar_cartas_pasadas(self, mano_original: List[int],
                                  pasadas: List[int]) -> None:
        """Re-lee la mano tras confirmar y verifica que las cartas pasadas
        ya NO estén. Emite alerta si alguna sigue presente (posible tap
        fallido en la carta equivocada)."""
        from src.captura.modelos import carta_a_str
        time.sleep(0.5)  # margen para la animación de pase
        for _ in range(3):
            img = self.cli.captura()
            actual = [c for c in leer_mano(
                img, self.reg, self.rec) if c is not None]
            if len(actual) >= 13:
                break
            time.sleep(0.3)
        sobrantes = set(pasadas) & set(actual)
        if sobrantes:
            self.log(f"  ⚠ ALERTA: estas cartas debían haberse ido pero siguen: "
                     + " ".join(carta_a_str(c) for c in sorted(sobrantes)))
        else:
            self.log(
                "  ✅ verificación: las 3 cartas pasadas ya no están en la mano")

    # ---- confirmar (botón círculo-check) ----------------------------------

    def _confirmar(self) -> bool:
        """Toca el botón de confirmar.

        Estrategia:
        1. Breve pausa para que las animaciones de selección terminen.
        2. Localiza el botón por plantilla (check blanco + círculo amarillo).
        3. Si falla, usa el centro de la región calibrada.
        4. Guarda debug shot con la cruz en el punto ANTES de tocar."""
        # ── breve pausa para animaciones ──
        time.sleep(0.15)

        # ── intentar localizar por plantilla (color + canny) ──
        t_loc = time.perf_counter()
        punto = None
        if self._tpl_confirmar and self._tpl_confirmar.is_file():
            punto = self._localizar_confirmar()
            if punto is not None:
                self.log(
                    f"  🔘 confirmar: localizado en ({punto[0]}, {punto[1]})  [{time.perf_counter()-t_loc:.2f}s]")
            else:
                self.log(
                    f"  ⚠ plantilla no localizada, usando centro de región  [{time.perf_counter()-t_loc:.2f}s]")

        # ── resolver punto de tap ──
        img = None
        if punto is None and self.reg.confirmar is not None:
            img = self.cli.captura()
            H, W = img.shape[:2]
            x, y, w, h = self.reg.confirmar
            cx, cy = int((x + w / 2) * W), int((y + h / 2) * H)
            self.log(f"  🔘 confirmar: centro región ({cx}, {cy})")
            punto = (cx, cy)

        if punto is None:
            return False

        # ── debug shot con la cruz en el punto de confirmación ──
        self._debug_shot("confirmar", punto=punto, img=img)

        # ── tap ──
        self.cli.tap(*punto)
        time.sleep(0.15)
        return True

    def _localizar_confirmar(self) -> Optional[tuple[int, int]]:
        """Localiza el botón de confirmar. Prueba dos plantillas en orden:
        1. confirmar_check.png (check blanco de key_OK, muy distintivo).
        2. confirmar.png (círculo amarillo round_button, fallback).
        Para cada una: color multiescala → bordes Canny si necesario.
        Si ninguna supera el umbral, devuelve None."""
        img = self.cli.captura()
        H, W = img.shape[:2]
        if self.reg.confirmar is not None:
            x, y, w, h = self.reg.confirmar
            x0, y0 = int(x * W), int(y * H)
            roi = img[y0:y0 + int(h * H), x0:x0 + int(w * W)]
        else:
            x0, y0, roi = 0, 0, img

        # ── plantillas a probar en orden ──
        tpl_paths = []
        for p in [self._tpl_confirmar,
                  self._tpl_confirmar.parent / "confirmar_check.png"
                  if self._tpl_confirmar else None]:
            if p and p.is_file():
                tpl_paths.append(p)

        mejor_score, mejor_centro = 0.0, None
        for tpl_path in tpl_paths:
            tpl = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
            if tpl is None:
                continue
            score, centro = self._match_plantilla(roi, tpl, x0, y0)
            if score > mejor_score:
                mejor_score = score
                mejor_centro = centro
            if mejor_score >= self.cfg.umbral_confirmar:
                break  # ya tenemos match bueno

        # Umbral de "buen match" más alto que el mínimo: si el score es
        # marginal (<0.55) preferimos NO arriesgarnos a un falso positivo y
        # delegamos en el centro de la región.
        UMBRAL_CONFIABLE = 0.55
        self.log(
            f"  🔍 confirmar: score_max={mejor_score:.3f} (umbral_min={self.cfg.umbral_confirmar}, confiable={UMBRAL_CONFIABLE})")
        if mejor_score < UMBRAL_CONFIABLE:
            self.log(
                f"  ⚠ score bajo ({mejor_score:.3f}<{UMBRAL_CONFIABLE}) → usando centro de región")
            return None  # fuerza fallback al centro
        return mejor_centro

    def _match_plantilla(self, roi, tpl, x0, y0
                         ) -> tuple[float, Optional[tuple[int, int]]]:
        """Template matching multiescala (color + Canny fallback)."""
        mejor_score, mejor_centro = 0.0, None

        # ── color multiescala ──
        for escala in [0.7, 0.85, 1.0, 1.15]:
            tpl_r = cv2.resize(tpl, None, fx=escala, fy=escala)
            if tpl_r.shape[0] > roi.shape[0] or tpl_r.shape[1] > roi.shape[1]:
                continue
            res = cv2.matchTemplate(roi, tpl_r, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv > mejor_score:
                mejor_score = maxv
                mejor_centro = (x0 + maxloc[0] + tpl_r.shape[1] // 2,
                                y0 + maxloc[1] + tpl_r.shape[0] // 2)

        # ── bordes Canny ──
        if mejor_score < self.cfg.umbral_confirmar:
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tpl_edges = cv2.Canny(tpl_gray, 40, 120)
            roi_edges = cv2.Canny(roi_gray, 40, 120)
            for escala in [0.7, 0.85, 1.0, 1.15]:
                tpl_e = cv2.resize(tpl_edges, None, fx=escala, fy=escala)
                if tpl_e.shape[0] > roi_edges.shape[0] or tpl_e.shape[1] > roi_edges.shape[1]:
                    continue
                res = cv2.matchTemplate(roi_edges, tpl_e, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)
                if maxv > mejor_score:
                    mejor_score = maxv
                    mejor_centro = (x0 + maxloc[0] + tpl_e.shape[1] // 2,
                                    y0 + maxloc[1] + tpl_e.shape[0] // 2)

        return mejor_score, mejor_centro

    # ---- cartas recibidas (pases zone + fallback por diferencia) ----------

    def _esperar_recibidas(self, mano_antes: List[int], pasadas: List[int]
                           ) -> List[int]:
        """Espera a que el pase se resuelva y devuelve las cartas recibidas.

        Primero intenta leer la zona de pases con `leer_pases` (matchTemplate
        contra los sprites completos de la APK, más fiable que la diferencia).
        Si no se leen 3 cartas, cae al método por diferencia:
        recibidas = mano_nueva − (mano_antes − pasadas).
        """
        base = set(mano_antes) - set(pasadas)   # lo que conservo
        # La animación de pase tarda ~1.5-2s en resolverse en el dispositivo.
        time.sleep(2.0)

        # ── intentar leer de la zona de pases (más fiable) ──
        for _ in range(self.cfg.intentos_recibidas):
            img = self.cli.captura()
            pases = [p for p in leer_pases(
                img, self.reg, self.rec) if p is not None]
            if len(pases) >= 3:
                from src.captura.modelos import carta_a_str
                self.log(f"  📥 recibidas por pases: "
                         + " ".join(carta_a_str(c) for c in sorted(pases[:3])))
                return pases[:3]
            time.sleep(self.cfg.poll_s)

        # ── fallback: diferencia de mano ──
        self.log("  ⚠ no se leyeron las recibidas del pases; usando diferencia")
        for _ in range(self.cfg.intentos_recibidas):
            img = self.cli.captura()
            mano = [c for c in leer_mano(
                img, self.reg, self.rec) if c is not None]
            recibidas = sorted(set(mano) - base)
            if len(mano) >= 13 and len(recibidas) >= 3:
                return recibidas[:3] if len(recibidas) > 3 else recibidas
            time.sleep(self.cfg.poll_s)
        # último intento
        img = self.cli.captura()
        mano = [c for c in leer_mano(img, self.reg, self.rec) if c is not None]
        return sorted(set(mano) - base)

    # ---- orquestación -----------------------------------------------------

    def ejecutar(self) -> ResultadoPase:
        t0 = time.perf_counter()
        img = self.cli.captura()
        self._debug_shot("inicio", img=img)
        direccion = self._fase_pase(img)
        if direccion is None:
            return ResultadoPase(direccion="?", nota="No es fase de pase (banner).")
        self.log(
            f"Fase de pase detectada → dirección: {direccion}  [{time.perf_counter()-t0:.2f}s]")

        t_lec = time.perf_counter()
        _, cartas = self._leer_mano_completa()
        mano_ids = [c.carta_id for c in cartas if c.carta_id is not None]
        self.log(
            f"Mano leída: {len(mano_ids)}/13 cartas  [{time.perf_counter()-t_lec:.2f}s]")
        if len(mano_ids) < 13:
            return ResultadoPase(direccion=direccion,
                                 nota=f"Solo leí {len(mano_ids)}/13 cartas; no toco nada.")

        # Cache de posiciones: las cartas NO se reordenan entre taps de
        # selección — solo se iluminan. Reutilizamos las posiciones de la
        # lectura inicial para evitar 1-2 capturas+vision extra.
        pos_cache = {c.carta_id: (int(c.centro[0]), int(c.centro[1]))
                     for c in cartas if c.carta_id is not None}

        a_pasar = list(self.recomendar(mano_ids, direccion))[:3]
        from src.captura.modelos import carta_a_str
        self.log("Modelo recomienda PASAR: "
                 + "  ".join(carta_a_str(c) for c in a_pasar))

        t_sel = time.perf_counter()
        seleccionadas: List[int] = []
        for cid in a_pasar:
            if self._seleccionar(cid, pos_cache):
                seleccionadas.append(cid)
                self.log(
                    f"  ✓ seleccionada {carta_a_str(cid)}  [{time.perf_counter()-t_sel:.2f}s]")
            else:
                self.log(
                    f"  ✗ no pude localizar {carta_a_str(cid)} para tocarla")
        self.log(
            f"Selección completada (3 cartas)  [{time.perf_counter()-t_sel:.2f}s]")

        res = ResultadoPase(direccion=direccion, pasadas=seleccionadas)
        if len(seleccionadas) < 3:
            res.nota = "No seleccioné las 3 cartas; no confirmo."
            return res

        # ── verificación en zona de pases (contra sprites de la APK) ──
        self._verificar_pases(seleccionadas)

        t_conf = time.perf_counter()
        res.confirmado = self._confirmar()
        if not res.confirmado:
            res.nota = ("3 cartas seleccionadas, pero no hay botón de confirmar "
                        "calibrado (regiones.confirmar o plantilla). Confirma a mano.")
            return res
        self.log(f"  ✓ pase confirmado  [{time.perf_counter()-t_conf:.2f}s]")

        # ── verificación: ¿desaparecieron las 3 cartas que debíamos pasar? ──
        self._verificar_cartas_pasadas(mano_ids, seleccionadas)

        t_rec = time.perf_counter()
        res.recibidas = self._esperar_recibidas(mano_ids, seleccionadas)
        self._debug_shot("recibidas")
        self.log(f"Cartas RECIBIDAS: "
                 + ("  ".join(carta_a_str(c) for c in res.recibidas)
                    if res.recibidas else "(no detectadas)")
                 + f"  [{time.perf_counter()-t_rec:.2f}s]")
        self.log(f"⏱ TOTAL: {time.perf_counter()-t0:.2f}s")
        return res


__all__ = ["ConfigAutoPase", "ResultadoPase", "ControladorPase", "Recomendar"]
