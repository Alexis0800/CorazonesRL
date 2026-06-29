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
from typing import Callable, List, Optional, Tuple

import numpy as np

import cv2

from src.captura.adb import ClienteADB
from src.captura.vision_hearts import (BannerClasificador, CartaMano, Regiones,
                                       leer_mano_posiciones,
                                       leer_pases)
from src.captura.vision_cartas import localizar_cartas

# Callback de decisión: (mano_ids, direccion) -> 3 ids a pasar.
Recomendar = Callable[[List[int], str], List[int]]


@dataclass
class ConfigAutoPase:
    poll_s: float = 0.04           # espera entre capturas en bucles de sondeo
    settle_s: float = 0.05         # fallback si _esperar_estabilidad no converge
    intentos_lectura: int = 3      # reintentos para leer la mano completa (13)
    intentos_carta: int = 5        # reintentos para localizar+tocar una carta
    intentos_recibidas: int = 15   # sondeos esperando a que el pase se resuelva
    intentos_correccion: int = 3   # reintentos de corrección si el pase no cuadra
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
        """Devuelve la direccion del pase si el banner es de pase, si no None."""
        r, distancias = self.clf.clasificar_verbose_con_regiones(img, self.reg)
        top_str = "  ".join(f"{t}={d:.2f}" for t, d in distancias)
        self.log(f"  📊 banner: {r.tag} d={r.distancia:.2f} → {r.categoria}/{r.dato}  |  top: {top_str}")

        # ── guardar recorte del banner para crear plantillas del dispositivo ──
        self._guardar_banner_device(img)

        return r.dato if r.categoria == "pase" else None

    def _guardar_banner_device(self, img: np.ndarray) -> None:
        """Guarda un recorte del banner en debug_dir/banners_device/ para
        crear plantillas especificas de este dispositivo. Solo en modo debug."""
        if not self.cfg.debug_dir:
            return
        from src.captura.banner import capturar_banner
        capturar_banner(img, self.reg, Path(self.cfg.debug_dir) / "banners_device")

    def _leer_mano_completa(self) -> tuple[np.ndarray, List[CartaMano]]:
        """Captura y lee la mano; reintenta hasta tener 13 cartas reconocidas."""
        from src.captura.modelos import carta_a_str
        mejor_img, mejor = None, []
        for intento in range(self.cfg.intentos_lectura):
            img = self.cli.captura()
            cartas = leer_mano_posiciones(img, self.reg, self.rec)
            ok = [c for c in cartas if c.carta_id is not None]
            if len(ok) > len(mejor):
                mejor_img, mejor = img, cartas
            if len(ok) >= 13:
                self._debug_mano_overlay(img, cartas,
                                         f"leer_mano_{len(ok):02d}ok")
                return img, cartas
            time.sleep(self.cfg.poll_s)
        # ── último intento: guardar overlay de debug aunque no sean 13 ──
        if mejor_img is not None and mejor:
            self._debug_mano_overlay(mejor_img, mejor,
                                     f"leer_mano_{len([c for c in mejor if c.carta_id is not None]):02d}ok_mejor")
        return mejor_img, mejor

    def _debug_mano_overlay(self, img: np.ndarray, cartas: List[CartaMano],
                            etapa: str) -> None:
        """Guarda un screenshot de debug con TODAS las cartas anotadas:
        círculo verde + ID si reconocida, círculo rojo + '??' si no."""
        if not self.cfg.debug_dir:
            return
        from src.captura.modelos import carta_a_str
        out = img.copy()
        for i, c in enumerate(cartas):
            x, y = int(c.centro[0]), int(c.centro[1])
            if c.carta_id is not None:
                label = f"{i}:{carta_a_str(c.carta_id)}"
                color = (0, 255, 0)  # verde
                cv2.circle(out, (x, y), 18, color, 2)
            else:
                label = f"{i}:??"
                color = (0, 0, 255)  # rojo
                cv2.circle(out, (x, y), 18, color, 2)
            # método en letra chica
            if c.metodo:
                label += f"({c.metodo[0]})"
            cv2.putText(out, label, (x - 30, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
        name = f"{self.cfg.debug_dir}/debug_{self._debug_n:02d}_{etapa}.png"
        Path(self.cfg.debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(name, out)
        self._debug_n += 1

    # ---- selección de una carta (siempre con captura fresca) -------------

    def _seleccionar(self, carta_id: int) -> bool:
        """Localiza la carta por id en una captura FRESCA y la toca.

        NUNCA cachea posiciones: los bloques de la mano se reordenan al tocar
        una carta. Cada llamada captura pantalla y busca la carta de nuevo."""
        # ── búsqueda rápida (rango, la más veloz) ──
        img = self.cli.captura()
        punto = self._localizar_carta_rapida(img, carta_id)
        if punto is not None:
            self._debug_shot(f"tap_{carta_id}", punto=punto, img=img)
            self.cli.tap(*punto)
            return True

        # ── fallback: lectura completa ──
        cartas = leer_mano_posiciones(img, self.reg, self.rec)
        objetivo = next((c for c in cartas if c.carta_id == carta_id), None)
        if objetivo is not None:
            x, y = objetivo.centro
            self._debug_shot(f"tap_fb_{carta_id}",
                             punto=(int(x), int(y)), img=img)
            self.cli.tap(x, y)
            return True
        return False

    # ---- búsqueda rápida de UNA carta (usa bloques de palo, sin buscar_mitad) -

    # Umbral mínimo de correlación para aceptar un match de rango.
    _UMBRAL_RANGO = 0.50
    # Umbral para aceptar un match de rango filtrado (solo 4 tpl, score más bajo).
    _UMBRAL_RANGO_FILTRADO = 0.42
    # Factor ancho/alto de carta (sprite APK ≈ 168/217).
    _ANCHO_CARTA_FRAC = 0.774
    # Umbral mínimo para determinar el palo de un bloque desde su última carta.
    _UMBRAL_PALO_BLOQUE = 0.45

    def _localizar_carta_rapida(self, img: np.ndarray, carta_id: int
                                ) -> Optional[tuple[int, int]]:
        """Busca UNA carta en la mano usando BLOQUES DE PALO (sin template matching
        de mitad, que falla en cartas solapadas).

        Estrategia (igual que `leer_mano_posiciones` pero sin leer TODAS las cartas):
        1. Agrupar cartas en BLOQUES (filas) con `_filas_de_cartas`.
        2. Para cada bloque, determinar el PALO desde la carta MÁS A LA DERECHA
           (la única visible entera; probamos carta completa > mitad > rango).
        3. Buscar el RANGO de la carta objetivo dentro de cada bloque con
           `buscar_rango_por_rank` (solo 4 tpl × 10 esc = 40 matchTemplate
           por posición). Si el rango coincide Y el palo del bloque es el
           esperado → encontrada.

        Si el palo del bloque no coincide con el esperado, devolvemos None para
        que `_seleccionar` use el fallback `leer_mano_posiciones` (más lento
        pero 100% fiable)."""
        from src.captura.modelos import carta_a_str

        target_full = carta_a_str(carta_id)   # "QP"
        rank = target_full[:-1]               # "Q"
        expected_suit = target_full[-1]       # "P"

        H, W = img.shape[:2]
        mx, my = self.reg.mano[0], self.reg.mano[1]
        mw, mh = self.reg.mano[2], self.reg.mano[3]
        x0, y0 = int(mx * W), int(my * H)
        mano_roi = img[int(my * H):int((my + mh) * H),
                        int(mx * W):int((mx + mw) * W)]

        from src.captura.vision_cartas import _filas_de_cartas

        filas = sorted(_filas_de_cartas(mano_roi, 0.01),
                       key=lambda b: (b[1] // 50, b[0]))
        if not filas:
            return None

        for (fx, fy, fw, fh) in filas:
            blob = mano_roi[fy:fy + fh, fx:fx + fw]
            cajas = localizar_cartas(blob, 0.01)
            if not cajas:
                continue
            cardw = int(self._ANCHO_CARTA_FRAC * fh)
            xs = [c[0] for c in cajas]
            n = len(xs)
            pos = ([int(xs[0] + (xs[-1] - xs[0]) * i / (n - 1))
                    for i in range(n)] if n > 1 else xs)

            # ── determinar PALO del bloque (carta más a la derecha) ──
            bx = pos[-1]
            palo_blk: Optional[str] = None
            blk_x1 = min(bx + int(1.20 * cardw), blob.shape[1])

            # 1) carta completa
            s_c, name_c = self.rec.buscar_carta(
                blob[0:fh, bx:blk_x1], fh)
            if name_c and s_c >= self._UMBRAL_PALO_BLOQUE:
                palo_blk = name_c[-1]

            # 2) mitad (si sigue sin palo)
            if palo_blk is None:
                b_visible = min(cardw, fw - bx)
                if b_visible < 0.85 * cardw:
                    b_x1_m = min(bx + int(0.48 * cardw), blob.shape[1])
                    b_win_m = blob[0:int(0.95 * fh), bx:b_x1_m]
                    s_bm, name_bm = self.rec.buscar_mitad(b_win_m, fh)
                    if name_bm and s_bm >= self._UMBRAL_PALO_BLOQUE:
                        palo_blk = name_bm[-1]

            # 3) rango (último recurso)
            if palo_blk is None:
                b_x1_r = min(bx + int(0.40 * cardw), blob.shape[1])
                b_win_r = blob[0:int(0.52 * fh), bx:b_x1_r]
                s_br, name_br = self.rec.buscar_rango(b_win_r, fh)
                if name_br and s_br >= self._UMBRAL_PALO_BLOQUE + 0.15:
                    palo_blk = name_br[-1]

            if palo_blk is None:
                continue  # no pudimos determinar el palo → siguiente bloque

            # ── buscar el RANGO dentro de este bloque ──
            for i, cx in enumerate(pos):
                visible = (pos[i + 1] - cx) if i + 1 < n else min(cardw,
                                                                   fw - cx)
                wx0 = max(0, cx - int(0.12 * cardw))
                wx1 = min(cx + min(int(0.40 * cardw),
                                   visible + int(0.10 * cardw)),
                          blob.shape[1])
                wy0 = max(0, -int(0.05 * fh))
                wy1 = min(int(0.52 * fh), blob.shape[0])
                if wy1 <= 0 or wx0 >= wx1:
                    continue
                win = blob[wy0:wy1, wx0:wx1]
                if win.size == 0:
                    continue

                score, encontrada = self.rec.buscar_rango_por_rank(
                    win, fh, rank)
                if encontrada is None or score < self._UMBRAL_RANGO_FILTRADO:
                    continue

                # ── ¿el palo del bloque es el esperado? ──
                if palo_blk == expected_suit:
                    # Punto de toque: centro de la franja VISIBLE
                    tx = (x0 + fx + cx + max(1, visible) // 2)
                    ty = y0 + fy + fh // 2
                    return (tx, ty)
                else:
                    # El palo del bloque NO es el esperado → usar fallback
                    self.log(
                        f"  ⚠ bloque {palo_blk} ≠ {expected_suit} "
                        f"(buscaba {target_full}) → usando fallback")
                    return None

        return None

    # ---- verificación en zona de pases -----------------------------------

    def _verificar_pases(self, esperadas: List[int]) -> bool:
        """Lee la zona de pases y verifica que coincidan con esperadas.
        Devuelve True solo si las 3 coinciden exactamente."""
        from src.captura.modelos import carta_a_str
        time.sleep(0.2)
        pases: List[int] = []
        for intento in range(4):
            img = self.cli.captura()
            pases = [p for p in leer_pases(
                img, self.reg, self.rec) if p is not None]
            self._debug_shot(f"zona_pases_{intento}", img=img)
            if len(pases) >= 3:
                break
            time.sleep(0.25)

        ok = sorted(pases) == sorted(esperadas)
        emoji = "✅" if ok else "❌"
        self.log(f"  {emoji} pases: leídos="
                 + "  ".join(carta_a_str(c) for c in sorted(pases))
                 + "  |  esperados="
                 + "  ".join(carta_a_str(c) for c in sorted(esperadas)))
        return ok

    # ---- corrección automática del pase ----------------------------------

    def _leer_pases_posiciones(self, img: np.ndarray
                               ) -> List[tuple[int, tuple[int, int]]]:
        """Lee la zona de pases y devuelve [(carta_id, (x_centro, y_centro)), ...]
        en coordenadas del screenshot completo. Las cartas en el pase están
        completas y visibles → matchTemplate del sprite de la APK es muy fiable."""
        if self.reg.pases is None:
            return []
        from src.captura.modelos import str_a_carta_id
        from src.captura.vision_cartas import detectar_cartas

        roi = Regiones.recortar(img, self.reg.pases)
        H, W = img.shape[:2]
        px0 = int(self.reg.pases[0] * W)
        py0 = int(self.reg.pases[1] * H)
        cartas = detectar_cartas(roi, min_area_frac=0.02)
        out: List[tuple[int, tuple[int, int]]] = []
        for cx, cy, cw, ch in sorted(cartas, key=lambda b: b[0]):
            card = roi[cy:cy + ch, cx:cx + cw]
            score, name = self.rec.buscar_carta(card, ch)
            if name is not None and score >= self.rec.umbral:
                cid = str_a_carta_id(name)
                centro = (px0 + cx + cw // 2, py0 + cy + ch // 2)
                out.append((cid, centro))
        return out

    def _corregir_pase(self, esperadas: List[int],
                       max_intentos: int = 3) -> bool:
        """Corrige la selección si la zona de pases no coincide con lo esperado.

        1. Lee la zona de pases.
        2. Las cartas que están en pases pero NO en esperadas → las toca DOS VECES
           en la zona de pases para DE-SELECCIONARLAS (a veces un solo tap no basta
           en ciertas apps/versiones; el doble tap es más fiable).
        3. Espera estabilidad visual (bloques se reordenan).
        4. Las cartas que están en esperadas pero NO en pases → las localiza
           en la mano y las toca para SELECCIONARLAS.
        5. Repite hasta que coincidan o se agoten los intentos.

        Devuelve True si al final la zona de pases coincide con esperadas.
        """
        from src.captura.modelos import carta_a_str

        for intento in range(max_intentos):
            time.sleep(0.25)  # margen para animaciones
            # ── leer qué hay AHORA en la zona de pases ──
            img = self.cli.captura()
            pases_actuales = self._leer_pases_posiciones(img)
            ids_en_pases = {cid for cid, _ in pases_actuales}
            esperadas_set = set(esperadas)

            if ids_en_pases == esperadas_set:
                self.log(f"  ✅ pase corregido (intento {intento+1}): "
                         + " ".join(carta_a_str(c)
                                    for c in sorted(ids_en_pases)))
                return True

            # ── de-seleccionar cartas que NO deberían estar ──
            # Doble tap: un solo tap a veces no basta para deseleccionar
            sobrantes = ids_en_pases - esperadas_set
            for cid, (cx, cy) in pases_actuales:
                if cid in sobrantes:
                    self.log(
                        f"  ↩ de-seleccionando {carta_a_str(cid)} (no debe pasar)")
                    self._debug_shot(f"corregir_desel_{carta_a_str(cid)}",
                                     punto=(cx, cy), img=img)
                    # Doble tap con pausa entre taps
                    self.cli.tap(cx, cy)
                    time.sleep(0.10)
                    self.cli.tap(cx, cy)
                    time.sleep(0.15)

            if sobrantes:
                time.sleep(0.2)
                self._esperar_estabilidad()

            # ── seleccionar cartas que FALTAN ──
            faltantes = esperadas_set - ids_en_pases
            if faltantes:
                time.sleep(0.3)  # bloques asentándose
            for cid in sorted(faltantes):
                self.log(
                    f"  ↪ re-seleccionando {carta_a_str(cid)} (faltaba)")
                ok = self._seleccionar(cid)
                if ok:
                    time.sleep(0.25)
                    self._esperar_estabilidad()
                else:
                    self.log(
                        f"  ✗ no encontré {carta_a_str(cid)} para re-seleccionar")

        # ── verificación final ──
        time.sleep(0.25)
        img = self.cli.captura()
        self._debug_shot("corregir_final", img=img)
        pases_final = self._leer_pases_posiciones(img)
        ids_final = {cid for cid, _ in pases_final}
        return ids_final == esperadas_set

    # ---- cartas recibidas (post-confirmacion, zona de pases) -------------

    def _esperar_banner(self, categoria: str, timeout: float = 6.0
                        ) -> Optional[str]:
        """Espera hasta que el banner tenga la categoria indicada.
        Devuelve el tag del banner o None si se acaba el timeout.
        Al final (si falla), loguea las distancias de Hamming para diagnosticar."""
        t0 = time.perf_counter()
        ultimo_tag = "?"
        ultima_d: float = float("inf")
        while time.perf_counter() - t0 < timeout:
            try:
                img = self.cli.captura()
            except Exception:
                time.sleep(0.3)
                continue
            r, distancias = self.clf.clasificar_verbose_con_regiones(img, self.reg)
            ultimo_tag, ultima_d = r.tag, r.distancia
            if r.categoria == categoria:
                return r.tag
            time.sleep(self.cfg.poll_s)
        # ── no se detectó el banner esperado → diagnóstico ──
        self.log(f"  ⚠ _esperar_banner({categoria}): timeout tras {timeout}s")
        self.log(f"    último banner: {ultimo_tag} d={ultima_d:.2f}")
        # Releer una ultima vez para loggear el top-5 y guardar banner
        try:
            img = self.cli.captura()
            _, distancias = self.clf.clasificar_verbose_con_regiones(img, self.reg)
            top_str = "    top: " + "  ".join(f"{t}={d:.2f}" for t, d in distancias)
            self.log(top_str)
            self._debug_shot(f"banner_fail_{categoria}", img=img)
            self._guardar_banner_device(img)
        except Exception:
            pass
        return None

    def _esperar_recibidas(self, pasadas: List[int]) -> List[int]:
        """Espera el banner 'Cartas pasadas para ti' y lee las cartas
        recibidas de la zona de pases, donde aparecen completas (matchTemplate
        contra el sprite de la APK → muy fiable).

        Si no aparece el banner o no se leen 3, devuelve lista vacía.
        """
        from src.captura.modelos import carta_a_str

        tag = self._esperar_banner("pase_fin")
        if tag is None:
            self.log("  ⚠ no vi el banner 'Cartas pasadas para ti'")
            return []
        self.log(f"  📨 banner recibidas: {tag}")

        # Pequeña pausa para que terminen de renderizar las cartas
        time.sleep(0.3)

        recibidas: List[int] = []
        for _ in range(self.cfg.intentos_recibidas):
            try:
                img = self.cli.captura()
            except Exception:
                time.sleep(0.3)
                continue
            recibidas = [p for p in leer_pases(
                img, self.reg, self.rec) if p is not None]
            self._debug_shot("recibidas_zona", img=img)
            if len(recibidas) >= 3:
                break
            time.sleep(self.cfg.poll_s)

        if len(recibidas) >= 3:
            self.log(f"  📥 recibidas: "
                     + " ".join(carta_a_str(c) for c in recibidas[:3]))
            return recibidas[:3]
        else:
            self.log(f"  ⚠ solo leí {len(recibidas)}/3 cartas recibidas")
            return recibidas

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
        # ── Protección: si hay más de 13 posiciones, tomar solo las 13
        #    con ID reconocido más cercanas a la izquierda (orden natural). ──
        if len(mano_ids) > 13:
            self.log(
                f"⚠ {len(mano_ids)} posiciones detectadas → acotando a 13")
            # Conservar las 13 primeras con ID (ordenadas por x de centro)
            con_id = [(c.carta_id, c.centro[0])
                      for c in cartas if c.carta_id is not None]
            con_id.sort(key=lambda t: t[1])  # ordenar por x
            mano_ids = [cid for cid, _ in con_id[:13]]
        self.log(
            f"Mano leída: {len(mano_ids)}/13 cartas  [{time.perf_counter()-t_lec:.2f}s]")
        if len(mano_ids) < 13:
            return ResultadoPase(direccion=direccion,
                                 nota=f"Solo leí {len(mano_ids)}/13 cartas; no toco nada.")

        a_pasar = list(self.recomendar(mano_ids, direccion))[:3]
        from src.captura.modelos import carta_a_str
        self.log("Modelo recomienda PASAR: "
                 + "  ".join(carta_a_str(c) for c in a_pasar))

        # ── Selección: SIN cache de posiciones. Los bloques de la mano se
        #    reordenan al tocar una carta. Cada tap se hace sobre una captura
        #    fresca, con espera de estabilidad entre carta y carta. ──
        t_sel = time.perf_counter()
        seleccionadas: List[int] = []
        for cid in a_pasar:
            if self._seleccionar(cid):
                seleccionadas.append(cid)
                self.log(
                    f"  ✓ seleccionada {carta_a_str(cid)}  [{time.perf_counter()-t_sel:.2f}s]")
                # Breve pausa para que la animación + reordenamiento de
                # bloques terminen antes de buscar la siguiente carta.
                time.sleep(0.08)
                self._esperar_estabilidad()
            else:
                self.log(
                    f"  ✗ no pude localizar {carta_a_str(cid)} para tocarla")
        self.log(
            f"Selección completada ({len(seleccionadas)}/3 cartas)  [{time.perf_counter()-t_sel:.2f}s]")

        res = ResultadoPase(direccion=direccion, pasadas=seleccionadas)
        if len(seleccionadas) < 3:
            res.nota = "No seleccioné las 3 cartas; no confirmo."
            return res

        # ── 1) verificar zona de pases (ANTES de confirmar) ──
        if not self._verificar_pases(seleccionadas):
            # ── intentar corrección automática ──
            self.log("  🔄 intentando corrección automática del pase...")
            corregido = self._corregir_pase(
                seleccionadas, self.cfg.intentos_correccion)
            if not corregido:
                res.nota = ("Las cartas en la zona de pases no coinciden "
                            "tras corrección. No confirmo.")
                return res

        # ── 2) confirmar ──
        t_conf = time.perf_counter()
        res.confirmado = self._confirmar()
        if not res.confirmado:
            res.nota = ("3 cartas seleccionadas, pero no hay botón de confirmar "
                        "calibrado (regiones.confirmar o plantilla). Confirma a mano.")
            return res
        self.log(f"  ✓ pase confirmado  [{time.perf_counter()-t_conf:.2f}s]")

        # ── 3) esperar banner "Cartas pasadas para ti" y leer recibidas ──
        t_rec = time.perf_counter()
        res.recibidas = self._esperar_recibidas(seleccionadas)
        self.log(f"Cartas RECIBIDAS: "
                 + ("  ".join(carta_a_str(c) for c in res.recibidas)
                    if res.recibidas else "(no detectadas)")
                 + f"  [{time.perf_counter()-t_rec:.2f}s]")
        self.log(f"⏱ TOTAL: {time.perf_counter()-t0:.2f}s")
        return res


__all__ = ["ConfigAutoPase", "ResultadoPase", "ControladorPase", "Recomendar"]
