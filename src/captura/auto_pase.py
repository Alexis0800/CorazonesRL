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

# Callback de decisión: (mano_ids, direccion) -> 3 ids a pasar.
Recomendar = Callable[[List[int], str], List[int]]


@dataclass
class ConfigAutoPase:
    poll_s: float = 0.04           # espera entre capturas en bucles de sondeo
    settle_s: float = 0.05         # fallback si _esperar_estabilidad no converge
    reflujo_s: float = 0.22        # espera tras un tap a que los bloques se reordenen
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
        self.log(
            f"  📊 banner: {r.tag} d={r.distancia:.2f} → {r.categoria}/{r.dato}  |  top: {top_str}")

        # ── guardar recorte del banner para crear plantillas del dispositivo ──
        self._guardar_banner_device(img)

        return r.dato if r.categoria == "pase" else None

    def _guardar_banner_device(self, img: np.ndarray) -> None:
        """Guarda un recorte del banner en debug_dir/banners_device/ para
        crear plantillas especificas de este dispositivo. Solo en modo debug."""
        if not self.cfg.debug_dir:
            return
        from src.captura.banner import capturar_banner
        capturar_banner(img, self.reg, Path(
            self.cfg.debug_dir) / "banners_device")

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

    # ---- helpers de mano (sin captura: operan sobre una lectura ya hecha) ----

    @staticmethod
    def _coord_de(cartas: List[CartaMano],
                  carta_id: int) -> Optional[tuple[int, int]]:
        """Punto de toque de `carta_id` en una lectura de mano, o None."""
        for c in cartas:
            if c.carta_id == carta_id:
                return (int(c.centro[0]), int(c.centro[1]))
        return None

    def _log_mano(self, cartas: List[CartaMano], etapa: str,
                  img: Optional[np.ndarray] = None) -> None:
        """Loguea una lectura de mano YA hecha (no captura ni relee). Si se pasa
        `img` y hay debug_dir, guarda el overlay."""
        from src.captura.modelos import carta_a_str
        mano_ids = [c.carta_id for c in cartas]
        n_ok = sum(1 for c in mano_ids if c is not None)
        self.log(f"  📋 MANO [{etapa}]: {len(cartas)} posiciones  |  "
                 f"{n_ok} reconocidas")
        self.log("    " + " ".join(
            carta_a_str(c) if c is not None else "??" for c in mano_ids))
        self.log("    métodos: " + " | ".join(c.metodo or "--" for c in cartas))
        if img is not None:
            self._debug_mano_overlay(img, cartas, f"dump_{etapa}")

    def _verificar_seleccion_por_mano(self, esperadas: List[int]) -> bool:
        """Verifica la selección RE-LEYENDO LA MANO: si las 3 cartas esperadas
        YA NO están en la mano, es que fueron correctamente seleccionadas.

        Mucho más fiable que leer la zona de pases (cuyas cartas se renderizan
        a escala distinta y confunden J↔5, Q↔9 incluso con la esquina).

        Usa el MISMO `leer_mano_posiciones` que lee 13/13 correctamente."""
        from src.captura.modelos import carta_a_str

        time.sleep(0.2)
        for intento in range(3):
            _, cartas = self._leer_mano_completa()
            mano_ids = {c.carta_id for c in cartas if c.carta_id is not None}
            faltan_en_mano = [cid for cid in esperadas if cid in mano_ids]

            self.log(f"  🔍 verificación mano (intento {intento+1}): "
                     + f"{len(mano_ids)}/13 cartas en mano")

            if not faltan_en_mano:
                self.log("  ✅ las 3 cartas YA NO están en la mano → selección OK")
                return True

            self.log(
                f"  ⚠ aún en mano: {' '.join(carta_a_str(c) for c in faltan_en_mano)}")
            if intento < 2:
                time.sleep(0.25)

        return False

    def _verificar_pase_zona(self, esperadas: List[int],
                             img: Optional[np.ndarray] = None) -> bool:
        """Verificación FINAL leyendo la ZONA DE PASES: las 3 cartas
        seleccionadas deben aparecer ahí. En esa zona las cartas están
        completas y sin solapar, así que `buscar_carta` (carta entera, vía
        `leer_pases`) es muy fiable.

        Reusa `img` (el frame que dejó el último `_tap_y_releer`) → sin captura
        extra en el caso común; si aún no se ven las 3 (carta en vuelo),
        recaptura unas pocas veces."""
        from src.captura.modelos import carta_a_str
        objetivo = sorted(esperadas)
        leidas: List[int] = []
        for intento in range(3):
            if img is None:
                img = self.cli.captura()
            leidas = [p for p in leer_pases(img, self.reg, self.rec)
                      if p is not None]
            if sorted(leidas) == objetivo:
                self.log("  ✅ zona de pases: "
                         + "  ".join(carta_a_str(c) for c in objetivo)
                         + " → selección OK")
                return True
            img = None  # forzar recaptura en el siguiente intento
            time.sleep(self.cfg.poll_s)
        self.log("  ⚠ zona de pases: leídas="
                 + "  ".join(carta_a_str(c) for c in sorted(leidas))
                 + "  |  esperadas="
                 + "  ".join(carta_a_str(c) for c in objetivo))
        return False

    # ---- selección de una carta (SIEMPRE con leer_mano_posiciones) -----

    def _tap_y_releer(self, carta_id: int, coord: tuple[int, int],
                      img_prev: np.ndarray
                      ) -> tuple[np.ndarray, List[CartaMano]]:
        """Toca `carta_id` en `coord` y relee la mano hasta que la carta SALE de
        ella (señal de que la animación de reordenamiento de bloques terminó).

        Esto fusiona en UNA sola captura lo que antes hacían tres pasos con
        captura propia (debug-shot + esperar-estabilidad + dump): el debug-shot
        reusa `img_prev` y el frame final releído sirve a la vez para localizar
        la siguiente carta y para verificar. Devuelve (img, cartas) del último
        frame leído."""
        from src.captura.modelos import carta_a_str
        self._debug_shot(f"tap_{carta_a_str(carta_id)}", punto=coord,
                         img=img_prev)
        self.cli.tap(*coord)
        img, cartas = img_prev, []
        for i in range(self.cfg.intentos_carta):
            time.sleep(self.cfg.reflujo_s if i == 0 else self.cfg.poll_s)
            img = self.cli.captura()
            cartas = leer_mano_posiciones(img, self.reg, self.rec)
            ids = {c.carta_id for c in cartas if c.carta_id is not None}
            if carta_id not in ids:  # ya salió de la mano → reflujo terminado
                break
        return img, cartas

    def _seleccionar(self, carta_id: int,
                     coord: Optional[tuple[int, int]] = None) -> bool:
        """Toca la carta en `coord`. Si no se pasa coordenada, la busca
        re-leyendo la mano con `leer_mano_posiciones`.

        Sin verificación de salida (la usa la ruta de corrección, donde el
        llamante verifica aparte). En la ruta normal se usa `_tap_y_releer`."""
        from src.captura.modelos import carta_a_str

        if coord is not None:
            self.log(
                f"  📍 {carta_a_str(carta_id)} en ({coord[0]},{coord[1]}) → tap")
            self._debug_shot(f"tap_{carta_a_str(carta_id)}", punto=coord)
            self.cli.tap(*coord)
            return True

        # ── fallback: lectura completa (mismo método fiable) ──
        self.log(f"  📸 leyendo mano para localizar {carta_a_str(carta_id)}...")
        img = self.cli.captura()
        cartas = leer_mano_posiciones(img, self.reg, self.rec)
        for c in cartas:
            if c.carta_id == carta_id:
                x, y = int(c.centro[0]), int(c.centro[1])
                self.log(f"  ✓ encontrada → tap ({x},{y})")
                self._debug_shot(f"tap_{carta_a_str(carta_id)}",
                                 punto=(x, y))
                self.cli.tap(x, y)
                return True
        self.log(f"  ✗ NO encontrada {carta_a_str(carta_id)} en la mano")
        return False

    # ---- verificación en zona de pases -----------------------------------

    # ---- corrección automática del pase ----------------------------------

    def _limpiar_pases(self) -> None:
        """Toca repetidamente la zona de pases para de-seleccionar TODAS
        las cartas. Más fiable que intentar de-seleccionar cartas
        individuales (que depende del lector de zona de pases, el cual
        confunde J↔5 y Q↔9 a escala pequeña).

        Estrategia: 3 toques espaciados en el centro de la zona de pases
        + 1 toque en cada extremo. La app de Hearts suele responder a
        taps en la carta seleccionada para de-seleccionarla."""
        if self.reg.pases is None:
            return
        img = self.cli.captura()
        self._debug_shot("limpiar_pases_antes", img=img)
        H, W = img.shape[:2]
        x, y, w, h = self.reg.pases
        cx = int((x + w / 2) * W)
        cy = int((y + h / 2) * H)
        px0, px1 = int(x * W), int((x + w) * W)

        self.log("  🧹 limpiando zona de pases (3 taps en centro + extremos)...")
        for px in [px0 + int(0.15 * w * W), cx, px1 - int(0.15 * w * W)]:
            self.cli.tap(px, cy)
            time.sleep(0.12)
        time.sleep(0.3)
        self._esperar_estabilidad()
        img2 = self.cli.captura()
        self._debug_shot("limpiar_pases_despues", img=img2)

    def _corregir_pase(self, esperadas: List[int],
                       max_intentos: int = 2) -> bool:
        """Corrige la selección: limpia TODO y re-selecciona desde cero.

        Verifica LEYENDO LA MANO (no la zona de pases, que confunde J↔5, Q↔9).
        Si las 3 cartas ya no están en la mano → selección correcta.

        Usa las coordenadas de `_leer_mano_completa`. NO re-busca con otro método.

        Máximo 2 intentos (no infinito).
        """
        from src.captura.modelos import carta_a_str

        for intento in range(max_intentos):
            self.log(f"  🔄 corrección intento {intento+1}/{max_intentos}")

            # ── 1) Limpiar TODO ──
            self._limpiar_pases()

            # ── 2) Re-leer mano → coordenadas fiables ──
            _, cartas = self._leer_mano_completa()
            mano_ids = [c.carta_id for c in cartas if c.carta_id is not None]
            self.log(f"  📖 mano post-limpiar: {len(mano_ids)}/13 cartas")

            # ── 3) Re-seleccionar con coordenadas de la mano leída ──
            ok_count = 0
            for cid in esperadas:
                # ── buscar coordenada en la mano recién leída ──
                coord: Optional[tuple[int, int]] = None
                for c in cartas:
                    if c.carta_id == cid:
                        coord = (int(c.centro[0]), int(c.centro[1]))
                        break
                if coord is None:
                    # ── si no está, _seleccionar hará fallback con leer_mano_posiciones ──
                    self.log(
                        f"  ⚠ {carta_a_str(cid)} no en mano post-limpiar → buscando...")

                if self._seleccionar(cid, coord):
                    ok_count += 1
                    self.log(f"  ✓ re-seleccionada {carta_a_str(cid)}")
                    time.sleep(0.10)
                    self._esperar_estabilidad()
                else:
                    self.log(f"  ✗ no pude re-seleccionar {carta_a_str(cid)}")

            if ok_count < 3:
                self.log(f"  ⚠ solo {ok_count}/3 re-seleccionadas")
                continue

            # ── 4) Verificar LEYENDO LA MANO (no la zona de pases) ──
            time.sleep(0.3)
            if self._verificar_seleccion_por_mano(esperadas):
                self.log(f"  ✅ pase corregido en intento {intento+1}")
                return True

        self.log("  ❌ corrección agotada (2 intentos)")
        return False

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
            r, distancias = self.clf.clasificar_verbose_con_regiones(
                img, self.reg)
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
            _, distancias = self.clf.clasificar_verbose_con_regiones(
                img, self.reg)
            top_str = "    top: " + \
                "  ".join(f"{t}={d:.2f}" for t, d in distancias)
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

    def ejecutar(self, confirmar: bool = True) -> ResultadoPase:
        """Ejecuta el pase completo.

        Si `confirmar=False`, selecciona las 3 cartas y verifica leyendo la
        mano (que las 3 ya no estén), pero NO toca el botón de confirmar y NO
        espera las recibidas. Útil para probar la selección sin comprometerse."""
        t0 = time.perf_counter()
        img = self.cli.captura()
        self._debug_shot("inicio", img=img)
        direccion = self._fase_pase(img)
        if direccion is None:
            return ResultadoPase(direccion="?", nota="No es fase de pase (banner).")
        self.log(
            f"Fase de pase detectada → dirección: {direccion}  [{time.perf_counter()-t0:.2f}s]")

        t_lec = time.perf_counter()
        img, cartas = self._leer_mano_completa()
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

        # ── Selección: localiza cada carta en la lectura ACTUAL de la mano y la
        #    toca. `_tap_y_releer` toca, espera a que la carta salga de la mano
        #    (fin del reordenamiento) y devuelve el nuevo frame+lectura en UNA
        #    sola captura, que alimenta la búsqueda de la siguiente carta. ──
        t_sel = time.perf_counter()
        seleccionadas: List[int] = []
        for cid in a_pasar:
            coord = self._coord_de(cartas, cid)
            if coord is None:
                # ── no está en la lectura actual → re-leer una vez ──
                self.log(
                    f"  ⚠ {carta_a_str(cid)} no en mano actual → re-leyendo...")
                img, cartas = self._leer_mano_completa()
                coord = self._coord_de(cartas, cid)
            if coord is None:
                self.log(
                    f"  ✗ no pude localizar {carta_a_str(cid)} para tocarla")
                continue

            self.log(
                f"  📍 {carta_a_str(cid)} en ({coord[0]},{coord[1]}) → tap")
            img, cartas = self._tap_y_releer(cid, coord, img)
            seleccionadas.append(cid)
            self.log(
                f"  ✓ seleccionada {carta_a_str(cid)}  [{time.perf_counter()-t_sel:.2f}s]")
            if self.cfg.debug_dir:
                self._log_mano(cartas, f"post_{carta_a_str(cid)}", img)
        self.log(
            f"Selección completada ({len(seleccionadas)}/3 cartas)  [{time.perf_counter()-t_sel:.2f}s]")

        res = ResultadoPase(direccion=direccion, pasadas=seleccionadas)
        if len(seleccionadas) < 3:
            res.nota = "No seleccioné las 3 cartas; no confirmo."
            return res

        # ── Verificación FINAL: leer la ZONA DE PASES (carta completa, fiable
        #    ahí) reusando el frame del último tap → sin captura extra. Si la
        #    zona no está calibrada, cae a comprobar que ya no estén en la mano.
        if self.reg.pases is not None:
            seleccion_ok = self._verificar_pase_zona(seleccionadas, img=img)
        else:
            en_mano = {c.carta_id for c in cartas if c.carta_id is not None}
            seleccion_ok = not any(c in en_mano for c in seleccionadas)
            self.log("  ✅ las 3 cartas YA NO están en la mano → selección OK"
                     if seleccion_ok else "  ⚠ alguna sigue en la mano")

        if not seleccion_ok:
            # ── SIEMPRE intentar corrección (incluso en modo sin-confirmar) ──
            self.log("  🔄 intentando corrección automática del pase...")
            corregido = self._corregir_pase(
                seleccionadas, self.cfg.intentos_correccion)
            if not corregido:
                if confirmar:
                    res.nota = ("Las cartas no desaparecieron de la mano "
                                "tras corrección. No confirmo.")
                    return res
                else:
                    self.log(
                        "  ⚠ corrección no pudo verificar (modo sin-confirmar: continúo igual)")

        # ── 2) confirmar (solo si se pidió) ──
        if not confirmar:
            res.nota = ("3 cartas seleccionadas y verificadas. "
                        "No confirmo (modo sin-confirmar).")
            self.log(f"⏱ TOTAL (sin confirmar): {time.perf_counter()-t0:.2f}s")
            return res

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
