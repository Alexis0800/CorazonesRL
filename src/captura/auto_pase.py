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

from src.captura.adb import ClienteADB
from src.captura.vision_hearts import (BannerClasificador, CartaMano, Regiones,
                                       leer_mano, leer_mano_posiciones)

# Callback de decisión: (mano_ids, direccion) -> 3 ids a pasar.
Recomendar = Callable[[List[int], str], List[int]]


@dataclass
class ConfigAutoPase:
    poll_s: float = 0.5            # espera entre capturas en bucles de sondeo
    settle_s: float = 0.9         # espera tras un tap (animación de selección)
    intentos_lectura: int = 8     # reintentos para leer la mano completa (13)
    intentos_carta: int = 4       # reintentos para localizar+tocar una carta
    intentos_recibidas: int = 30  # sondeos esperando a que el pase se resuelva
    umbral_confirmar: float = 0.6 # correlación mínima de la plantilla del botón


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

    def _seleccionar(self, carta_id: int) -> bool:
        """Localiza la carta por id (releyendo la mano) y la toca. Reintenta:
        tras seleccionar, los bloques se reordenan, así que cada vez se vuelve a
        mirar dónde quedó la siguiente."""
        for _ in range(self.cfg.intentos_carta):
            img = self.cli.captura()
            cartas = leer_mano_posiciones(img, self.reg, self.rec)
            objetivo = next((c for c in cartas if c.carta_id == carta_id), None)
            if objetivo is not None:
                x, y = objetivo.centro
                self.cli.tap(x, y)
                time.sleep(self.cfg.settle_s)
                return True
            time.sleep(self.cfg.poll_s)
        return False

    # ---- confirmar (botón círculo-check) ----------------------------------

    def _confirmar(self) -> bool:
        """Toca el botón de confirmar. Prefiere localizarlo por plantilla dentro
        de su región; si no hay plantilla, toca el centro de la región. Si no hay
        ninguna de las dos cosas calibrada, no puede confirmar."""
        if self._tpl_confirmar and self._tpl_confirmar.is_file():
            punto = self._localizar_confirmar()
            if punto is not None:
                self.cli.tap(*punto)
                time.sleep(self.cfg.settle_s)
                return True
        if self.reg.confirmar is not None:
            img = self.cli.captura()
            H, W = img.shape[:2]
            x, y, w, h = self.reg.confirmar
            self.cli.tap(int((x + w / 2) * W), int((y + h / 2) * H))
            time.sleep(self.cfg.settle_s)
            return True
        return False

    def _localizar_confirmar(self) -> Optional[tuple[int, int]]:
        import cv2

        img = self.cli.captura()
        H, W = img.shape[:2]
        if self.reg.confirmar is not None:
            x, y, w, h = self.reg.confirmar
            x0, y0 = int(x * W), int(y * H)
            roi = img[y0:y0 + int(h * H), x0:x0 + int(w * W)]
        else:
            x0, y0, roi = 0, 0, img
        tpl = cv2.imread(str(self._tpl_confirmar), cv2.IMREAD_COLOR)
        if tpl is None or tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
            return None
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < self.cfg.umbral_confirmar:
            return None
        return (x0 + maxloc[0] + tpl.shape[1] // 2,
                y0 + maxloc[1] + tpl.shape[0] // 2)

    # ---- cartas recibidas (por diferencia, sin región especial) -----------

    def _esperar_recibidas(self, mano_antes: List[int], pasadas: List[int]
                           ) -> List[int]:
        """Espera a que el pase se resuelva y deduce las cartas recibidas:
        recibidas = mano_nueva − (mano_antes − pasadas)."""
        base = set(mano_antes) - set(pasadas)   # lo que conservo
        for _ in range(self.cfg.intentos_recibidas):
            img = self.cli.captura()
            # cuando ya no estamos en fase de pase, o la mano vuelve a 13, leemos
            mano = [c for c in leer_mano(img, self.reg, self.rec) if c is not None]
            recibidas = sorted(set(mano) - base)
            if len(mano) >= 13 and len(recibidas) >= 3:
                return recibidas[:3] if len(recibidas) > 3 else recibidas
            time.sleep(self.cfg.poll_s)
        # último intento: lo que haya
        img = self.cli.captura()
        mano = [c for c in leer_mano(img, self.reg, self.rec) if c is not None]
        return sorted(set(mano) - base)

    # ---- orquestación -----------------------------------------------------

    def ejecutar(self) -> ResultadoPase:
        img = self.cli.captura()
        direccion = self._fase_pase(img)
        if direccion is None:
            return ResultadoPase(direccion="?", nota="No es fase de pase (banner).")
        self.log(f"Fase de pase detectada → dirección: {direccion}")

        _, cartas = self._leer_mano_completa()
        mano_ids = [c.carta_id for c in cartas if c.carta_id is not None]
        if len(mano_ids) < 13:
            return ResultadoPase(direccion=direccion,
                                 nota=f"Solo leí {len(mano_ids)}/13 cartas; no toco nada.")

        a_pasar = list(self.recomendar(mano_ids, direccion))[:3]
        from src.captura.modelos import carta_a_str
        self.log("Modelo recomienda PASAR: "
                 + "  ".join(carta_a_str(c) for c in a_pasar))

        seleccionadas: List[int] = []
        for cid in a_pasar:
            if self._seleccionar(cid):
                seleccionadas.append(cid)
                self.log(f"  ✓ seleccionada {carta_a_str(cid)}")
            else:
                self.log(f"  ✗ no pude localizar {carta_a_str(cid)} para tocarla")

        res = ResultadoPase(direccion=direccion, pasadas=seleccionadas)
        if len(seleccionadas) < 3:
            res.nota = "No seleccioné las 3 cartas; no confirmo."
            return res

        res.confirmado = self._confirmar()
        if not res.confirmado:
            res.nota = ("3 cartas seleccionadas, pero no hay botón de confirmar "
                        "calibrado (regiones.confirmar o plantilla). Confirma a mano.")
            return res
        self.log("  ✓ pase confirmado")

        res.recibidas = self._esperar_recibidas(mano_ids, seleccionadas)
        self.log("Cartas RECIBIDAS: "
                 + ("  ".join(carta_a_str(c) for c in res.recibidas)
                    if res.recibidas else "(no detectadas)"))
        return res


__all__ = ["ConfigAutoPase", "ResultadoPase", "ControladorPase", "Recomendar"]
