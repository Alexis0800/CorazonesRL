"""
Auto-juego de las BAZAS por ADB: cierra el lazo visión→decisión→acción para la
fase de juego (lo que falta tras el pase, ver `auto_pase.py`).

Idea clave (más robusta que reconstruir la legalidad desde la mesa, que se lee
mal): la app OSCURECE las cartas que no se pueden jugar y deja CLARAS solo las
legales. Así que cada turno:

  1. Espera el banner "tu turno".
  2. Lee la mano y mide el BRILLO de cada carta → las claras son las jugables.
     - Si hay 1 jugable, la toca directo (rápido, sin consultar al modelo).
     - Si hay varias, el modelo elige ENTRE esas (`recomendar_jugada_entre`).
  3. Cuenta cuántas cartas hay puestas en la mesa (detección de PRESENCIA, más
     fiable que identificarlas) → deduce mi posición y el líder de la baza.
  4. Toca la carta elegida.
  5. Espera a que estén las 4 cartas puestas (o el banner "X recoge la baza"),
     las lee y registra la baza (cementerio, vacíos, corazones, Q♠).

Robustez: si una baza no se puede leer completa, se DEGRADA con gracia (quita mi
carta de la mano y avanza) en vez de colgarse. Nunca toca una carta que no
localiza positivamente por plantilla.

⚠ Auto-tocar una app puede violar sus términos de servicio. Úsalo bajo tu
responsabilidad, para fines personales/de investigación.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

import cv2

from src.captura.adb import ClienteADB
from src.captura.maquina import ROTACION_HORARIA
from src.captura.vision_hearts import (BannerClasificador, CartaMano, Regiones,
                                       leer_mano_posiciones, leer_mesa)
from src.dominio.carta import Carta


@dataclass
class ConfigAutoJuego:
    poll_s: float = 0.15            # espera entre sondeos de presencia (baratos)
    settle_s: float = 0.35         # espera tras un tap antes de releer
    timeout_turno_s: float = 30.0  # espera máx. por mi turno; si no llega, se
    #                                asume fin de mano (contra bots el turno
    #                                llega en segundos). Evita colgarse 3 min.
    timeout_baza_s: float = 20.0   # espera máxima a que la baza se complete
    # Factor (× el blanco más alto de la mano) por encima del cual un píxel se
    # cuenta como "blanco real". El velo gris de las cartas no jugables no llega
    # ahí, así que su blancura cae a 0%. Súbelo si marca de más; bájalo si de menos.
    umbral_brillo_rel: float = 0.85
    confirmar_jugada: bool = False  # si la app exige confirmar tras tocar la carta
    debug_dir: Optional[str] = None


class ControladorBazas:
    """Conduce la fase de bazas tocando la pantalla por ADB.

    Usa el `Recomendador` (estado público + modelo) para decidir, la visión para
    leer la mano/mesa, y ADB para tocar. Una llamada a `jugar_mano()` juega las
    13 bazas de la mano EN CURSO (la mano del agente debe estar ya cargada en
    `recomendador.mano`, p.ej. tras aplicar el pase).
    """

    def __init__(
        self,
        cliente: ClienteADB,
        regiones: Regiones,
        banner_clf: BannerClasificador,
        reconocedor_mano,
        recomendador,
        seat_de_posicion: Optional[Dict[str, int]] = None,
        config: Optional[ConfigAutoJuego] = None,
        confirmar_jugada_fn: Optional[Callable[[], bool]] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cli = cliente
        self.reg = regiones
        self.clf = banner_clf
        self.rec_mano = reconocedor_mano
        self.rec = recomendador
        self.seat_de_pos = dict(seat_de_posicion or ROTACION_HORARIA)
        self.cfg = config or ConfigAutoJuego()
        self.confirmar_jugada_fn = confirmar_jugada_fn
        self.log = log
        self.me = self.seat_de_pos.get("abajo", 0)
        self._debug_n = 0
        self.fin_mano = False   # se activa al detectar «se lleva el resto»

    # ---- debug ------------------------------------------------------------

    def _debug_shot(self, etapa: str, punto: Optional[tuple[int, int]] = None,
                    img: Optional[np.ndarray] = None) -> None:
        if not self.cfg.debug_dir:
            return
        shot = img.copy() if img is not None else self.cli.captura()
        if punto is not None:
            px, py = punto
            cv2.line(shot, (px - 20, py), (px + 20, py), (0, 0, 255), 3)
            cv2.line(shot, (px, py - 20), (px, py + 20), (0, 0, 255), 3)
            cv2.circle(shot, (px, py), 25, (0, 0, 255), 2)
        name = f"{self.cfg.debug_dir}/baza_{self._debug_n:03d}_{etapa}.png"
        Path(self.cfg.debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(name, shot)
        self._debug_n += 1

    def _debug_mano_jugables(self, img: np.ndarray, cartas: List[CartaMano],
                             jugables: List[CartaMano], etapa: str) -> None:
        """Overlay de la mano: VERDE = jugable (clara), GRIS = oscurecida."""
        if not self.cfg.debug_dir:
            return
        from src.captura.modelos import carta_a_str
        ids_jug = {id(c) for c in jugables}
        out = img.copy()
        for c in cartas:
            x, y = int(c.centro[0]), int(c.centro[1])
            verde = id(c) in ids_jug
            color = (0, 255, 0) if verde else (140, 140, 140)
            cv2.circle(out, (x, y), 16, color, 3 if verde else 2)
            etq = carta_a_str(c.carta_id) if c.carta_id is not None else "??"
            cv2.putText(out, etq, (x - 28, y - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        name = f"{self.cfg.debug_dir}/baza_{self._debug_n:03d}_{etapa}.png"
        Path(self.cfg.debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(name, out)
        self._debug_n += 1

    # ---- banner / mesa ----------------------------------------------------

    def _banner(self, img: np.ndarray):
        return self.clf.clasificar_con_regiones(img, self.reg)

    def _es_mi_turno(self, b) -> bool:
        return b.categoria == "turno" and b.dato == "abajo"

    def _presencia_mesa(self, img: np.ndarray) -> Dict[int, str]:
        """Cuenta qué posiciones de la mesa tienen una carta PUESTA (blob),
        independientemente de si se reconoce su valor. Más fiable que
        identificarla. Devuelve {asiento: posicion_pantalla}."""
        from src.captura.vision_cartas import detectar_cartas
        out: Dict[int, str] = {}
        for pos, caja in self.reg.mesa.items():
            roi = Regiones.recortar(img, caja)
            if detectar_cartas(roi, min_area_frac=0.05):
                asiento = self.seat_de_pos.get(pos)
                if asiento is not None:
                    out[asiento] = pos
        return out

    # Escalas estrechas para la mesa: las cartas se ven enteras y a tamano casi
    # fijo, asi que basta ~1.0 ±10%. Acelera la identificacion (la lectura cara)
    # sin perder acierto frente a probar las 10 escalas.
    _ESCALAS_MESA = (0.9, 0.95, 1.0, 1.05, 1.1)

    def _seats_en_mesa(self, img: np.ndarray) -> Dict[int, int]:
        """Lee la mesa e identifica {asiento: carta_id} (solo las reconocidas)."""
        out: Dict[int, int] = {}
        for pos, cid in leer_mesa(img, self.reg, self.rec_mano,
                                  escalas=self._ESCALAS_MESA).items():
            if cid is None:
                continue
            asiento = self.seat_de_pos.get(pos)
            if asiento is not None:
                out[asiento] = cid
        return out

    # ---- brillo (carta jugable vs oscurecida) ----------------------------

    def _franja_gris(self, gray: np.ndarray, cm: CartaMano) -> np.ndarray:
        """Recorte en gris de la franja visible de una carta de la mano."""
        half_w = max(2, cm.visible_w // 2)
        half_h = max(2, cm.card_h // 2)
        x0 = max(0, cm.centro[0] - half_w)
        x1 = min(gray.shape[1], cm.centro[0] + half_w)
        y0 = max(0, cm.centro[1] - half_h)
        y1 = min(gray.shape[0], cm.centro[1] + half_h)
        return gray[y0:y1, x0:x1]

    def _cartas_jugables(self, img: np.ndarray,
                         cartas: List[CartaMano]) -> List[CartaMano]:
        """De la mano leída, devuelve las cartas CLARAS (jugables).

        La app pone un velo GRIS sobre las no jugables que SATURA el blanco del
        fondo (~180), mientras las jugables conservan blanco real (~245). Así que
        el discriminante robusto NO es el brillo medio (las cartas de pocos pips
        tienen mucho fondo y engañan), sino la BLANCURA: fracción de píxeles
        realmente blancos. Las oscurecidas dan 0%; las jugables, decenas de %.
        Funciona igual si todas son jugables (todas con blanco) que si solo una.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        medidos = [(cm, self._franja_gris(gray, cm))
                   for cm in cartas if cm.carta_id is not None]
        medidos = [(cm, c) for cm, c in medidos if c.size > 0]
        if not medidos:
            return []
        # Umbral de "blanco real": una fracción del blanco más alto visto. El
        # velo gris no llega ahí, así que separa jugables de oscurecidas.
        maxbright = max(float(np.percentile(c, 98)) for _, c in medidos)
        umbral_blanco = max(200.0, self.cfg.umbral_brillo_rel * maxbright)
        jugables = [cm for cm, c in medidos
                    if float((c > umbral_blanco).mean()) > 0.03]
        return jugables or [cm for cm, _ in medidos]  # nunca vacío

    # ---- decisión ---------------------------------------------------------

    def _mesa_antes(self, ids: Dict[int, int], lider: int) -> List[tuple]:
        """[(asiento, Carta)] de las cartas YA reconocidas en la mesa, en orden de
        turno desde `lider` (sin mi propia carta). Solo para enriquecer la obs."""
        presentes = {a: c for a, c in ids.items() if a != self.me}
        orden = sorted(presentes, key=lambda a: (a - lider) % 4)
        return [(a, Carta._TODAS[presentes[a]]) for a in orden]

    def _elegir(self, img: np.ndarray, jugables: List[CartaMano],
                ids_mesa: Dict[int, int], lider: int) -> Optional[CartaMano]:
        """Elige qué CartaMano tocar entre las jugables (claras)."""
        from src.captura.modelos import carta_a_str
        if not jugables:
            return None
        if len(jugables) == 1:
            self.log(f"  ☝ única jugable: {carta_a_str(jugables[0].carta_id)}")
            return jugables[0]
        mesa_antes = self._mesa_antes(ids_mesa, lider)
        candidatas = [Carta._TODAS[cm.carta_id] for cm in jugables]
        elegida = self.rec.recomendar_jugada_entre(mesa_antes, candidatas)
        self.log("  ★ modelo elige " + carta_a_str(elegida.id)
                 + "  entre [" + " ".join(carta_a_str(c.id) for c in candidatas) + "]")
        for cm in jugables:
            if cm.carta_id == elegida.id:
                return cm
        return jugables[0]

    # ---- baza ------------------------------------------------------------

    @staticmethod
    def _ganador(jugadas: List[tuple]) -> int:
        """jugadas = [(asiento, Carta)] en orden de turno. Asiento que gana."""
        palo_salida = jugadas[0][1].palo
        mejor_asiento, mejor_valor = jugadas[0][0], -1
        for asiento, c in jugadas:
            if c.palo == palo_salida and c.valor > mejor_valor:
                mejor_asiento, mejor_valor = asiento, c.valor
        return mejor_asiento

    def _esperar_turno_agente(self) -> Optional[np.ndarray]:
        """Sondea hasta el banner 'tu turno'. Devuelve un frame fresco y asentado
        (cartas y oscurecimiento ya renderizados), o None si la mano terminó
        (banner «se lleva el resto», o no llega mi turno → se asume fin de mano).

        Si la mano terminó, deja `self.fin_mano = True` para que `jugar_mano` lo
        reporte como cierre limpio (no como error)."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.cfg.timeout_turno_s:
            img = self.cli.captura()
            b = self._banner(img)
            # «X se llevará todo el resto» → la mano acaba sin más jugadas.
            if b.categoria == "remate":
                self.fin_mano = True
                self.log("  🏁 banner «se lleva el resto» → fin de la mano.")
                return None
            if self._es_mi_turno(b):
                time.sleep(self.cfg.settle_s)   # dejar asentar el oscurecimiento
                return self.cli.captura()
            time.sleep(self.cfg.poll_s)
        # Contra bots el turno llega en segundos; si no llega en `timeout_turno_s`
        # la mano ya terminó (p.ej. remate aún sin plantilla). Cierre limpio en
        # vez de colgarse 3 min.
        self.fin_mano = True
        self.log(f"  🏁 sin mi turno en {self.cfg.timeout_turno_s:.0f}s → "
                 "asumo fin de la mano.")
        return None

    def _esperar_y_registrar_baza(self, lider: int, mi_carta: Carta,
                                  numero: int,
                                  mesa_previa: Optional[Dict[int, int]] = None
                                  ) -> None:
        """Tras jugar yo, ACUMULA las cartas de la mesa a lo largo de la baza
        (cada una cuando se ve estable, no todas de golpe al final) y REGISTRA
        cuando el banner anuncia que alguien recoge la baza. Degrada con gracia.

        `mesa_previa` (asiento->carta_id) son las cartas que YA estaban puestas
        cuando me tocó jugar, leídas en el frame ASENTADO de mi turno (lectura
        fiable). Se siembran en el buffer: así, si juego ÚLTIMO, la baza queda
        completa de inmediato sin depender de leer durante la animación (que falla
        y antes colgaba). Para las cartas jugadas DESPUÉS de mí se sondea por
        PRESENCIA (barato) y se identifican TODAS de una vez sobre el frame en que
        la baza se completa (4 cartas aún estáticas, antes de la animación)."""
        from src.captura.modelos import carta_a_str
        buf: Dict[int, int] = {self.me: mi_carta.id}   # mi carta la sé de cierto
        for seat, cid in (mesa_previa or {}).items():  # cartas previas (fiables)
            if seat != self.me:
                buf[seat] = cid
        tomada = False
        ult_img = None
        # Si NO juego último, faltan cartas: las jugadas DESPUÉS de mí. Sondeo
        # BARATO por PRESENCIA (~0.002s/lectura vs ~0.4s identificando) hasta que
        # la baza está completa (4 cartas en la mesa) o alguien la recoge; SOLO
        # entonces identifico (caro, una vez), con las cartas aún estáticas. Antes
        # se identificaba en cada vuelta (~1s/vuelta): demasiado lento para seguir
        # el ritmo de los bots → se perdían cartas y se colgaba hasta el timeout.
        if len(buf) < 4:
            t0 = time.perf_counter()
            disparo: Optional[np.ndarray] = None
            pico = 0  # nº máx. de cartas vistas en la mesa
            while time.perf_counter() - t0 < self.cfg.timeout_baza_s:
                img = self.cli.captura()
                ult_img = img
                b = self._banner(img)
                npres = len(self._presencia_mesa(img))   # barato (sin identificar)
                pico = max(pico, npres)
                if b.categoria == "baza":          # "X recoge la baza"
                    tomada = True
                    disparo = img
                    break
                if npres >= 4:                     # baza completa, aún estática
                    disparo = img
                    break
                if pico >= 2 and npres == 0:       # ya recogida y mesa limpia
                    break                          # (perdí la ventana; no cuelgo)
                time.sleep(self.cfg.poll_s)
            # Identificar las 4 de una sola vez sobre el frame de disparo.
            if disparo is not None:
                for seat, cid in self._seats_en_mesa(disparo).items():
                    if seat != self.me and seat not in buf:
                        buf[seat] = cid

        if ult_img is not None:
            self._debug_shot(f"baza{numero}_cierre", img=ult_img)

        orden = [(lider + j) % 4 for j in range(4)]
        jugadas = [(a, Carta._TODAS[buf[a]]) for a in orden if a in buf]
        if len(jugadas) == 4:
            ganador = self._ganador(jugadas)
            self.rec.registrar_baza(jugadas, ganador)
            quien = "YO" if ganador == self.me else f"a{ganador}"
            self.log("  baza: "
                     + "  ".join(carta_a_str(c.id) for _, c in jugadas)
                     + f"  → recoge {quien}"
                     + ("" if tomada else "  (por conteo, sin banner)"))
            return
        # ── degradar: al menos quitar mi carta para avanzar la mano ──
        faltan = [a for a in range(4) if a not in buf]
        self.log(f"  ⚠ baza incompleta: leí {len(jugadas)}/4 "
                 + "(" + " ".join(carta_a_str(c.id) for _, c in jugadas) + ")"
                 + f"; faltan asientos {faltan}. Descuento mi carta y sigo.")
        if mi_carta in self.rec.mano:
            self.rec.mano.remove(mi_carta)
        self.rec.numero_baza += 1

    def jugar_baza(self, numero: int) -> bool:
        """Juega UNA baza. Devuelve False si no pude (fin de mano / no localizo).

        Registra tiempos por fase (turno · mano · jugar · baza) para ver dónde se
        va el tiempo."""
        from src.captura.modelos import carta_a_str

        self.log(f"\n── Baza {numero}")
        t_ini = time.perf_counter()
        img = self._esperar_turno_agente()
        t_turno = time.perf_counter()
        if img is None:
            # fin_mano (remate / no llega turno) ya lo reportó _esperar_turno_agente
            return False

        # ── 1) mano + brillo → cartas jugables (claras) ──
        cartas = leer_mano_posiciones(img, self.reg, self.rec_mano)
        jugables = self._cartas_jugables(img, cartas)
        self._debug_mano_jugables(img, cartas, jugables, f"baza{numero}_mano")
        self.log(f"  🖐 mano: {len([c for c in cartas if c.carta_id is not None])} leídas, "
                 + f"{len(jugables)} jugables: "
                 + " ".join(carta_a_str(c.carta_id) for c in jugables))

        # ── 2) posición en la baza por CONTEO de cartas puestas (no por ID) ──
        present = self._presencia_mesa(img)
        present.pop(self.me, None)
        k = len(present)
        lider = (self.me - k) % 4
        ids_mesa = self._seats_en_mesa(img)     # IDs best-effort, solo para la obs
        ids_mesa.pop(self.me, None)
        self.log(f"  🪑 {k} cartas antes de mí → líder=a{lider}"
                 + (f"  (leídas: {' '.join(carta_a_str(c) for c in ids_mesa.values())})"
                    if ids_mesa else ""))
        t_mano = time.perf_counter()

        # ── 3) elegir y tocar ──
        if not jugables:
            self.log("  ⚠ no detecté cartas jugables por brillo; uso el modelo "
                     "sobre la mano completa (fallback).")
            cm = self._fallback_modelo(img, cartas, ids_mesa, lider)
        else:
            cm = self._elegir(img, jugables, ids_mesa, lider)
        if cm is None or cm.carta_id is None:
            self.log("  ✗ no pude elegir/localizar carta; abandono la mano.")
            return False

        carta = Carta._TODAS[cm.carta_id]
        x, y = int(cm.centro[0]), int(cm.centro[1])
        self.log(f"  📍 juego {carta_a_str(carta.id)} en ({x},{y}) → tap")
        self._debug_shot(f"baza{numero}_tap_{carta_a_str(carta.id)}",
                         punto=(x, y), img=img)
        self.cli.tap(x, y)
        time.sleep(self.cfg.settle_s)
        if self.cfg.confirmar_jugada and self.confirmar_jugada_fn:
            self.confirmar_jugada_fn()
        t_tap = time.perf_counter()

        # ── 4) esperar cierre de baza y registrar ──
        #    `ids_mesa` (leído en mi frame asentado) es fiable: lo paso como
        #    semilla para no depender de releer esas cartas en la animación.
        self._esperar_y_registrar_baza(lider, carta, numero, mesa_previa=ids_mesa)
        t_fin = time.perf_counter()
        self.log(f"  ⏱ turno {t_turno - t_ini:.1f}s · mano {t_mano - t_turno:.1f}s"
                 f" · jugar {t_tap - t_mano:.1f}s · baza {t_fin - t_tap:.1f}s"
                 f" · total {t_fin - t_ini:.1f}s")
        return True

    def _fallback_modelo(self, img: np.ndarray, cartas: List[CartaMano],
                         ids_mesa: Dict[int, int], lider: int
                         ) -> Optional[CartaMano]:
        """Si el brillo no detectó jugables, deja que el modelo elija sobre la
        mano completa (legalidad del motor) y devuelve esa CartaMano."""
        legibles = [cm for cm in cartas if cm.carta_id is not None]
        if not legibles:
            return None
        mesa_antes = self._mesa_antes(ids_mesa, lider)
        candidatas = [Carta._TODAS[cm.carta_id] for cm in legibles]
        elegida = self.rec.recomendar_jugada_entre(mesa_antes, candidatas)
        for cm in legibles:
            if cm.carta_id == elegida.id:
                return cm
        return legibles[0]

    # ---- mano completa ----------------------------------------------------

    def jugar_mano(self) -> None:
        """Juega las bazas hasta vaciar la mano del agente (máx. 13).

        Cada baza se aísla: si una lanza una excepción (p.ej. un frame raro
        durante la animación), se registra el traceback y se ABANDONA la mano con
        gracia en vez de tumbar toda la sesión."""
        import traceback
        baza = 1
        while self.rec.mano and baza <= 13:
            try:
                ok = self.jugar_baza(baza)
            except Exception:
                self.log(f"  ✗ excepción en la baza {baza}; abandono la mano:")
                self.log(traceback.format_exc())
                self._debug_shot(f"baza{baza}_ERROR")
                break
            if not ok:
                break
            baza += 1
        cierre = ("🏁 se llevó el resto" if self.fin_mano
                  else "mano vacía" if not self.rec.mano else "mano abandonada")
        self.log(f"\n✅ Mano jugada ({baza-1} bazas) — {cierre}."
                 + (f" Quedan {len(self.rec.mano)} cartas en mano."
                    if self.rec.mano else ""))


def continuar_tras_recibir(cliente: ClienteADB,
                           log: Callable[[str], None] = print,
                           debug_dir: Optional[str] = None) -> tuple[int, int]:
    """Descarta el overlay de 'Cartas pasadas para ti' tocando el CENTRO de la
    pantalla, para que arranque la primera baza.

    En esta app el overlay de cartas recibidas se cierra con un toque en
    cualquier parte; el centro-arriba es seguro (no cae sobre la mano)."""
    img = cliente.captura()
    H, W = img.shape[:2]
    punto = (W // 2, int(H * 0.45))
    log(f"  👆 continuar tras recibir: tap centro ({punto[0]}, {punto[1]})")
    if debug_dir:
        shot = img.copy()
        cv2.line(shot, (punto[0] - 20, punto[1]),
                 (punto[0] + 20, punto[1]), (0, 0, 255), 3)
        cv2.line(shot, (punto[0], punto[1] - 20),
                 (punto[0], punto[1] + 20), (0, 0, 255), 3)
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/continuar_tras_recibir.png", shot)
    cliente.tap(*punto)
    time.sleep(0.4)
    return punto


__all__ = ["ConfigAutoJuego", "ControladorBazas", "continuar_tras_recibir"]
