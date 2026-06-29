"""
`AdaptadorVisual`: implementacion de `AdaptadorJuego` que produce eventos a
partir de una FUENTE DE FOTOGRAMAS, usando la vision (`vision_hearts` +
`vision_cartas`) y la `MaquinaCaptura`.

La fuente es cualquier iterable de imagenes BGR:
  - carpeta del video / fotogramas exportados -> validar offline TODO el pipeline.
  - ADB en vivo (poll de `ClienteADB.captura()`) -> capturar de verdad.

El MISMO cerebro (maquina) sirve para ambos: solo cambia de donde salen los
frames. Eso permite probar sin movil y luego enchufar ADB sin tocar la logica.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np

from src.captura.maquina import MaquinaCaptura, ROTACION_HORARIA
from src.captura.puerto import AdaptadorJuego, Evento, InicioPartida
from src.captura.vision_hearts import (
    BannerClasificador, Regiones, leer_estado,
)
from src.captura.vision_cartas import ReconocedorPlantilla


def fuente_carpeta(carpeta: str | Path, patron: str = "*.png",
                   paso: int = 1) -> Iterator[np.ndarray]:
    """Itera fotogramas de una carpeta en orden. `paso` submuestrea (1=todos)."""
    import cv2

    archivos = sorted(Path(carpeta).glob(patron))
    for i, fp in enumerate(archivos):
        if i % paso:
            continue
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is not None:
            yield img


def _firma_rapida(img: np.ndarray) -> bytes:
    import cv2

    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (24, 24),
                   interpolation=cv2.INTER_AREA)
    return (g > g.mean()).astype(np.uint8).tobytes()


class AdaptadorVisual(AdaptadorJuego):
    def __init__(
        self,
        fuente_frames: Iterable[np.ndarray],
        regiones: Regiones,
        banner_clf: BannerClasificador,
        reconocedor: ReconocedorPlantilla,
        asiento_agente: int = 0,
        app: str = "hearts",
        rotacion: Optional[dict] = None,
        marcador_final: Optional[List[int]] = None,
        saltar_repetidos: bool = False,
    ) -> None:
        self.fuente_frames = fuente_frames
        self.regiones = regiones
        self.banner_clf = banner_clf
        self.reconocedor = reconocedor
        self.asiento_agente = asiento_agente
        self.app = app
        self.rotacion = dict(rotacion or ROTACION_HORARIA)
        self.marcador_final = marcador_final
        self.saltar_repetidos = saltar_repetidos

    def eventos(self) -> Iterator[Evento]:
        yield InicioPartida(asiento_agente=self.asiento_agente,
                            fuente=f"adb:{self.app}")
        maquina = MaquinaCaptura(asiento_agente=self.asiento_agente,
                                 seat_de_posicion=self.rotacion)
        ultima_firma = None
        for img in self.fuente_frames:
            if self.saltar_repetidos:
                f = _firma_rapida(img)
                if f == ultima_firma:
                    continue
                ultima_firma = f
            estado = leer_estado(img, self.regiones, self.banner_clf,
                                 self.reconocedor)
            yield from maquina.procesar(estado)
        yield from maquina.finalizar(self.marcador_final)


__all__ = ["AdaptadorVisual", "fuente_carpeta"]
