"""
Orquestador: consume el stream de eventos de un `AdaptadorJuego`, los agrega en
`RegistroPartida` y los persiste con un `EscritorJsonl`.

No sabe NADA de ADB ni de consola: solo de eventos. Esa es la frontera SOLID que
permite cambiar de app/fuente sin tocar esta clase.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, List, Optional

from src.captura.escritor import EscritorJsonl
from src.captura.modelos import Jugada, RegistroMano, RegistroPartida
from src.captura.puerto import (
    AdaptadorJuego, FinMano, FinPartida, InicioMano, InicioPartida,
    JugadaObservada, PaseAgente,
)


def _ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ranking(marcador: List[int]) -> List[int]:
    """Asientos ordenados de MEJOR a peor (menos puntos gana en Corazones)."""
    return sorted(range(len(marcador)), key=lambda s: marcador[s])


class RecolectorPartidas:
    def __init__(
        self,
        adaptador: AdaptadorJuego,
        escritor: Optional[EscritorJsonl] = None,
        al_terminar_partida: Optional[Callable[[RegistroPartida], None]] = None,
    ) -> None:
        self.adaptador = adaptador
        self.escritor = escritor
        self.al_terminar_partida = al_terminar_partida

    def ejecutar(self) -> List[RegistroPartida]:
        """Consume todos los eventos y devuelve las partidas completadas."""
        completadas: List[RegistroPartida] = []
        partida: Optional[RegistroPartida] = None
        mano: Optional[RegistroMano] = None
        try:
            for ev in self.adaptador.eventos():
                if isinstance(ev, InicioPartida):
                    partida = RegistroPartida(
                        partida_id=ev.partida_id or uuid.uuid4().hex[:12],
                        timestamp=_ahora_iso(),
                        asiento_agente=ev.asiento_agente,
                        fuente=ev.fuente,
                    )
                    mano = None
                elif isinstance(ev, InicioMano):
                    self._exigir(partida, "InicioMano sin InicioPartida")
                    mano = RegistroMano(
                        numero_mano=ev.numero_mano,
                        direccion_pase=ev.direccion_pase,
                        mano_inicial_agente=list(ev.mano_agente),
                    )
                    partida.manos.append(mano)
                elif isinstance(ev, PaseAgente):
                    self._exigir(mano, "PaseAgente sin InicioMano")
                    mano.pase_dado = list(ev.dadas)
                    mano.pase_recibido = list(ev.recibidas)
                elif isinstance(ev, JugadaObservada):
                    self._exigir(mano, "JugadaObservada sin InicioMano")
                    mano.jugadas.append(
                        Jugada(asiento=ev.asiento, carta_id=ev.carta_id, baza=ev.baza)
                    )
                elif isinstance(ev, FinMano):
                    self._exigir(mano, "FinMano sin InicioMano")
                    mano.puntuacion_mano = list(ev.puntuacion)
                elif isinstance(ev, FinPartida):
                    self._exigir(partida, "FinPartida sin InicioPartida")
                    partida.marcador_final = list(ev.marcador)
                    partida.ranking_final = _ranking(ev.marcador)
                    if self.escritor is not None:
                        self.escritor.escribir(partida)
                    if self.al_terminar_partida is not None:
                        self.al_terminar_partida(partida)
                    completadas.append(partida)
                    partida, mano = None, None
                else:  # pragma: no cover - contrato del puerto
                    raise TypeError(f"Evento desconocido: {type(ev).__name__}")
        finally:
            self.adaptador.cerrar()
        return completadas

    @staticmethod
    def _exigir(obj, msg: str) -> None:
        if obj is None:
            raise ValueError(f"Stream de eventos inconsistente: {msg}.")


__all__ = ["RecolectorPartidas"]
