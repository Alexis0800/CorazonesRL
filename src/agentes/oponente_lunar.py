"""
Oponente Lunar — rival de entrenamiento que corona lunas al estilo humano.

Los humanos lunean ~2.5 %/mano por rival y mayormente se comprometen MID-MANO
(oportunista). El pool solo tenía BotLunatico, que decide desde el pase (estilo
equivocado) — el agente nunca sintió presión de luna realista. Este oponente
compone ModoLunar (gate aprendido + compromiso dinámico + aborts) sobre una
base BotExperto: cuando ModoLunar devuelve None, juega el experto.

Serialización Ray: el pool serializa los oponentes hacia los workers y
EstimadorMoonProb contiene redes torch. Aquí se guarda solo el dir de pesos y
el ModoLunar se construye lazy en el primer uso (patrón
SnapshotPolicy._get_model); __getstate__ nunca serializa torch.

Firma de bot del pool: __call__(motor, idx, legales) -> Carta,
pasar(motor, idx) -> List[Carta], reset() por mano.
"""
from __future__ import annotations

from typing import List, Optional

from src.agentes.bot_experto import BotExperto
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones


class OponenteLunar:
    """ModoLunar + BotExperto por composición: luna oportunista o juego experto."""

    def __init__(
        self,
        moon_dir: Optional[str] = None,   # None = RUTA_MOON (lazy, sin importar torch)
        umbral_pase: float = 0.10,
        umbral_juego: float = 0.30,
        umbral_abort: float = 0.10,
    ):
        self._moon_dir = moon_dir
        self._umbral_pase = umbral_pase
        self._umbral_juego = umbral_juego
        self._umbral_abort = umbral_abort
        self._base = BotExperto()
        self._modo = None  # ModoLunar lazy: contiene redes torch (no serializables)

    # ------------------------------------------------------------------

    def _get_modo(self):
        """Construye ModoLunar (y las redes torch) la primera vez (lazy)."""
        if self._modo is None:
            from src.agentes.modo_lunar import ModoLunar
            from src.entorno.moon_model import EstimadorMoonProb, RUTA_MOON
            self._modo = ModoLunar(
                EstimadorMoonProb(self._moon_dir or RUTA_MOON),
                umbral_pase=self._umbral_pase,
                umbral_juego=self._umbral_juego,
                umbral_abort=self._umbral_abort,
            )
        return self._modo

    def reset(self) -> None:
        """Reset por mano (lo llama el env en _reset_oponentes_por_mano).

        ModoLunar ya auto-detecta mano nueva, pero el reset explícito es más
        robusto. BotExperto se auto-resetea al detectar mano nueva (sin reset()).
        """
        if self._modo is not None:
            self._modo.nueva_partida()

    def pasar(self, motor: MotorCorazones, idx: int) -> List[Carta]:
        """Pase constructivo de luna si la mano PRE-pase promete; si no, experto."""
        cartas = self._get_modo().elegir_pase(motor, idx)
        if cartas is not None:
            return cartas
        return self._base.pasar(motor, idx)

    def __call__(self, motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta:
        carta = self._get_modo().elegir_jugada(motor, idx, legales)
        if carta is not None:
            return carta
        return self._base(motor, idx, legales)

    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_modo"] = None  # nunca serializar las redes torch; se reconstruye lazy
        return state


__all__ = ["OponenteLunar"]
