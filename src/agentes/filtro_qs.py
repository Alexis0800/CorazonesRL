"""
Filtro Q♠ — evita comer la Dama de Picas EVITABLE (Fase 1a del plan
docs/plan_mejora_vs_humanos_2026-07-25.md).

Evidencia (dos métodos independientes, 528 Q♠ comidas en 842 partidas reales):
el 14-15 % eran evitables por sustitución directa — jugamos una carta que
ganaba la baza con la Q♠ en mesa teniendo una legal que perdía, jugamos
nuestra propia Q♠ pudiendo no hacerlo, o la lideramos sin que K/A♠ hubieran
salido. Las manos grandes del bot SON la Q♠ (98.8 %).

Diseño: el filtro NO elige cartas — RESTRINGE el conjunto legal y el campeón
elige entre lo permitido (patrón `recomendar_jugada_entre`). Devuelve la misma
lista si no aplica. Nunca deja la lista vacía.

Reglas:
  R1  Q♠ en la mesa y existe legal que NO gana la baza → vetar las ganadoras.
  R2  Liderando: NUNCA liderar la propia Q♠ habiendo alternativa. (Con K/A♠
      ya fuera, liderarla es autocomerse 13 GARANTIZADO — es la pica más alta
      restante; con K/A vivas sigue siendo la peor salida común. La descarga
      buena de la Q es off-suit o bajo K/A en mesa, nunca liderándola.)
  R3  Siguiendo picas: si nuestra Q♠ ganaría la baza y hay alternativa,
      vetarla. (Si la Q♠ PIERDE — K/A♠ en mesa — se conserva: es descarga.)
  Descarte off-suit de la Q♠: nunca se veta (regalarla es bueno).
  Excepción endgame ESTRECHA: con cualquier marcador ≥ 87 el filtro NO aplica
  (solo ahí comer 13 a propósito es plausible — rematar a un rival al borde
  de 100; el campeón entrenó el marcador, el filtro no). La primera versión
  usaba ≥74 y regalaba un tercio de la cobertura.
"""
from __future__ import annotations

from typing import List

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones

_PICA = 2
_QS_ID = 36
_KS_VALOR, _AS_VALOR = 13, 14
_UMBRAL_ENDGAME = 87


class FiltroQS:
    """Restringe las jugadas legales para no comer la Q♠ evitable."""

    def __init__(self) -> None:
        self.stats = {"intervenciones": 0, "consultas": 0}

    def filtrar(self, motor: MotorCorazones, idx: int,
                legales: List[Carta]) -> List[Carta]:
        self.stats["consultas"] += 1
        resultado = self._filtrar(motor, idx, legales)
        if len(resultado) < len(legales):
            self.stats["intervenciones"] += 1
        return resultado

    def _filtrar(self, motor: MotorCorazones, idx: int,
                 legales: List[Carta]) -> List[Carta]:
        if len(legales) <= 1:
            return legales
        # Excepción endgame: región terminal → criterio del campeón intacto.
        if any(s >= _UMBRAL_ENDGAME for s in motor.puntuaciones_historicas()):
            return legales

        mesa = motor.mesa
        palo_salida = motor.palo_de_salida

        # R2: liderando, NUNCA la propia Q♠ habiendo alternativa (con K/A♠
        # fuera es autocomida garantizada; con K/A vivas, la peor salida común).
        if not mesa:
            if any(c.id == _QS_ID for c in legales):
                resto = [c for c in legales if c.id != _QS_ID]
                if resto:
                    return resto
            return legales

        # Siguiendo una baza en curso:
        max_salida = max((c.valor for _, c in mesa if c.palo == palo_salida),
                         default=0)

        def gana(c: Carta) -> bool:
            return c.palo == palo_salida and c.valor > max_salida

        qs_en_mesa = any(c.id == _QS_ID for _, c in mesa)

        # R1: Q♠ en mesa → vetar toda carta que gane la baza, si hay perdedoras.
        if qs_en_mesa:
            pierden = [c for c in legales if not gana(c)]
            if pierden:
                return pierden
            return legales

        # R3: siguiendo picas con la Q♠ en mano.
        if palo_salida == _PICA:
            qs = next((c for c in legales if c.id == _QS_ID), None)
            if qs is not None and gana(qs):
                resto = [c for c in legales if c.id != _QS_ID]
                if resto:
                    return resto
        return legales

__all__ = ["FiltroQS"]
