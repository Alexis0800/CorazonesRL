"""
Bot Castigador — Oponente que explota debilidades con Q♠.

Diseñado como oponente de self-play para forzar al agente RL a aprender
a gestionar Q♠ correctamente. El análisis PIMC reveló que el error #1
del modelo v3 es jugar Q♠ voluntariamente (coste +15.2 pts).

Estrategia del BotCastigador:
  1. **Lidera ♠** cuando Q♠ está activa y no la tiene él mismo.
     Juega la ♠ más alta disponible para maximizar la presión.
  2. **Sigue ♠ agresivamente** (carta más alta) para forzar al portador
     de Q♠ a jugarla o ganar la baza con riesgo.
  3. **Si Q♠ ya está en la mesa**, evita ganar la baza (juega bajo).
  4. **En otros palos**, juega conservador (minimizar).
  5. **Si está void**, descarta corazones para "pintar" y castigar.

Strategy Pattern: implementa la misma firma que los bots heurísticos
    (motor, jugador_idx, legales) → Carta
"""

from __future__ import annotations

from typing import List, Set, Optional, Dict

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones

_TREBOL, _DIAMANTE, _PICA, _CORAZON = 0, 1, 2, 3


class BotCastigador:
    """Bot que presiona al portador de Q♠ liderando y siguiendo ♠ agresivamente.

    Mantiene estado intramano mínimo:
      - _vacios: jugador → set de palos donde es void (inferido por descarte).
      - _ultimo_baza_num: para detectar cambio de baza y resetear en nueva mano.
      - _palo_salida_baza_actual: palo de la baza en curso.

    Se reinicia automáticamente al comenzar una nueva mano.
    """

    def __init__(self) -> None:
        self._reset_estado()

    # ──────────────────────────────────────────────────────────────
    # Estado intramano
    # ──────────────────────────────────────────────────────────────

    def _reset_estado(self) -> None:
        """Reinicia el estado al inicio de cada mano."""
        self._vacios: Dict[int, Set[int]] = {i: set() for i in range(4)}
        self._ultimo_baza_num: int = 0
        self._palo_salida_baza_actual: Optional[int] = None

    # ──────────────────────────────────────────────────────────────
    # Interfaz Strategy
    # ──────────────────────────────────────────────────────────────

    def __call__(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
    ) -> Carta:
        """Selecciona la mejor carta según la estrategia castigadora."""
        self._actualizar_estado(motor, idx)

        if not motor.mesa:
            carta = self._liderar(motor, idx, legales)
        else:
            palo = motor.palo_de_salida
            mismo_palo = [c for c in legales if c.palo == palo]
            if mismo_palo:
                carta = self._seguir_palo(motor, idx, mismo_palo)
            else:
                carta = self._descartar(motor, idx, legales)

        # Registrar palo de salida si lideramos
        if not motor.mesa:
            self._palo_salida_baza_actual = carta.palo

        return carta

    # ──────────────────────────────────────────────────────────────
    # Actualización de estado
    # ──────────────────────────────────────────────────────────────

    def _actualizar_estado(self, motor: MotorCorazones, idx: int) -> None:
        """Actualiza voids inferidos y detecta nueva mano."""
        # Detectar nueva mano
        if motor.numero_baza == 1 and self._ultimo_baza_num > 1:
            self._reset_estado()

        # Nueva baza: inferir voids del turno anterior
        if motor.numero_baza > self._ultimo_baza_num:
            self._ultimo_baza_num = motor.numero_baza
            self._palo_salida_baza_actual = motor.palo_de_salida

        # Actualizar palo de salida
        if motor.palo_de_salida is not None:
            self._palo_salida_baza_actual = motor.palo_de_salida

        # Inferir voids desde los jugadores que ya jugaron en ESTA baza
        palo = motor.palo_de_salida
        if palo is not None and motor.mesa:
            starter_baza = motor.mesa[0][0]
            for jug_idx, carta in motor.mesa:
                if jug_idx != starter_baza and carta.palo != palo:
                    self._vacios[jug_idx].add(palo)

    # ──────────────────────────────────────────────────────────────
    # Consultas de estado
    # ──────────────────────────────────────────────────────────────

    def _qs_activa(self, motor: MotorCorazones) -> bool:
        """True si Q♠ no ha sido capturada todavía."""
        for j in motor.jugadores:
            if any(c.es_dama_de_picas for c in j.bazas_ganadas):
                return False
        return True

    def _tengo_qs(self, motor: MotorCorazones, idx: int) -> bool:
        """True si el bot tiene Q♠ en su mano."""
        return any(c.es_dama_de_picas for c in motor.jugadores[idx].mano)

    # ──────────────────────────────────────────────────────────────
    # Lógica de decisión
    # ──────────────────────────────────────────────────────────────

    def _liderar(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
    ) -> Carta:
        """Decide qué carta liderar.

        Estrategia castigadora:
          1. Si Q♠ activa y no la tengo → liderar ♠ más alta.
          2. Si no, jugar conservador (carta más baja sin puntos).
        """
        # ── Castigo: liderar ♠ para forzar Q♠ ──
        if self._qs_activa(motor) and not self._tengo_qs(motor, idx):
            picas = [c for c in legales if c.palo == _PICA]
            if picas:
                return max(picas, key=lambda c: c.valor)

        # ── Juego conservador ──
        sin_puntos = [c for c in legales if c.puntos == 0]
        if sin_puntos:
            return min(sin_puntos, key=lambda c: c.valor)
        return min(legales, key=lambda c: c.puntos * 100 + c.valor)

    def _seguir_palo(
        self,
        motor: MotorCorazones,
        idx: int,
        mismo_palo: List[Carta],
    ) -> Carta:
        """Decide qué carta jugar siguiendo el palo de salida.

        En ♠ con Q♠ activa: jugar la más alta para presionar.
        Si Q♠ ya está en la mesa: jugar la más baja para no ganar.
        En otros palos: conservador.
        """
        palo_salida = motor.palo_de_salida

        # ── Siguiendo ♠ con Q♠ activa ──
        if palo_salida == _PICA and self._qs_activa(motor):
            qs_en_mesa = any(c.es_dama_de_picas for _, c in motor.mesa)

            if qs_en_mesa:
                # Q♠ ya en mesa → NO quiero ganar → jugar la más baja
                return min(mismo_palo, key=lambda c: c.valor)
            else:
                # Q♠ aún no jugada → presionar con la más alta
                return max(mismo_palo, key=lambda c: c.valor)

        # ── Siguiendo otros palos: conservador ──
        return min(mismo_palo, key=lambda c: c.valor)

    def _descartar(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
    ) -> Carta:
        """Decide qué carta descartar cuando estamos void en el palo de salida.

        Estrategia:
          1. Si tenemos Q♠, descartarla si no es ♠ (dump estratégico).
          2. Si podemos "pintar" (descartar corazones), hacerlo para castigar.
          3. Si no, descartar la carta más alta (deshacerse de liabilities).
        """
        # ── Descartar Q♠ si es legal (no es palo de salida) ──
        mi_qs = [c for c in legales if c.es_dama_de_picas]
        if mi_qs:
            return mi_qs[0]

        # ── Pintar: descartar corazones para castigar ──
        corazones = [c for c in legales if c.es_corazon]
        if corazones:
            # Descartar el corazón más alto (más daño)
            return max(corazones, key=lambda c: c.valor)

        # ── Sin corazones: deshacerse de la carta más alta ──
        return max(legales, key=lambda c: c.valor)
