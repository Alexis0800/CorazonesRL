"""
Bot Lunático — Cazador de pozo (Shooting the Moon).

Arquetipo humano que, con una mano fuerte de corazones altos y control, intenta
capturar los 26 puntos de penalización (13 corazones + Q de picas). Si la mano no
es apta, o si un rival ya capturó puntos (pozo imposible), juega evasivo.

Su valor: (a) como rival y en los rollouts de PIMC, fuerza al agente a aprender
a DEFENDER el pozo — algo que ningún otro bot del pool provoca; los humanos sí
lo intentan (2.52 %/mano cada uno, ver docs/auditoria_moon_2026-07-20.md).
(b) Desde 2026-07-21, su pase constructivo y su `_jugar_moon` son la política
de OFENSIVA de `ModoLunar` (src/agentes/modo_lunar.py).

Strategy Pattern: (motor, jugador_idx, legales) → Carta. Estado por mano
(se reinicia con reset()).
"""
from __future__ import annotations

from typing import List, Optional

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.agentes.heuristicos import bot_evasivo


class BotLunatico:
    """Intenta el pozo con manos fuertes; si no, juega evasivo."""

    def __init__(self) -> None:
        self._modo: Optional[str] = None  # None=indeciso, "MOON" o "NORMAL"

    def reset(self) -> None:
        self._modo = None

    def pasar(self, motor: MotorCorazones, idx: int) -> List[Carta]:
        """Pase acorde al perfil cazador de pozo: CONSERVA cartas altas (control)
        y corazones altos; suelta las cartas BAJAS (que perderían bazas y romperían
        el pozo). Pasa las 3 de menor valor, soltando antes corazones bajos.
        """
        mano = list(motor.jugadores[idx].mano)
        if len(mano) <= 3:
            return mano[:3]
        # Orden de descarte: primero corazones bajos (peligro para el pozo), luego
        # cualquier carta baja. Conservar A/K/Q y corazones altos.
        def _riesgo_para_pozo(c: Carta) -> float:
            base = c.valor  # menor valor = soltar antes
            if c.es_corazon and c.valor <= 6:
                base -= 5  # corazón bajo: soltar cuanto antes (rompería el pozo)
            return base
        return sorted(mano, key=_riesgo_para_pozo)[:3]

    def __call__(self, motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta:
        # Decidir el modo una vez por mano (en la primera decisión, mano ~completa).
        if self._modo is None:
            self._modo = "MOON" if self._moon_viable(motor, idx) else "NORMAL"

        # Si un rival ya capturó puntos, el pozo es imposible → abortar.
        if self._modo == "MOON" and self._otros_tienen_puntos(motor, idx):
            self._modo = "NORMAL"

        if self._modo == "MOON":
            return self._jugar_moon(motor, idx, legales)
        return bot_evasivo(motor, idx, legales)

    # ──────────────────────────────────────────────────────────────

    def _moon_viable(self, motor: MotorCorazones, idx: int) -> bool:
        mano = motor.jugadores[idx].mano
        corazones = [c for c in mano if c.es_corazon]
        altos = [c for c in mano if c.valor >= 12]  # Q/K/A de cualquier palo (control)
        corazones_altos = [c for c in corazones if c.valor >= 11]
        # Mano apta: muchos corazones, varios altos de control y corazones altos.
        return len(corazones) >= 5 and len(altos) >= 3 and len(corazones_altos) >= 2

    def _otros_tienen_puntos(self, motor: MotorCorazones, idx: int) -> bool:
        return any(
            i != idx and motor.jugadores[i].contar_puntos_bazas() > 0
            for i in range(4)
        )

    def _jugar_moon(self, motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta:
        # Liderar: jugar alto para ganar la baza y mantener el control.
        if not motor.mesa:
            return max(legales, key=lambda c: c.valor)

        palo = motor.palo_de_salida
        mismo_palo = [c for c in legales if c.palo == palo]
        if mismo_palo:
            ganadora = self._carta_ganadora(motor)
            # Ganar con el mínimo que supere a la ganadora actual (capturar puntos).
            if ganadora is not None:
                gana = [c for c in mismo_palo if c.valor > ganadora.valor]
                if gana:
                    return min(gana, key=lambda c: c.valor)
            # No puedo ganar: soltar la más baja (preservar altas para ganar luego).
            return min(mismo_palo, key=lambda c: c.valor)

        # Void: no puedo seguir; descartar la más baja sin tirar mis cartas de control.
        return min(legales, key=lambda c: c.valor)

    def _carta_ganadora(self, motor: MotorCorazones) -> Optional[Carta]:
        palo = motor.palo_de_salida
        ganadora: Optional[Carta] = None
        for _, c in motor.mesa:
            if c.palo == palo and (ganadora is None or c.valor > ganadora.valor):
                ganadora = c
        return ganadora


__all__ = ["BotLunatico"]
