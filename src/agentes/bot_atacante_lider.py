"""
Bot Atacante del Líder — el "acuerdo no hablado" humano.

Arquetipo de Estrategias Avanzadas §1: el foco colectivo contra el jugador con
MENOS puntos (el líder). Juega evasivo para sí mismo, pero cuando debe ceder
puntos (descarte estando void), se los dirige al líder actual de la mesa, y evita
dárselos a un rival rezagado.

Valor: introduce dinámica de marcador dependiente del estado de la partida que
ningún heurístico simple modela. Útil como rival diverso y en rollouts.

Strategy Pattern: (motor, jugador_idx, legales) → Carta. Sin estado por mano.
"""
from __future__ import annotations

from typing import List, Optional

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.agentes.heuristicos import bot_evasivo

_PICA, _CORAZON = 2, 3


class BotAtacanteLider:
    """Evasivo, pero dirige los puntos que cede hacia el líder (menor score)."""

    def reset(self) -> None:  # interfaz uniforme (sin estado, no-op)
        pass

    def pasar(self, motor: MotorCorazones, idx: int) -> List[Carta]:
        """Pase acorde al perfil: si el RECEPTOR (por rotación) es el líder de la
        partida, le pasa lo más peligroso (Q♠/picas altas/corazones altos) para
        cargarlo. Si no, suelta sus propios liabilities (como evasivo).
        """
        from src.agentes.pase import pase_heuristico
        mano = list(motor.jugadores[idx].mano)
        if len(mano) <= 3:
            return mano[:3]

        receptor = motor.receptor_pase(idx)
        if receptor is None:
            return pase_heuristico(motor, idx)
        lider = min((i for i in range(4) if i != idx),
                    key=lambda i: motor.jugadores[i].puntuacion_historica)

        if receptor == lider:
            # Cargar al líder: Q♠ > picas altas > corazones altos > altas.
            def _danio(c: Carta) -> float:
                if c.es_dama_de_picas:
                    return 100.0
                if c.palo == _PICA and c.valor >= 13:
                    return 50.0 + c.valor
                if c.es_corazon:
                    return 20.0 + c.valor
                return float(c.valor)
            return sorted(mano, key=_danio, reverse=True)[:3]

        # El receptor no es el líder → no armarlo; soltar liabilities propios.
        return pase_heuristico(motor, idx)

    def __call__(self, motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta:
        # Líder = rival con MENOS puntos acumulados (el que va ganando la partida).
        rivales = [i for i in range(4) if i != idx]
        lider = min(rivales, key=lambda i: motor.jugadores[i].puntuacion_historica)

        palo = motor.palo_de_salida
        void_en_palo = bool(motor.mesa) and palo is not None and \
            not any(c.palo == palo for c in legales)

        # Solo podemos "dirigir" puntos al descartar estando void: ahí elegimos
        # qué carta soltar sabiendo quién ganará la baza.
        if void_en_palo:
            ganador = self._ganador_actual(motor)
            puntos = [c for c in legales if c.puntos > 0]
            sin_puntos = [c for c in legales if c.puntos == 0]
            if ganador == lider and puntos:
                # El líder se lleva la baza → cargarle el máximo de puntos.
                return max(puntos, key=lambda c: c.puntos * 100 + c.valor)
            # No es el líder quien gana → no alimentar: soltar alto sin puntos.
            if sin_puntos:
                return max(sin_puntos, key=lambda c: c.valor)
            # Forzado a soltar puntos a un no-líder → minimizar el daño cedido.
            return min(legales, key=lambda c: c.puntos * 100 + c.valor)

        # En el resto de casos, juego evasivo estándar (no comer puntos uno mismo).
        return bot_evasivo(motor, idx, legales)

    def _ganador_actual(self, motor: MotorCorazones) -> Optional[int]:
        palo = motor.palo_de_salida
        ganadora: Optional[Carta] = None
        ganador: Optional[int] = None
        for jug_idx, c in motor.mesa:
            if c.palo == palo and (ganadora is None or c.valor > ganadora.valor):
                ganadora = c
                ganador = jug_idx
        return ganador


__all__ = ["BotAtacanteLider"]
