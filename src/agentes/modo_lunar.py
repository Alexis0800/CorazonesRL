"""
Modo lunar — capa de OFENSIVA de pozo compuesta sobre el campeón.

La auditoría de ofensiva (docs/auditoria_moon_2026-07-20.md, §2026-07-21) mostró
que el campeón casi no lunea en ningún contexto (0.27 %/mano vs 2.52 % de cada
humano) y que la brecha tiene dos componentes: el pase no construye manos de
luna (rivales ×3.6 con el pase, nosotros ×0.9) y falta persecución en-mano
(0.82 vs 0.29 % en manos hold). Esta clase ataca ambas por COMPOSICIÓN, sin RL:

  - Pase: si P(luna) pre-pase ≥ umbral_pase → pase constructivo de BotLunatico
    (conserva controles altos, suelta bajas). Si no → None (pase del campeón).
  - Juego: en la primera decisión de la mano, si P(luna) ≥ umbral_juego →
    comprometerse y jugar como BotLunatico._jugar_moon; se aborta con el gate
    duro (otro jugador capturó puntos) → None (vuelve el campeón).

El gate de P(luna) usa EstimadorMoonProb con trackers vacíos — exactamente la
configuración validada sobre datos reales (AUC 0.945 val al inicio de mano).

Uso (el llamador decide qué hacer con None = "no aplica, política normal"):

    modo = ModoLunar(EstimadorMoonProb())
    cartas = modo.elegir_pase(motor, idx)          # None o [c1, c2, c3]
    carta = modo.elegir_jugada(motor, idx, legales)  # None o Carta
"""
from __future__ import annotations

from typing import List, Optional

from src.agentes.bot_lunatico import BotLunatico
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.moon_model import EstimadorMoonProb


class ModoLunar:
    """Ofensiva de luna por composición: gate aprendido + política heurística."""

    def __init__(
        self,
        estimador: EstimadorMoonProb,
        umbral_pase: float = 0.10,
        umbral_juego: float = 0.30,  # barrido 300 partidas: 0.15→win plano, 0.30→+8.7pp, 0.50→+7pp
    ):
        self._estimador = estimador
        self._umbral_pase = umbral_pase
        self._umbral_juego = umbral_juego
        self._bot = BotLunatico()

        # Estado por mano
        self._comprometida = False       # persiguiendo el pozo esta mano
        self._pase_ofensivo = False      # esta mano se pasó en modo constructivo
        self._decidido_juego = False     # ya se evaluó el compromiso de juego
        self._cartas_vistas_prev = 0     # para detectar el inicio de mano nueva

        # Estadísticas acumuladas (para la evaluación A/B)
        self.stats = {
            "pases_ofensivos": 0,
            "manos_comprometidas": 0,
            "abortos_gate": 0,
        }

    # ------------------------------------------------------------------

    def nueva_partida(self) -> None:
        self._reset_mano()
        self._cartas_vistas_prev = 0

    def _reset_mano(self) -> None:
        self._comprometida = False
        self._pase_ofensivo = False
        self._decidido_juego = False
        self._bot.reset()

    def _prob_luna(self, motor: MotorCorazones, idx: int) -> float:
        """P(luna propia) con trackers vacíos (config validada de inicio de mano)."""
        return self._estimador.propio(
            motor=motor,
            agente_idx=idx,
            vacios=[set() for _ in range(4)],
            historial=[],
            cartas_dadas=[],
            cartas_recibidas=[],
            puntuacion_historica=list(motor.puntuaciones_historicas()),
            puntos_mano_actual=[j.contar_puntos_bazas() for j in motor.jugadores],
            dama_picas_en=None,
        )

    def _gate_vivo(self, motor: MotorCorazones, idx: int) -> bool:
        """El pozo sigue posible: ningún OTRO jugador capturó puntos."""
        return not any(
            i != idx and motor.jugadores[i].contar_puntos_bazas() > 0
            for i in range(4)
        )

    # ------------------------------------------------------------------

    def elegir_pase(self, motor: MotorCorazones, idx: int) -> Optional[List[Carta]]:
        """Pase constructivo si la mano pre-pase promete luna; None si no aplica."""
        self._reset_mano()  # el pase siempre abre una mano nueva
        self._cartas_vistas_prev = 0
        if self._prob_luna(motor, idx) < self._umbral_pase:
            return None
        self._pase_ofensivo = True
        self.stats["pases_ofensivos"] += 1
        return self._bot.pasar(motor, idx)

    def elegir_jugada(
        self, motor: MotorCorazones, idx: int, legales: List[Carta]
    ) -> Optional[Carta]:
        """Carta de persecución de pozo, o None → que juegue el campeón."""
        # Detectar mano nueva sin pase (hold): el total de cartas jugadas cayó.
        vistas = sum(len(j.bazas_ganadas) for j in motor.jugadores) + len(motor.mesa)
        if vistas < self._cartas_vistas_prev:
            self._reset_mano()
        self._cartas_vistas_prev = vistas

        # Compromiso: una sola vez por mano, en nuestra primera decisión
        # (mano completa de 13, post-pase si lo hubo).
        if not self._decidido_juego and len(motor.jugadores[idx].mano) == 13:
            self._decidido_juego = True
            if self._prob_luna(motor, idx) >= self._umbral_juego:
                self._comprometida = True
                self.stats["manos_comprometidas"] += 1

        if not self._comprometida:
            return None

        if not self._gate_vivo(motor, idx):
            # Pozo imposible → abortar de forma definitiva esta mano.
            self._comprometida = False
            self.stats["abortos_gate"] += 1
            return None

        return self._bot._jugar_moon(motor, idx, legales)

    @property
    def comprometida(self) -> bool:
        """La mano en curso está en persecución de pozo (para métricas)."""
        return self._comprometida


__all__ = ["ModoLunar"]
