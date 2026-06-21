"""
Sistema de recompensas para v3: score + distancia.

Extiende el sistema score-based de v2_ronda con un componente
de distancia al lider, que codifica el contexto multi-mano.

Señales planificadas:
  1. Score propio (heredado de v2): minimizar puntos
  2. Delta al lider: bonus/penalizacion por distancia al primer puesto
  3. Presion de final de partida: incentivos cerca de 100 puntos

En desarrollo. La implementacion actual es el placeholder score-based
de v2, que se refinara con la formula de distancia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RewardConfigScore:
    """Configuracion inmutable de recompensas score-based (v2).

    Placeholder inicial para v3. Se extendera con constantes de distancia.
    """

    # --- Senales por baza (shaping denso) ---
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -13.0

    # --- Senales de fin de mano ---
    REWARD_SHOOTING_MOON: float = 78.0
    REWARD_MEJOR_MANO: float = 5.0
    REWARD_PEOR_MANO: float = -5.0


class CalculadoraRecompensasScore:
    """Calculadora de recompensas score-based — Strategy Pattern.

    Placeholder inicial para v3. Proporciona la interfaz que el entorno
    necesita. Se extendera con calculos de distancia.
    """

    def __init__(self, config: Optional[RewardConfigScore] = None):
        self.cfg = config or RewardConfigScore()

    # ------------------------------------------------------------------
    # Per-baza
    # ------------------------------------------------------------------

    def recompensa_baza(
        self,
        cartas_baza: list,
        agente_idx: int,
        ganador: int,
    ) -> float:
        """Recompensa por ganar/perder una baza.

        Solo suma los puntos de cartas capturadas. Sin tacticas.

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Indice del agente.
            ganador: Indice del ganador de la baza.

        Returns:
            Suma de penalizaciones por ♡ y Q♠, o 0 si no gano el agente.
        """
        if ganador != agente_idx:
            return 0.0

        reward = 0.0
        for c in cartas_baza:
            if c.es_corazon:
                reward += self.cfg.REWARD_CORAZON
            if c.es_dama_de_picas:
                reward += self.cfg.REWARD_DAMA_PICAS
        return reward

    # ------------------------------------------------------------------
    # Fin de mano
    # ------------------------------------------------------------------

    def recompensa_fin_mano(self, mis_puntos: int) -> float:
        """Recompensa principal al final de la mano.

        Formula: 26 - mis_puntos  (rango [0, 26])

        Args:
            mis_puntos: Puntos tomados por el agente en esta mano.

        Returns:
            Reward en [0, 26]. 26 = mano perfecta, 0 = desastre.
        """
        return 26.0 - float(mis_puntos)

    def recompensa_shooting_moon(
        self,
        agente_idx: int,
        pleno_jugador: Optional[int],
    ) -> float:
        """Recompensa por shooting the moon.

        Args:
            agente_idx: Indice del agente.
            pleno_jugador: Quien hizo moon (None si nadie).

        Returns:
            78.0 si el agente hizo moon, 0.0 en otro caso.
        """
        if pleno_jugador == agente_idx:
            return self.cfg.REWARD_SHOOTING_MOON
        return 0.0

    def recompensa_posicion(
        self,
        agente_idx: int,
        puntuaciones_mano: List[int],
    ) -> float:
        """Bonus ligero por posicion relativa en la mano.

        Args:
            agente_idx: Indice del agente.
            puntuaciones_mano: Puntos de cada jugador en esta mano.

        Returns:
            +5 si fue el mejor (menos puntos), -5 si fue el peor, 0 en medio.
        """
        mi_pts = puntuaciones_mano[agente_idx]
        min_pts = min(puntuaciones_mano)
        max_pts = max(puntuaciones_mano)

        if mi_pts == min_pts and min_pts < max_pts:
            return self.cfg.REWARD_MEJOR_MANO
        if mi_pts == max_pts and max_pts > min_pts:
            return self.cfg.REWARD_PEOR_MANO
        return 0.0
