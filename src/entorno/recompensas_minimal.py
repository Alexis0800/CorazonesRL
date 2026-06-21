"""
Sistema de recompensas minimal (v13_minimal) — Strategy Pattern.

Solo 3 señales fundamentales:
  1. Puntos capturados: -1 por corazón, -13 por Q♠ (señal densa natural)
  2. Reward de distancia: cambio de ventaja ponderado por cercanía
  3. Shooting moon: +26 base (la distancia complementa)

Sin señales tácticas densas. La simplicidad permite que el value head
de PPO aprenda consistentemente.

Separado de recompensas.py para mantener ambos sistemas independientes
y facilitar experimentación (Open/Closed Principle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RewardConfigMinimal:
    """Configuración inmutable de recompensas minimal.

    Solo 3 señales fundamentales. Sin recompensas tácticas.
    """

    # --- Señales por evento (densas, naturales) ---
    REWARD_CORAZON: float = -1.0       # cada corazón capturado
    REWARD_DAMA_PICAS: float = -13.0   # Q♠ capturada
    REWARD_SHOOTING_MOON: float = 26.0  # base por shooting the moon

    # --- Señal por distancia (al final de cada mano) ---
    ESCALA_DISTANCIA: float = 5.0       # factor de escala para reward_distancia

    # --- Fin de partida ---
    PUNTUACION_MAXIMA: float = 100.0


def calcular_reward_distancia(
    puntuacion_antes: List[int],
    puntuacion_despues: List[int],
    agente_idx: int,
    escala: float = 5.0,
) -> float:
    """Calcula el reward basado en cambio de ventaja ponderado por cercanía.

    Fórmula:
      Para cada rival:
        ventaja_antes  = mi_puntuacion_antes  - rival_puntuacion_antes
        ventaja_despues = mi_puntuacion_despues - rival_puntuacion_despues
        cambio = ventaja_despues - ventaja_antes  # negativo = mejoré
        peso = 1 / (|ventaja_antes| + 1)          # +cerca = +importante
        contrib = -cambio × peso                   # mejorar → positivo

      reward = suma(contrib) × escala

    Args:
        puntuacion_antes: Puntuación acumulada de cada jugador ANTES de la mano.
        puntuacion_despues: Puntuación acumulada DESPUÉS de la mano.
        agente_idx: Índice del agente.
        escala: Factor de escala para la recompensa total.

    Returns:
        Recompensa de distancia (positivo = mejoró posición relativa).
    """
    dist = 0.0
    for rival in range(4):
        if rival == agente_idx:
            continue
        v_antes = puntuacion_antes[agente_idx] - puntuacion_antes[rival]
        v_despues = puntuacion_despues[agente_idx] - puntuacion_despues[rival]
        # más negativo = mejoré (me alejé, bajé puntos)
        cambio = v_despues - v_antes
        peso = 1.0 / (abs(v_antes) + 1.0)
        dist += -cambio * peso  # signo: mejorar → positivo

    return dist * escala


class CalculadoraRecompensasMinimal:
    """Calculadora de recompensas minimal — Strategy Pattern.

    Proporciona la interfaz que CorazonesEnv necesita para calcular
    recompensas, pero con solo 3 señales fundamentales.
    """

    def __init__(self, config: Optional[RewardConfigMinimal] = None):
        self.cfg = config or RewardConfigMinimal()

    def recompensa_baza(
        self,
        cartas_baza: list,
        agente_idx: int,
        ganador: int,
        pozo_viable: bool = False,
    ) -> float:
        """Recompensa por ganar/perder una baza.

        Minimal: solo suma los puntos de las cartas capturadas.
        Sin penalizaciones tácticas adicionales.

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            ganador: Índice del ganador de la baza.
            pozo_viable: Si es viable shooting the moon (ignorado en minimal).
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

    def recompensa_shooting_moon(
        self,
        agente_idx: int,
        pleno_jugador: Optional[int],
    ) -> float:
        """Recompensa por shooting the moon.

        Args:
            agente_idx: Índice del agente.
            pleno_jugador: Quién hizo moon (None si nadie).
        """
        if pleno_jugador == agente_idx:
            return self.cfg.REWARD_SHOOTING_MOON
        return 0.0

    def recompensa_distancia(
        self,
        puntuacion_antes: List[int],
        puntuacion_despues: List[int],
        agente_idx: int,
    ) -> float:
        """Recompensa de distancia al final de cada mano.

        Args:
            puntuacion_antes: Puntuación ANTES de esta mano.
            puntuacion_despues: Puntuación DESPUÉS de esta mano.
            agente_idx: Índice del agente.
        """
        return calcular_reward_distancia(
            puntuacion_antes, puntuacion_despues,
            agente_idx, self.cfg.ESCALA_DISTANCIA,
        )

    def recompensa_fin_partida(
        self,
        agente_idx: int,
        puntuacion_historica: List[int],
    ) -> float:
        """Recompensa terminal al final de la partida.

        Minimal: la distancia ya captura el ranking. Solo damos una señal
        binaria de ganar/perder para reforzar el objetivo final.

        Args:
            agente_idx: Índice del agente.
            puntuacion_historica: Puntuación final de cada jugador.
        """
        ranking = sorted(range(4), key=lambda i: puntuacion_historica[i])
        posicion = ranking.index(agente_idx)
        if posicion == 0:
            return 10.0
        elif posicion == 3:
            return -10.0
        return 0.0


__all__ = [
    "RewardConfigMinimal",
    "CalculadoraRecompensasMinimal",
    "calcular_reward_distancia",
]
