"""
Configuración de recompensas para el entorno Corazones.

Extraído de entorno.py para cumplir Single Responsibility Principle (SRP).
Centraliza todas las constantes de recompensa y la lógica de cálculo
asociada a eventos de baza y fin de mano.

Las recompensas son de suma cero en el largo plazo:
    - Positivas: evitar puntos, shooting the moon, quedar 1º/2º
    - Negativas: ganar corazones/Q♠, quedar 3º/4º
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RewardConfig:
    """Configuración inmutable de todas las constantes de recompensa.

    Usa dataclass para facilitar serialización, comparación y logging.
    """

    # --- Recompensas por evento (casting de cartas) ---
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -10.0
    REWARD_SHOOTING_MOON: float = 50.0

    # --- Recompensa final ---
    REWARD_PRIMERO: float = 500.0
    REWARD_SEGUNDO: float = 200.0
    REWARD_TERCERO: float = -200.0
    REWARD_CUARTO: float = -500.0

    # --- Recompensas densas (reward shaping) ---
    REWARD_NO_GANAR_BAZA_CON_PUNTOS: float = 0.5
    REWARD_DESCARTAR_CORAZON_SEGURO: float = 0.3
    REWARD_DESCARTAR_DAMA_SEGURO: float = 3.0
    REWARD_GANAR_BAZA_SIN_PUNTOS: float = -0.15
    REWARD_PERDER_MANO: float = -2.0
    REWARD_GANAR_MANO: float = 2.0

    # --- Correcciones estratégicas v5 ---
    REWARD_Q_SPADES_SIN_POZO: float = -8.0
    REWARD_GANAR_BAZA_CON_CORAZON: float = -3.0
    REWARD_POR_PUNTO_EN_MANO: float = -0.2

    # --- Umbral de fin de partida ---
    PUNTUACION_MAXIMA: float = 100.0


class CalculadoraRecompensas:
    """Calcula recompensas para eventos dentro de una partida de Corazones.

    Separa la lógica de "qué recompensa dar" de la lógica del entorno
    (ciclo de vida Gymnasium), cumpliendo SRP.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()

    def recompensa_baza_ganada(
        self,
        cartas_baza: list,
        agente_idx: int,
        ganador: int,
        pozo_viable: bool,
    ) -> float:
        """Recompensa cuando el agente ganó la baza.

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            ganador: Índice del ganador de la baza.
            pozo_viable: Si es viable shooting the moon.

        Returns:
            Recompensa acumulada.
        """
        reward = 0.0
        if ganador != agente_idx:
            return 0.0

        gano_corazon = False
        gano_q_spades = False
        puntos_baza = 0

        for c in cartas_baza:
            if c.es_corazon:
                reward += self.cfg.REWARD_CORAZON
                gano_corazon = True
            if c.es_dama_de_picas:
                reward += self.cfg.REWARD_DAMA_PICAS
                gano_q_spades = True
            puntos_baza += c.puntos

        if gano_q_spades and not pozo_viable:
            reward += self.cfg.REWARD_Q_SPADES_SIN_POZO
        if gano_corazon and not pozo_viable:
            reward += self.cfg.REWARD_GANAR_BAZA_CON_CORAZON
        if puntos_baza == 0:
            reward += self.cfg.REWARD_GANAR_BAZA_SIN_PUNTOS

        return reward

    def recompensa_baza_evitada(
        self,
        cartas_baza: list,
        agente_idx: int,
        idx_agente_en_mesa: Optional[int],
    ) -> float:
        """Recompensa cuando el agente NO ganó la baza (evitó puntos).

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            idx_agente_en_mesa: Posición del agente en la mesa, o None.

        Returns:
            Recompensa acumulada.
        """
        reward = 0.0
        puntos_baza = sum(c.puntos for c in cartas_baza)

        if puntos_baza > 0:
            reward += self.cfg.REWARD_NO_GANAR_BAZA_CON_PUNTOS * \
                min(puntos_baza, 3)

        if idx_agente_en_mesa is not None:
            carta_agente = cartas_baza[idx_agente_en_mesa]
            if carta_agente.es_corazon:
                reward += self.cfg.REWARD_DESCARTAR_CORAZON_SEGURO
            if carta_agente.es_dama_de_picas:
                reward += self.cfg.REWARD_DESCARTAR_DAMA_SEGURO

        return reward

    def recompensa_fin_mano(
        self,
        agente_idx: int,
        puntuaciones_mano: List[int],
        pleno_jugador: Optional[int],
    ) -> float:
        """Recompensa al final de una mano.

        Args:
            agente_idx: Índice del agente.
            puntuaciones_mano: Puntos de cada jugador en esta mano.
            pleno_jugador: Índice del jugador que hizo pleno (o None).

        Returns:
            Recompensa.
        """
        reward = 0.0

        if pleno_jugador is None:
            puntos_agente = puntuaciones_mano[agente_idx]
            puntos_otros = [puntuaciones_mano[i]
                            for i in range(4) if i != agente_idx]

            if puntos_agente < min(puntos_otros):
                reward += self.cfg.REWARD_GANAR_MANO
            elif puntos_agente > max(puntos_otros):
                reward += self.cfg.REWARD_PERDER_MANO

            reward += puntos_agente * self.cfg.REWARD_POR_PUNTO_EN_MANO
        elif pleno_jugador == agente_idx:
            reward += self.cfg.REWARD_GANAR_MANO * 2

        return reward

    def recompensa_fin_partida(
        self, agente_idx: int, puntuacion_historica: List[int]
    ) -> float:
        """Recompensa al final de la partida según posición.

        Args:
            agente_idx: Índice del agente.
            puntuacion_historica: Puntuación acumulada de cada jugador.

        Returns:
            Recompensa final.
        """
        ranking = sorted(range(4), key=lambda i: puntuacion_historica[i])
        posicion = ranking.index(agente_idx)

        if posicion == 0:
            return self.cfg.REWARD_PRIMERO
        elif posicion == 1:
            return self.cfg.REWARD_SEGUNDO
        elif posicion == 2:
            return self.cfg.REWARD_TERCERO
        else:
            return self.cfg.REWARD_CUARTO


# Instancia por defecto para compatibilidad
_recompensas = CalculadoraRecompensas()

__all__ = ["RewardConfig", "CalculadoraRecompensas"]
