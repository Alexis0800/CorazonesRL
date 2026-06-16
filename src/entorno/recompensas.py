"""
Configuración de recompensas para el entorno Corazones.

Extraído de entorno.py para cumplir Single Responsibility Principle (SRP).
Centraliza todas las constantes de recompensa y la lógica de cálculo
asociada a eventos de baza y fin de mano.

Cambios v9 (Fase 6):
  - Phase-gating: recompensas densas solo en bazas ≥9
  - REWARD_Q_SPADES_SIN_POZO: -8.0 → -10.0
  - REWARD_DESCARTAR_DAMA_SEGURO: 3.0 → 5.0
  - REWARD_POR_PUNTO_EN_MANO: -0.2 → -0.1
  - Nueva: REWARD_BLOQUEAR_POZO = 15.0
  - Nueva: REWARD_ALIMENTAR_EXITOSO = 10.0
  - Nueva: REWARD_CORAZON_POZO = 1.5 (positivo en modo pozo)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RewardConfig:
    """Configuración inmutable de todas las constantes de recompensa."""

    # --- Recompensas por evento (siempre activas) ---
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -10.0
    REWARD_SHOOTING_MOON: float = 50.0
    REWARD_CORAZON_POZO: float = 1.5       # positivo cuando pozo_viable=True

    # --- Recompensa final ---
    REWARD_PRIMERO: float = 500.0
    REWARD_SEGUNDO: float = 200.0
    REWARD_TERCERO: float = -200.0
    REWARD_CUARTO: float = -500.0

    # --- Recompensas densas (solo en bazas ≥9) ---
    REWARD_NO_GANAR_BAZA_CON_PUNTOS: float = 1.5
    REWARD_DESCARTAR_CORAZON_SEGURO: float = 0.3
    REWARD_DESCARTAR_DAMA_SEGURO: float = 5.0
    REWARD_GANAR_BAZA_SIN_PUNTOS: float = -0.5

    # --- Recompensas de fin de mano ---
    REWARD_PERDER_MANO: float = -2.0
    REWARD_GANAR_MANO: float = 2.0
    REWARD_POR_PUNTO_EN_MANO: float = -0.1

    # --- Correcciones estratégicas (siempre activas) ---
    REWARD_Q_SPADES_SIN_POZO: float = -10.0
    REWARD_GANAR_BAZA_CON_CORAZON: float = -3.0

    # --- Recompensas estratégicas nuevas (v9) ---
    REWARD_BLOQUEAR_POZO: float = 15.0
    REWARD_ALIMENTAR_EXITOSO: float = 10.0

    # --- Umbral de fin de partida ---
    PUNTUACION_MAXIMA: float = 100.0

    # --- Umbral de phase-gating ---
    BAZA_TARDIA: int = 9           # baza ≥9 activa dense rewards
    SCORE_RIVAL_CERCA: int = 85    # umbral para "rival cerca de 100"
    CORAZONES_ALERTA_POZO: int = 10  # corazones capturados para detectar intento de moon


class CalculadoraRecompensas:
    """Calcula recompensas para eventos dentro de una partida de Corazones."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()

    def recompensa_baza_ganada(
        self,
        cartas_baza: list,
        agente_idx: int,
        ganador: int,
        pozo_viable: bool,
        numero_baza: int = 1,
        en_modo_pozo: bool = False,
    ) -> float:
        """Recompensa cuando el agente ganó la baza.

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            ganador: Índice del ganador de la baza.
            pozo_viable: Si es viable shooting the moon.
            numero_baza: Número de baza actual (1-13). Dense rewards solo ≥9.
            en_modo_pozo: Si el agente está ejecutando shooting the moon.
        """
        if ganador != agente_idx:
            return 0.0

        reward = 0.0
        gano_corazon = False
        gano_q_spades = False
        puntos_baza = 0

        for c in cartas_baza:
            if c.es_corazon:
                if pozo_viable or en_modo_pozo:
                    reward += self.cfg.REWARD_CORAZON_POZO  # positivo en pozo
                else:
                    reward += self.cfg.REWARD_CORAZON
                gano_corazon = True
            if c.es_dama_de_picas:
                reward += self.cfg.REWARD_DAMA_PICAS
                gano_q_spades = True
            puntos_baza += c.puntos

        # Penalizaciones estratégicas (siempre activas)
        if gano_q_spades and not pozo_viable:
            reward += self.cfg.REWARD_Q_SPADES_SIN_POZO
        if gano_corazon and not pozo_viable and not en_modo_pozo:
            reward += self.cfg.REWARD_GANAR_BAZA_CON_CORAZON

        # Penalización por ganar baza sin puntos (solo bazas tardías)
        if puntos_baza == 0 and numero_baza >= self.cfg.BAZA_TARDIA:
            reward += self.cfg.REWARD_GANAR_BAZA_SIN_PUNTOS

        return reward

    def recompensa_baza_evitada(
        self,
        cartas_baza: list,
        agente_idx: int,
        idx_agente_en_mesa: Optional[int],
        numero_baza: int = 1,
    ) -> float:
        """Recompensa cuando el agente NO ganó la baza (evitó puntos).

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            idx_agente_en_mesa: Posición del agente en la mesa, o None.
            numero_baza: Número de baza actual (1-13). Dense rewards solo ≥9.
        """
        reward = 0.0
        puntos_baza = sum(c.puntos for c in cartas_baza)
        es_baza_tardia = numero_baza >= self.cfg.BAZA_TARDIA

        # Recompensa por evitar puntos (solo bazas tardías)
        if puntos_baza > 0 and es_baza_tardia:
            reward += self.cfg.REWARD_NO_GANAR_BAZA_CON_PUNTOS * min(puntos_baza, 3)

        # Recompensa por descartes estratégicos (siempre activa para Q♠)
        if idx_agente_en_mesa is not None:
            carta_agente = cartas_baza[idx_agente_en_mesa]
            if carta_agente.es_corazon and es_baza_tardia:
                reward += self.cfg.REWARD_DESCARTAR_CORAZON_SEGURO
            if carta_agente.es_dama_de_picas:  # descarte de Q♠: siempre recompensar
                reward += self.cfg.REWARD_DESCARTAR_DAMA_SEGURO

        return reward

    def recompensa_fin_mano(
        self,
        agente_idx: int,
        puntuaciones_mano: List[int],
        pleno_jugador: Optional[int],
    ) -> float:
        """Recompensa al final de una mano."""
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

    def recompensa_bloquear_pozo(
        self,
        agente_idx: int,
        ganador_q_espadas: Optional[int],
        corazones_por_jugador: List[int],
        pleno_jugador: Optional[int],
    ) -> float:
        """Recompensa si el agente bloqueó un intento de shooting the moon.

        Condición: algún rival acumuló ≥CORAZONES_ALERTA_POZO corazones esta mano
        pero el pozo no fue exitoso, Y el agente fue quien capturó Q♠.

        Args:
            agente_idx: Índice del agente.
            ganador_q_espadas: Quién capturó Q♠ (o None si no fue jugada).
            corazones_por_jugador: Corazones capturados por cada jugador esta mano.
            pleno_jugador: Quién hizo shooting the moon (None si nadie).
        """
        if pleno_jugador is not None:
            return 0.0  # pozo exitoso, no hubo bloqueo

        # ¿Hubo algún rival que intentó el moon (≥ umbral corazones)?
        hay_intento = any(
            j != agente_idx and corazones_por_jugador[j] >= self.cfg.CORAZONES_ALERTA_POZO
            for j in range(4)
        )
        if not hay_intento:
            return 0.0

        # El agente bloqueó si capturó Q♠ (impidió que el shooter acumulara todos los pts)
        if ganador_q_espadas == agente_idx:
            return self.cfg.REWARD_BLOQUEAR_POZO

        return 0.0

    def recompensa_alimentar(
        self,
        agente_idx: int,
        puntuaciones_mano: List[int],
        puntuaciones_historicas: List[int],
    ) -> float:
        """Recompensa si el agente "alimentó" al rival objetivo (cerca de 100).

        Condición: existe un rival con puntuacion_historica ≥ SCORE_RIVAL_CERCA
        Y ese rival recibió puntos esta mano.

        Args:
            agente_idx: Índice del agente.
            puntuaciones_mano: Puntos recibidos esta mano por cada jugador.
            puntuaciones_historicas: Puntuación acumulada ANTES de esta mano.
        """
        for j in range(4):
            if j == agente_idx:
                continue
            if (puntuaciones_historicas[j] >= self.cfg.SCORE_RIVAL_CERCA
                    and puntuaciones_mano[j] > 0):
                return self.cfg.REWARD_ALIMENTAR_EXITOSO

        return 0.0

    def recompensa_fin_partida(
        self, agente_idx: int, puntuacion_historica: List[int]
    ) -> float:
        """Recompensa al final de la partida según posición."""
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
