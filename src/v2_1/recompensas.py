"""
Sistema de recompensas mejorado (v2_1).

Extiende el sistema score-based de v2_ronda con 4 señales tácticas nuevas
que enseñan comportamientos estratégicos del BotExperto:

  1. Q♠ dump: +12 cuando descartas Q♠ siendo void y un rival gana la baza
  2. Moon block: +15 cuando ganas baza con puntos y un rival tiene ≥6♥
  3. Early safe burn: +1.5 por ganar baza con 0 puntos en bazas 1-7
  4. Liability hold: -5 tras baza 7 si retienes A♠/K♠ con Q♠ activa

Las señales base (v2_ronda) se mantienen intactas. Las nuevas son aditivas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RewardConfigV21:
    """Configuración inmutable de recompensas v2_1.

    Incluye las 5 constantes base de v2_ronda + 4 nuevas señales tácticas.
    """

    # --- Señales base (v2_ronda) ---
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -13.0
    REWARD_SHOOTING_MOON: float = 78.0
    REWARD_MEJOR_MANO: float = 5.0
    REWARD_PEOR_MANO: float = -5.0

    # --- Señales tácticas nuevas (v2_1) ---
    # Dump Q♠ en rival cuando somos void en palo de salida
    REWARD_QS_DUMP: float = 12.0
    # Bloquear pozo ajeno: ganar baza con puntos cuando rival tiene ≥6♥
    REWARD_MOON_BLOCK: float = 15.0
    # Quemar palos temprano: ganar baza con 0 pts en bazas 1-7
    REWARD_EARLY_SAFE_BURN: float = 1.5
    # Penalización por retener A♠/K♠ con Q♠ activa tras baza 7
    REWARD_LIABILITY_HOLD: float = -5.0

    # Umbral para detección de moon block
    MOON_BLOCK_CORAZONES_UMBRAL: int = 6


class CalculadoraRecompensasV21:
    """Calculadora de recompensas v2_1 — Strategy Pattern.

    Extiende la calculadora score-based con señales tácticas.
    Todos los métodos son stateless (reciben estado como argumentos).
    """

    def __init__(self, config: Optional[RewardConfigV21] = None):
        self.cfg = config or RewardConfigV21()

    # ------------------------------------------------------------------
    # Per-baza (base, sin cambios)
    # ------------------------------------------------------------------

    def recompensa_baza(
        self,
        cartas_baza: list,
        agente_idx: int,
        ganador: int,
    ) -> float:
        """Recompensa base por ganar/perder una baza.

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            ganador: Índice del ganador de la baza.

        Returns:
            Suma de penalizaciones por ♡ y Q♠ si el agente ganó, 0 si no.
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
    # NEW: Q♠ dump on rival
    # ------------------------------------------------------------------

    def recompensa_qs_dump(
        self,
        agente_idx: int,
        carta_jugada_agente,  # Carta que jugó el agente en esta baza
        palo_salida: Optional[int],
        ganador: int,
    ) -> float:
        """Recompensa por descartar Q♠ sobre un rival.

        Se activa cuando:
        - El agente jugó Q♠
        - El agente era void en el palo de salida (la descartó, no la siguió)
        - Un rival ganó la baza (la Q♠ cayó en otro)

        Args:
            agente_idx: Índice del agente.
            carta_jugada_agente: La carta que jugó el agente en esta baza.
            palo_salida: Palo de salida de la baza (None si el agente lideró).
            ganador: Índice del ganador de la baza.

        Returns:
            +12 si el agente descartó Q♠ sobre un rival, 0 en otro caso.
        """
        if not carta_jugada_agente.es_dama_de_picas:
            return 0.0
        if ganador == agente_idx:
            return 0.0  # nosotros capturamos Q♠ → no es dump
        # El agente jugó Q♠. Si palo_salida no es ♠, fue un descarte.
        # Si palo_salida es None, el agente lideró → no es dump (es liderazgo).
        # Pero incluso liderar Q♠ cuando no somos máxima es buena estrategia.
        # Solo penalizamos el caso trivial: liderar Q♠ no es dump.
        # Si palo_salida es ♠, el agente siguió el palo → no es dump.
        if palo_salida is not None and palo_salida == 2:  # ♠
            return 0.0  # siguió palo de ♠, no fue descarte
        return self.cfg.REWARD_QS_DUMP

    # ------------------------------------------------------------------
    # NEW: Moon block
    # ------------------------------------------------------------------

    def recompensa_moon_block(
        self,
        agente_idx: int,
        ganador: int,
        puntos_en_baza: int,
        corazones_por_rival: List[int],
    ) -> float:
        """Recompensa por bloquear un pozo ajeno.

        Se activa cuando:
        - El agente ganó la baza
        - La baza contiene puntos (corazones o Q♠)
        - Algún rival tiene ≥6 corazones acumulados (sospecha de pozo)

        Args:
            agente_idx: Índice del agente.
            ganador: Índice del ganador de la baza.
            puntos_en_baza: Puntos totales en esta baza.
            corazones_por_rival: Corazones acumulados por cada rival (sin contar agente).

        Returns:
            +15 si el agente bloqueó el pozo, 0 en otro caso.
        """
        if ganador != agente_idx:
            return 0.0
        if puntos_en_baza == 0:
            return 0.0  # no había nada que bloquear
        if max(corazones_por_rival) < self.cfg.MOON_BLOCK_CORAZONES_UMBRAL:
            return 0.0
        return self.cfg.REWARD_MOON_BLOCK

    # ------------------------------------------------------------------
    # NEW: Early safe burn
    # ------------------------------------------------------------------

    def recompensa_early_safe_burn(
        self,
        agente_idx: int,
        ganador: int,
        puntos_en_baza: int,
        numero_baza: int,
    ) -> float:
        """Recompensa por quemar palos temprano (ganar baza limpia).

        Se activa cuando:
        - El agente ganó la baza
        - La baza tiene 0 puntos
        - Estamos en bazas 1-7

        Enseña al modelo a liderar con cartas altas de palos seguros (♣/♦)
        para ganar bazas limpias temprano, evitando ser forzado a ganar
        bazas con corazones después.

        Args:
            agente_idx: Índice del agente.
            ganador: Índice del ganador de la baza.
            puntos_en_baza: Puntos totales en esta baza.
            numero_baza: Número de baza (1-13).

        Returns:
            +1.5 si el agente ganó baza limpia temprano, 0 en otro caso.
        """
        if ganador != agente_idx:
            return 0.0
        if puntos_en_baza > 0:
            return 0.0
        if numero_baza > 7:
            return 0.0
        return self.cfg.REWARD_EARLY_SAFE_BURN

    # ------------------------------------------------------------------
    # NEW: Liability hold penalty
    # ------------------------------------------------------------------

    def recompensa_liability_hold(
        self,
        mano_agente: list,  # List[Carta]
        q_activa: bool,
        numero_baza: int,
    ) -> float:
        """Penalización por retener A♠/K♠ con Q♠ activa tras baza 7.

        A♠ y K♠ son liability cuando Q♠ sigue en circulación: si alguien
        lidera ♠, podemos vernos forzados a ganar la baza y recibir Q♠
        descartada encima (+13 pts). BotExperto las descarga en bazas limpias.

        Args:
            mano_agente: Cartas en mano del agente.
            q_activa: True si Q♠ no ha sido capturada aún.
            numero_baza: Número de baza actual (1-13).

        Returns:
            -5 por cada A♠/K♠ retenido si Q♠ activa y baza ≥7.
        """
        if not q_activa:
            return 0.0
        if numero_baza < 7:
            return 0.0

        count = 0
        for c in mano_agente:
            if c.palo == 2 and c.valor >= 13 and not c.es_dama_de_picas:
                count += 1  # A♠ (14) o K♠ (13)

        if count == 0:
            return 0.0
        return self.cfg.REWARD_LIABILITY_HOLD * count

    # ------------------------------------------------------------------
    # Fin de mano (base, sin cambios)
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
            agente_idx: Índice del agente.
            pleno_jugador: Quién hizo moon (None si nadie).

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
        """Bonus ligero por posición relativa en la mano.

        Args:
            agente_idx: Índice del agente.
            puntuaciones_mano: Puntos de cada jugador en esta mano.

        Returns:
            +5 si fue el mejor (menos puntos), -5 si fue el peor, 0 en medio.
        """
        mi_pts = puntuaciones_mano[agente_idx]
        min_pts = min(puntuaciones_mano)
        max_pts = max(puntuaciones_mano)

        if mi_pts == min_pts:
            return self.cfg.REWARD_MEJOR_MANO
        if mi_pts == max_pts:
            return self.cfg.REWARD_PEOR_MANO
        return 0.0


__all__ = [
    "RewardConfigV21",
    "CalculadoraRecompensasV21",
]
