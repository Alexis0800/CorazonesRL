"""
Recompensa v10 — partida completa con shaping basado en potencial (PBRS).

Reemplaza el esquema de recompensa por mano aislada de v5–v9. Dos componentes:

  1. R_terminal (objetivo primario, una sola vez al fin de partida): recompensa
     según el puesto final del agente. Es la ÚNICA señal que define el óptimo.

  2. Shaping basado en potencial (Ng, Harada & Russell, 1999):
        F(s → s') = γ·Φ(s') − Φ(s)
     Con Φ función del marcador acumulado y Φ(terminal)=0. La suma de F sobre el
     episodio telescopia a una constante independiente de la política, por lo que
     NO cambia el óptimo y NO puede ser "farmeada": solo densifica la señal y
     acelera la asignación de crédito.

Toda la estrategia avanzada (defender/hacer el pozo, alimentar al líder) emerge
de Φ sobre el marcador, sin codificar tácticas a mano.

Ver: docs/Rediseño_v10_partida_completa.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class RewardConfigPartida:
    """Configuración inmutable de la recompensa de partida completa (v10 SSOT)."""

    # --- R_terminal: recompensa por puesto final (objetivo "ganar > top-2") ---
    R_PRIMERO: float = 1.0
    R_SEGUNDO: float = 0.3
    R_TERCERO: float = -0.3
    R_CUARTO: float = -1.0

    # --- Shaping PBRS ---
    PHI_LAMBDA: float = 0.5      # peso del potencial (magnitud comparable a R_terminal)
    PHI_ESCALA: float = 100.0    # normalización de la diferencia de marcador

    # --- Partida ---
    LIMITE_PARTIDA: int = 100


class CalculadoraRecompensasPartida:
    """Calcula R_terminal y el potencial Φ para el shaping PBRS."""

    def __init__(self, config: RewardConfigPartida | None = None):
        self.cfg = config or RewardConfigPartida()

    # ------------------------------------------------------------------
    # Potencial Φ — base del shaping PBRS
    # ------------------------------------------------------------------

    def potencial(self, scores: Sequence[int], agente_idx: int) -> float:
        """Φ(s): qué tan buena es la posición del agente en el marcador.

        Φ = λ · clip( (media_scores_rivales − mi_score) / escala , −1, +1 )

        Mayor cuando el agente va por debajo (menos puntos) que el promedio rival.
        Por construcción Φ([0,0,0,0], ·) = 0, así que el potencial inicial de la
        partida es 0 y el shaping neto del episodio ≈ 0 (no farmeable).
        """
        mi = float(scores[agente_idx])
        otros = [float(scores[i]) for i in range(len(scores)) if i != agente_idx]
        media_otros = sum(otros) / len(otros)
        ventaja = (media_otros - mi) / self.cfg.PHI_ESCALA
        ventaja = max(-1.0, min(1.0, ventaja))
        return self.cfg.PHI_LAMBDA * ventaja

    def shaping(
        self,
        scores_antes: Sequence[int],
        scores_despues: Sequence[int],
        agente_idx: int,
        gamma: float,
        terminal: bool,
    ) -> float:
        """F = γ·Φ(s') − Φ(s). En estado terminal Φ(s')=0 por definición."""
        phi_antes = self.potencial(scores_antes, agente_idx)
        phi_despues = 0.0 if terminal else self.potencial(scores_despues, agente_idx)
        return gamma * phi_despues - phi_antes

    # ------------------------------------------------------------------
    # R_terminal — objetivo primario
    # ------------------------------------------------------------------

    def recompensa_terminal(self, scores_finales: Sequence[int], agente_idx: int) -> float:
        """Recompensa por puesto final del agente al terminar la partida.

        El puesto se calcula por puntuación ascendente (menos puntos = mejor).
        Los empates comparten el promedio de las recompensas de los puestos
        empatados, para no penalizar/premiar arbitrariamente por desempate.
        """
        valores = [self.cfg.R_PRIMERO, self.cfg.R_SEGUNDO,
                   self.cfg.R_TERCERO, self.cfg.R_CUARTO]
        mi_score = scores_finales[agente_idx]

        mejores = sum(1 for i, s in enumerate(scores_finales)
                      if i != agente_idx and s < mi_score)
        empatados = sum(1 for i, s in enumerate(scores_finales)
                        if i != agente_idx and s == mi_score)

        # Puestos que ocuparía el grupo empatado (incluido el agente)
        puestos = list(range(mejores, mejores + empatados + 1))
        return sum(valores[p] for p in puestos) / len(puestos)

    def puesto(self, scores_finales: Sequence[int], agente_idx: int) -> int:
        """Puesto del agente (1 = ganador). Empates → mejor puesto compartido."""
        mi_score = scores_finales[agente_idx]
        mejores = sum(1 for i, s in enumerate(scores_finales)
                      if i != agente_idx and s < mi_score)
        return mejores + 1


_recompensas_partida = CalculadoraRecompensasPartida()

__all__ = ["RewardConfigPartida", "CalculadoraRecompensasPartida"]
