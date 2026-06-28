"""
Heurística de pase para Corazones (v10b), codificada de los MD de estrategia.

`pase_heuristico(motor, idx)` devuelve las 3 cartas que un jugador competente
pasaría, según Estrategias Avanzadas.md:
  - Soltar picas altas (A♠/K♠/Q♠) si NO hay suficientes picas bajas que las
    protejan (riesgo de comerse la Q♠).
  - Soltar corazones altos (A♥/K♥/Q♥) para no ganar bazas con puntos.
  - Vaciar un palo corto (♣/♦) para ganar libertad de descarte pronto.
  - En general, soltar cartas altas peligrosas.

Sirve como (a) política de pase de los rivales en el env y (b) etiqueta-maestra
para el dataset BC de la fase de pase.
"""
from __future__ import annotations

from typing import List

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones

_PICA, _CORAZON = 2, 3


def _peligro(carta: Carta, conteo_palo: dict) -> float:
    """Puntúa cuán deseable es DESHACERSE de esta carta (más alto = pasar antes)."""
    palo = carta.palo
    valor = carta.valor
    score = 0.0

    # Picas altas (Q=12, K=13, A=14): muy peligrosas si hay pocas picas que protejan.
    if palo == _PICA and valor >= 12:
        picas = conteo_palo[_PICA]
        score += 12.0 if picas <= 3 else 4.0
        if carta.es_dama_de_picas:
            score += 2.0  # la Q♠ es el mayor riesgo

    # Corazones altos: ganan bazas con puntos.
    elif carta.es_corazon:
        score += float(valor)  # A♥=14 ... cuanto más alto, peor

    # Otras cartas altas (A/K de ♣/♦): moderadamente útiles de soltar.
    elif valor >= 13:
        score += valor * 0.4

    # Incentivo de vaciado: soltar cartas de un palo corto (≤2) ayuda a quedar void.
    if palo in (0, 1) and conteo_palo[palo] <= 2:
        score += 3.0 - conteo_palo[palo]  # palo de 1 → +2, de 2 → +1

    # Desempate suave por valor (preferir soltar la más alta).
    score += valor * 0.01
    return score


def pase_heuristico(motor: MotorCorazones, idx: int) -> List[Carta]:
    """Devuelve las 3 cartas que el jugador `idx` debería pasar."""
    mano = list(motor.jugadores[idx].mano)
    if len(mano) < 3:
        return mano[:3]
    conteo_palo = {p: sum(1 for c in mano if c.palo == p) for p in range(4)}
    ordenadas = sorted(mano, key=lambda c: _peligro(c, conteo_palo), reverse=True)
    return ordenadas[:3]


__all__ = ["pase_heuristico"]
