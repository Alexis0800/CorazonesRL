"""
PIMC Recursivo — Perfect Information Monte Carlo con lookahead multi-nivel.

Extiende el PIMC estándar optimizando no solo la decisión actual del agente,
sino también sus decisiones futuras en la misma mano. Esto permite que el
dataset BC enseñe estrategia multibaza real.

Algoritmo (profundidad D):
  Para cada carta legal c en el turno actual:
    1. Generar N mundos (determinizar cartas desconocidas)
    2. En cada mundo:
       a. Jugar c
       b. Simular oponentes con política de rollout hasta el próximo turno del agente
       c. Si D > 1 y quedan bazas: llamar recursivamente con D-1 para
          encontrar la mejor carta en el próximo turno del agente
       d. Si D == 1 o fin de mano: terminar con rollout heurístico
    3. Promediar la puntuación final del agente

Complejidad:
  - Profundidad 1: O(L × M) donde L=legales, M=mundos (igual que PIMC estándar)
  - Profundidad 2: O(L × M × L' × M') donde L',M' son legales y mundos en el
    próximo turno (aproximadamente L × M × L × M_d2)
  - Para dataset offline esto es aceptable (minutos por mano, no segundos)

Referencia:
  Furtak & Buro (2009) — "Recursive Monte Carlo Search for Imperfect
  Information Games" — la base teórica del PIMC recursivo.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.mcts.pimc import (
    _clonar_motor,
    determinizar,
    simular_resto_mano,
    _bots_por_defecto,
    crear_bots_rollout,
)


# ──────────────────────────────────────────────────────────────
# PIMC Recursivo
# ──────────────────────────────────────────────────────────────

def _puntaje_esperado_recursivo(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    profundidad: int = 2,
    num_mundos: int = 30,
    vacios: Optional[Dict[int, Set[int]]] = None,
    rng: Optional[np.random.Generator] = None,
    crear_bots: Optional[Callable[[], Dict[int, Callable]]] = None,
    rollout_tipo: str = "evasivo",
) -> Dict[int, float]:
    """Calcula la puntuación esperada por carta usando PIMC recursivo.

    A diferencia de _puntaje_esperado_por_carta (que usa rollout heurístico
    para todas las decisiones futuras del agente), esta función aplica PIMC
    recursivamente en los turnos futuros del agente, hasta la profundidad
    especificada.

    Args:
        motor: Estado actual (NO se modifica).
        agente_idx: Índice del agente.
        legales: Cartas legales para el agente.
        profundidad: Cuántos turnos del agente optimizar.
            0 = solo rollout heurístico (baseline)
            1 = PIMC estándar (optimiza este turno, rollout después)
            2 = PIMC en este turno + PIMC en el próximo turno del agente
            N = recursivo hasta N turnos del agente
        num_mundos: Mundos por nivel de PIMC.
        vacios: Voids conocidos para la determinización.
        rng: Generador aleatorio.
        crear_bots: Factory de políticas de rollout (para oponentes y
            para el agente en niveles de profundidad agotados).
        rollout_tipo: Tipo de rollout cuando se agota la profundidad.

    Returns:
        Dict[carta_id → puntuación promedio esperada].
    """
    if rng is None:
        rng = np.random.default_rng()
    if crear_bots is None:
        crear_bots = _bots_por_defecto

    # Caso base: una sola carta legal o profundidad agotada
    if len(legales) <= 1 or profundidad <= 0:
        return _rollout_plano(
            motor, agente_idx, legales, num_mundos,
            vacios, rng, crear_bots)

    # Caso base: profundidad 1 = PIMC estándar (sin recursión)
    if profundidad == 1:
        return _pimc_estandar_interno(
            motor, agente_idx, legales, num_mundos,
            vacios, rng, crear_bots)

    # ── PIMC recursivo (profundidad >= 2) ──
    acumulado: Dict[int, float] = {c.id: 0.0 for c in legales}

    for _ in range(num_mundos):
        # Una determinización compartida por todas las cartas en este nivel
        mundo = determinizar(motor, agente_idx, vacios=vacios, rng=rng)

        for carta in legales:
            clon = _clonar_motor(mundo)
            puntos = _simular_con_recursion(
                clon, agente_idx, carta,
                profundidad - 1,  # un nivel menos para el próximo turno
                num_mundos,
                vacios, rng, crear_bots, rollout_tipo)
            acumulado[carta.id] += puntos

    return {cid: total / num_mundos for cid, total in acumulado.items()}


def _rollout_plano(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    num_mundos: int,
    vacios: Optional[Dict[int, Set[int]]],
    rng: np.random.Generator,
    crear_bots: Callable[[], Dict[int, Callable]],
) -> Dict[int, float]:
    """Rollout heurístico puro (profundidad 0): sin optimización PIMC.

    Cada carta se evalúa simulando el resto de la mano con la política
    de rollout para TODOS los jugadores (incluyendo el agente).
    """
    acumulado: Dict[int, float] = {c.id: 0.0 for c in legales}
    for _ in range(num_mundos):
        mundo = determinizar(motor, agente_idx, vacios=vacios, rng=rng)
        for carta in legales:
            clon = _clonar_motor(mundo)
            bots = crear_bots()
            puntos = simular_resto_mano(clon, agente_idx, carta, bots)
            acumulado[carta.id] += puntos
    return {cid: total / num_mundos for cid, total in acumulado.items()}


def _pimc_estandar_interno(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    num_mundos: int,
    vacios: Optional[Dict[int, Set[int]]],
    rng: np.random.Generator,
    crear_bots: Callable[[], Dict[int, Callable]],
) -> Dict[int, float]:
    """PIMC estándar (profundidad 1): optimiza esta decisión, rollout después.

    Idéntico a _puntaje_esperado_por_carta pero recibe crear_bots
    en vez de usar _bots_por_defecto por defecto.
    """
    acumulado: Dict[int, float] = {c.id: 0.0 for c in legales}
    for _ in range(num_mundos):
        mundo = determinizar(motor, agente_idx, vacios=vacios, rng=rng)
        for carta in legales:
            clon = _clonar_motor(mundo)
            bots = crear_bots()
            puntos = simular_resto_mano(clon, agente_idx, carta, bots)
            acumulado[carta.id] += puntos
    return {cid: total / num_mundos for cid, total in acumulado.items()}


def _simular_con_recursion(
    motor: MotorCorazones,
    agente_idx: int,
    primera_carta: Carta,
    profundidad_restante: int,
    num_mundos: int,
    vacios: Optional[Dict[int, Set[int]]],
    rng: np.random.Generator,
    crear_bots: Callable[[], Dict[int, Callable]],
    rollout_tipo: str,
) -> float:
    """Simula la mano desde el turno actual con recursión PIMC en el
    próximo turno del agente.

    Flujo:
      1. El agente juega primera_carta.
      2. Delega a _simular_desde_estado_actual para el resto.
    """
    motor.jugar_carta(agente_idx, primera_carta)
    return _simular_desde_estado_actual(
        motor, agente_idx, profundidad_restante, num_mundos,
        vacios, rng, crear_bots, rollout_tipo)


def _simular_desde_estado_actual(
    motor: MotorCorazones,
    agente_idx: int,
    profundidad_restante: int,
    num_mundos: int,
    vacios: Optional[Dict[int, Set[int]]],
    rng: np.random.Generator,
    crear_bots: Callable[[], Dict[int, Callable]],
    rollout_tipo: str,
) -> float:
    """Simula la mano desde el estado actual del motor.

    A diferencia de _simular_con_recursion, NO juega una primera carta.
    Asume que el motor ya está en un estado donde corresponde jugar
    (posiblemente después de que el agente ya jugó su carta).

    En cada turno futuro del agente, aplica PIMC recursivo si
    profundidad_restante > 0.

    Args:
        motor: Estado actual (SE MODIFICA). Puede tener la mesa con
            1-3 cartas ya jugadas en la baza actual.
        agente_idx: Índice del agente.
        profundidad_restante: Niveles de recursión restantes.
        num_mundos: Mundos para PIMC en niveles recursivos.
        vacios: Voids conocidos.
        rng: Generador aleatorio.
        crear_bots: Factory de políticas.
        rollout_tipo: Tipo de rollout cuando se agota la profundidad.

    Returns:
        Puntuación final del agente (0-26).
    """
    bots = crear_bots()

    while True:
        # Resolver baza si está completa
        if len(motor.mesa) == 4:
            motor.resolver_baza()

        # Verificar fin de mano
        if all(len(j.mano) == 0 for j in motor.jugadores) and len(motor.mesa) == 0:
            break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)

        if not legales:
            break

        if idx == agente_idx and profundidad_restante > 0:
            # ── Turno del agente: aplicar PIMC recursivo ──
            # Reducir mundos agresivamente por nivel: la calidad del
            # nivel superior compensa la imprecisión de niveles profundos.
            mundos_internos = max(
                3, num_mundos // (4 * (3 - min(profundidad_restante, 2))))
            scores_rec = _puntaje_esperado_recursivo(
                motor, agente_idx, legales,
                profundidad=profundidad_restante,
                num_mundos=mundos_internos,
                vacios=vacios, rng=rng, crear_bots=crear_bots,
                rollout_tipo=rollout_tipo,
            )
            mejor_carta = min(
                legales, key=lambda c: scores_rec.get(c.id, 999.0))
            motor.jugar_carta(agente_idx, mejor_carta)
            profundidad_restante -= 1
        else:
            # ── Turno de oponente (o agente sin profundidad): heurística ──
            carta = bots[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)

    return motor.calcular_puntuacion_mano()[agente_idx]


# ──────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────

def pimc_mejor_jugada_recursivo(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    profundidad: int = 2,
    num_mundos: int = 30,
    vacios: Optional[Dict[int, Set[int]]] = None,
    rng: Optional[np.random.Generator] = None,
    crear_bots: Optional[Callable[[], Dict[int, Callable]]] = None,
    rollout_tipo: str = "evasivo",
) -> Carta:
    """Retorna la mejor carta legal usando PIMC recursivo multi-nivel.

    Es la alternativa a pimc_mejor_jugada() pero con lookahead
    de múltiples turnos del agente.

    Args:
        motor: Estado actual (NO se modifica).
        agente_idx: Índice del agente.
        legales: Cartas legales (no vacía).
        profundidad: Turnos del agente a optimizar (default 2).
        num_mundos: Mundos PIMC por nivel.
        vacios: Voids conocidos.
        rng: Generador aleatorio.
        crear_bots: Factory de políticas de rollout.
        rollout_tipo: "evasivo", "experto", "mixto".

    Returns:
        La carta legal con menor puntuación esperada.

    Raises:
        ValueError: Si legales está vacía.
    """
    if not legales:
        raise ValueError("La lista de cartas legales no puede estar vacía")
    if len(legales) == 1:
        return legales[0]

    if crear_bots is None:
        def crear_bots():
            return crear_bots_rollout(tipo=rollout_tipo, rng=rng)

    scores = _puntaje_esperado_recursivo(
        motor, agente_idx, legales,
        profundidad=profundidad,
        num_mundos=num_mundos,
        vacios=vacios,
        rng=rng,
        crear_bots=crear_bots,
        rollout_tipo=rollout_tipo,
    )
    return min(legales, key=lambda c: scores[c.id])


__all__ = [
    "_puntaje_esperado_recursivo",
    "pimc_mejor_jugada_recursivo",
    "_simular_desde_estado_actual",
]
