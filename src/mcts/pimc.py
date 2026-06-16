"""
PIMC — Perfect Information Monte Carlo para Corazones.

Algoritmo:
  Para cada carta legal del agente:
    1. Generar N mundos (determinizar cartas desconocidas de oponentes).
    2. En cada mundo, simular la mano completa con una política de rollout.
    3. Promediar la puntuación del agente.
  Retorna la carta que minimiza la puntuación esperada.

Referencia: Browne et al. (2012) — "A Survey of Monte Carlo Tree Search Methods"
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from src.dominio.carta import Carta
from src.dominio.jugador import Jugador
from src.dominio.motor import MotorCorazones
from src.agentes.heuristicos import bot_evasivo


# ──────────────────────────────────────────────────────────────
# Clonación eficiente del MotorCorazones
# ──────────────────────────────────────────────────────────────

def _clonar_motor(motor: MotorCorazones) -> MotorCorazones:
    """Deep copy ligero del motor sin recurrir a copy.deepcopy().

    Carta es un value object inmutable, así que las listas se copian
    en shallow (sin clonar los objetos Carta individuales).
    """
    clon = MotorCorazones.__new__(MotorCorazones)

    # Atributos escalares
    clon.corazones_rotos = motor.corazones_rotos
    clon.numero_baza = motor.numero_baza
    clon.palo_de_salida = motor.palo_de_salida
    clon.indice_jugador_inicial = motor.indice_jugador_inicial
    clon._mano_activa = motor._mano_activa

    # Mesa: lista de tuplas (int, Carta) — los objetos son inmutables
    clon.mesa = list(motor.mesa)

    # Baraja (no se usa después de repartir, copiamos por completitud)
    clon.baraja = motor.baraja

    # Jugadores
    clon.jugadores = []
    for j in motor.jugadores:
        nuevo = Jugador.__new__(Jugador)
        nuevo.nombre = j.nombre
        nuevo.mano = list(j.mano)
        nuevo.bazas_ganadas = list(j.bazas_ganadas)
        nuevo.puntuacion_historica = j.puntuacion_historica
        clon.jugadores.append(nuevo)

    return clon


# ──────────────────────────────────────────────────────────────
# Determinización
# ──────────────────────────────────────────────────────────────

def determinizar(
    motor: MotorCorazones,
    agente_idx: int,
    vacios: Optional[Dict[int, Set[int]]] = None,
    rng: Optional[np.random.Generator] = None,
    max_intentos: int = 40,
) -> MotorCorazones:
    """Crea una copia del motor con cartas oponentes aleatorizadas.

    Las cartas del agente y el estado público (bazas ganadas, mesa,
    corazones_rotos, número de baza) se preservan exactamente.
    Las cartas de los oponentes se redistribuyen aleatoriamente entre
    las cartas que no son visibles para el agente.

    Respeta los voids conocidos (si se proporcionan) mediante reintento:
    si una distribución viola un void, se regenera hasta max_intentos
    veces. Tras agotar los intentos, ignora los constraints restantes.

    Args:
        motor: Motor original (NO se modifica).
        agente_idx: Índice del jugador cuya mano es "conocida".
        vacios: Dict[jugador_idx → set de palos donde es void].
        rng: Generador aleatorio (para reproducibilidad).
        max_intentos: Reintentos antes de ignorar constraints de voids.

    Returns:
        Nuevo MotorCorazones con el mismo estado público pero con
        cartas oponentes aleatorizadas.
    """
    if rng is None:
        rng = np.random.default_rng()
    vacios = vacios or {}

    # Cartas ya colocadas públicamente (cementerio + mesa)
    vistas: Set[int] = set()
    for j in motor.jugadores:
        for c in j.bazas_ganadas:
            vistas.add(c.id)
    for _, c in motor.mesa:
        vistas.add(c.id)
    mano_agente_ids: Set[int] = {c.id for c in motor.jugadores[agente_idx].mano}

    # Pool de cartas desconocidas = están en manos de oponentes
    pool: List[Carta] = [
        c for c in Carta._TODAS
        if c.id not in vistas and c.id not in mano_agente_ids
    ]

    # Cuántas cartas debe recibir cada oponente
    num_cartas: Dict[int, int] = {
        i: len(motor.jugadores[i].mano)
        for i in range(4) if i != agente_idx
    }
    oponentes = list(num_cartas.keys())

    assert sum(num_cartas.values()) == len(pool), (
        f"Pool size {len(pool)} ≠ sum oponentes {sum(num_cartas.values())}"
    )

    # Separar oponentes con voids de los sin restricción
    con_void = [i for i in oponentes if i in vacios]
    sin_void = [i for i in oponentes if i not in vacios]

    for intento in range(max_intentos + 1):
        last_attempt = intento >= max_intentos
        pool_rest = list(pool)
        rng.shuffle(pool_rest)

        nuevas_manos: Dict[int, List[Carta]] = {}
        valido = True

        # Asignar primero los oponentes con voids (restricción más estricta)
        for jugador_idx in con_void:
            n = num_cartas[jugador_idx]
            palos_void = vacios[jugador_idx]

            if last_attempt:
                # Último intento: ignorar constraint, tomar las primeras n
                elegidas = pool_rest[:n]
                pool_rest = pool_rest[n:]
            else:
                elegibles = [c for c in pool_rest if c.palo not in palos_void]
                if len(elegibles) < n:
                    valido = False
                    break
                # Barajar elegibles y tomar n
                rng.shuffle(elegibles)
                elegidas = elegibles[:n]
                # Eliminar las elegidas del pool_rest
                ids_elegidas = {c.id for c in elegidas}
                pool_rest = [c for c in pool_rest if c.id not in ids_elegidas]

            nuevas_manos[jugador_idx] = elegidas

        if not valido:
            continue

        # Asignar oponentes sin restricción con el resto del pool
        for jugador_idx in sin_void:
            n = num_cartas[jugador_idx]
            nuevas_manos[jugador_idx] = pool_rest[:n]
            pool_rest = pool_rest[n:]

        break

    # Construir clon con las nuevas manos
    clon = _clonar_motor(motor)
    for jugador_idx, cartas in nuevas_manos.items():
        clon.jugadores[jugador_idx].mano = cartas

    return clon


# ──────────────────────────────────────────────────────────────
# Rollout (simulación hasta el fin de la mano)
# ──────────────────────────────────────────────────────────────

def simular_resto_mano(
    motor: MotorCorazones,
    agente_idx: int,
    primera_carta: Carta,
    bots: Dict[int, Callable],
) -> int:
    """Simula el resto de la mano desde el turno actual del agente.

    El agente juega primera_carta. Para todos los turnos siguientes
    (incluyendo turnos futuros del agente), se usa la política en bots.

    IMPORTANTE: modifica motor in-place. Para preservar el original,
    pasa un clon obtenido de determinizar() o _clonar_motor().

    Args:
        motor: Estado actual (SE MODIFICA).
        agente_idx: Índice del agente.
        primera_carta: Carta que juega el agente en este turno.
        bots: Dict[jugador_idx → policy(motor, idx, legales) → Carta].

    Returns:
        Puntuación de la mano del agente (0-26, ajustada por pleno).
    """
    motor.jugar_carta(agente_idx, primera_carta)

    while True:
        # Resolver baza si está completa
        if len(motor.mesa) == 4:
            motor.resolver_baza()

        # Verificar fin de mano
        if all(len(j.mano) == 0 for j in motor.jugadores) and len(motor.mesa) == 0:
            break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        carta = bots[idx](motor, idx, legales)
        motor.jugar_carta(idx, carta)

    return motor.calcular_puntuacion_mano()[agente_idx]


# ──────────────────────────────────────────────────────────────
# PIMC principal
# ──────────────────────────────────────────────────────────────

def _bots_por_defecto() -> Dict[int, Callable]:
    """Política de rollout: bot_evasivo para todos los jugadores."""
    return {i: bot_evasivo for i in range(4)}


def _puntaje_esperado_por_carta(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    vacios: Optional[Dict[int, Set[int]]] = None,
    num_mundos: int = 100,
    rng: Optional[np.random.Generator] = None,
    crear_bots: Optional[Callable[[], Dict[int, Callable]]] = None,
) -> Dict[int, float]:
    """Calcula la puntuación esperada por cada carta legal.

    Args:
        motor: Estado actual (NO se modifica).
        agente_idx: Índice del agente.
        legales: Cartas legales para el agente.
        vacios: Voids conocidos para la determinización.
        num_mundos: Número de mundos (determinizaciones) a simular.
        rng: Generador aleatorio.
        crear_bots: Factory que retorna dict de políticas de rollout.

    Returns:
        Dict[carta_id → puntuación promedio esperada].
    """
    if rng is None:
        rng = np.random.default_rng()
    if crear_bots is None:
        crear_bots = _bots_por_defecto

    acumulado: Dict[int, float] = {c.id: 0.0 for c in legales}

    for _ in range(num_mundos):
        # Una determinización por mundo, compartida por todas las cartas
        mundo = determinizar(motor, agente_idx, vacios=vacios, rng=rng)

        for carta in legales:
            # Clonar el mundo para cada carta (no contaminar entre evaluaciones)
            clon = _clonar_motor(mundo)
            bots = crear_bots()
            puntos = simular_resto_mano(clon, agente_idx, carta, bots)
            acumulado[carta.id] += puntos

    # Promediar
    return {carta_id: total / num_mundos for carta_id, total in acumulado.items()}


def pimc_mejor_jugada(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    vacios: Optional[Dict[int, Set[int]]] = None,
    num_mundos: int = 100,
    rng: Optional[np.random.Generator] = None,
    crear_bots: Optional[Callable[[], Dict[int, Callable]]] = None,
) -> Carta:
    """Retorna la carta legal que minimiza la puntuación esperada del agente.

    Usa Perfect Information Monte Carlo (PIMC / determinización):
    para cada mundo posible (distribución aleatoria de cartas oponentes),
    evalúa cada carta legal simulando la mano hasta el final con una
    política de rollout, y elige la carta con el menor puntaje promedio.

    Args:
        motor: Estado actual (NO se modifica).
        agente_idx: Índice del agente.
        legales: Cartas legales para el agente. Debe ser no vacía.
        vacios: Voids conocidos públicamente (palo → jugadores void).
        num_mundos: Número de mundos a simular (100 = buena precisión).
        rng: Generador aleatorio para reproducibilidad.
        crear_bots: Factory → dict de políticas de rollout.
                    Default: bot_evasivo para todos.

    Returns:
        La carta legal con el menor puntaje esperado.

    Raises:
        ValueError: Si legales está vacía.
    """
    if not legales:
        raise ValueError("La lista de cartas legales no puede estar vacía")

    if len(legales) == 1:
        return legales[0]

    scores = _puntaje_esperado_por_carta(
        motor, agente_idx, legales,
        vacios=vacios, num_mundos=num_mundos,
        rng=rng, crear_bots=crear_bots,
    )
    return min(legales, key=lambda c: scores[c.id])
