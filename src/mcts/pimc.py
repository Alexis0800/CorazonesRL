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
from src.agentes.heuristicos import bot_evasivo, bot_conservador, bot_agresivo


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
    mano_agente_ids: Set[int] = {
        c.id for c in motor.jugadores[agente_idx].mano}

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
        if not legales:
            # Estado degenerado (jugador sin cartas): la mano terminó, cortar.
            break
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


# ──────────────────────────────────────────────────────────────
# Rollout factories mejoradas
# ──────────────────────────────────────────────────────────────

def _crear_bots_experto() -> Dict[int, Callable]:
    """Rollout con BotExperto para todos los oponentes (más realista)."""
    from src.agentes.bot_experto import BotExperto
    bots = [BotExperto() for _ in range(4)]
    return {i: bots[i] for i in range(4)}


def _crear_bots_mixto(rng: np.random.Generator) -> Dict[int, Callable]:
    """Rollout con mezcla de ARQUETIPOS HUMANOS diversos (para generalizar).

    Cada uno de los 4 asientos recibe, al azar, uno de los arquetipos fuertes y
    distintos que juegan los humanos:
      - experto       (balanceado, conteo, gestión Q♠)
      - castigador    (presiona picas para forzar la Q♠)
      - lunático      (intenta el pozo con mano fuerte)
      - atacante_lider(carga puntos al líder de la partida)
      - evasivo       (defensivo, minimiza puntos propios)

    Instancias frescas en cada mundo (los bots con estado se reinician solos).
    Se excluyen los muy débiles (conservador/agresivo puros) para que el modelo
    de oponente de PIMC sea realista vs juego competente/humano.
    """
    from src.agentes.bot_experto import BotExperto
    from src.agentes.bot_castigador import BotCastigador
    from src.agentes.bot_lunatico import BotLunatico
    from src.agentes.bot_atacante_lider import BotAtacanteLider

    arquetipos: List[Callable[[], Callable]] = [
        lambda: BotExperto(),
        lambda: BotCastigador(),
        lambda: BotLunatico(),
        lambda: BotAtacanteLider(),
        lambda: bot_evasivo,
    ]
    elegidos = rng.integers(0, len(arquetipos), size=4)
    return {i: arquetipos[int(elegidos[i])]() for i in range(4)}


def crear_bots_rollout(
    tipo: str = "evasivo",
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, Callable]:
    """Factory de políticas de rollout parametrizable.

    Args:
        tipo: "evasivo" (default), "experto", "mixto", "conservador", "agresivo"
        rng: Generador aleatorio (para mixto).

    Returns:
        Dict[jugador_idx → callable(motor, idx, legales) → Carta]
    """
    if tipo == "experto":
        return _crear_bots_experto()
    if tipo == "mixto":
        return _crear_bots_mixto(rng or np.random.default_rng())
    if tipo == "conservador":
        return {i: bot_conservador for i in range(4)}
    if tipo == "agresivo":
        return {i: bot_agresivo for i in range(4)}
    return {i: bot_evasivo for i in range(4)}


# ──────────────────────────────────────────────────────────────
# MCTS — Monte Carlo Tree Search (con árbol)
# ──────────────────────────────────────────────────────────────

class _NodoMCTS:
    """Nodo del árbol MCTS para Corazones (minimización de puntos)."""

    __slots__ = (
        "visitas", "valor_total", "carta", "hijos",
        "idx_jugador", "legales",
    )

    def __init__(self, carta: Optional[Carta] = None) -> None:
        self.visitas: int = 0
        self.valor_total: float = 0.0
        self.carta: Optional[Carta] = carta  # carta que llevó a este nodo
        self.hijos: Dict[int, '_NodoMCTS'] = {}  # carta.id → _NodoMCTS
        self.idx_jugador: int = 0
        self.legales: List[Carta] = []

    @property
    def valor_medio(self) -> float:
        """Puntuación media (0=perfecto, 26=peor)."""
        if self.visitas == 0:
            return 0.0  # optimista: asumimos 0 pts para nodos no explorados
        return self.valor_total / self.visitas

    def ucb(self, c: float = np.sqrt(2.0)) -> float:
        """UCB para minimización: menor = mejor.

        Nodos no visitados reciben -inf para ser explorados primero.
        """
        if self.visitas == 0:
            return -float("inf")
        # valor_medio bajo es bueno; restamos exploración → más negativo = más atractivo
        return self.valor_medio - c * np.sqrt(np.log(self.visitas) / self.visitas)


def _seleccionar_mejor_hijo(
    nodo: _NodoMCTS,
    c: float = np.sqrt(2.0),
) -> _NodoMCTS:
    """Selecciona el hijo con menor UCB (minimización de puntos)."""
    return min(nodo.hijos.values(), key=lambda h: h.ucb(c))


def mcts_mejor_jugada(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    vacios: Optional[Dict[int, Set[int]]] = None,
    num_simulaciones: int = 500,
    rng: Optional[np.random.Generator] = None,
    crear_bots: Optional[Callable[[], Dict[int, Callable]]] = None,
    rollout_tipo: str = "experto",
    profundidad_agente: int = 1,
) -> Carta:
    """UCB plano en la raíz sobre determinizaciones (no un árbol multinivel).

    Cada simulación muestrea un mundo determinizado nuevo, elige una carta
    legal de la raíz con UCB (bandit plano sobre las cartas legales) y hace
    rollout hasta el fin de la mano; el coste se retropropaga solo al hijo
    elegido. No se construye un árbol de decisiones persistente ni se
    comparten estadísticas entre niveles. Con profundidad_agente>1, cada
    turno futuro del agente se resuelve con una búsqueda MCTS independiente
    (mini-MCTS), no como ramas de un árbol compartido.

    Args:
        motor: Estado actual (NO se modifica).
        agente_idx: Índice del agente.
        legales: Cartas legales.
        vacios: Voids conocidos.
        num_simulaciones: Total de simulaciones MCTS (default 500).
        rng: Generador aleatorio.
        crear_bots: Factory de políticas (si None, usa rollout_tipo).
        rollout_tipo: "evasivo", "experto", "mixto" (default "experto").
        profundidad_agente: Cuántos niveles de nodos del agente construir.
            1 = solo optimiza la decisión actual (comportamiento original).
            2 = también optimiza la siguiente decisión del agente.
            3 = optimiza 3 decisiones hacia adelante.

    Returns:
        La carta legal con menor puntuación esperada según MCTS.
    """
    if not legales:
        raise ValueError("La lista de cartas legales no puede estar vacía")
    if len(legales) == 1:
        return legales[0]

    if rng is None:
        rng = np.random.default_rng()
    if crear_bots is None:
        # Factory que crea bots FRESCOS en cada simulación.
        # BotExperto tiene estado interno (vacios, sospecha_pozo) que
        # se acumula entre simulaciones — NO se debe reutilizar.
        def crear_bots():
            return crear_bots_rollout(tipo=rollout_tipo, rng=rng)

    # Raíz del árbol
    raiz = _NodoMCTS()
    raiz.idx_jugador = agente_idx
    raiz.legales = list(legales)

    # Inicializar hijos para todas las cartas legales
    for carta in legales:
        hijo = _NodoMCTS(carta=carta)
        raiz.hijos[carta.id] = hijo

    for _ in range(num_simulaciones):
        # 1. Determinizar: crear un mundo
        mundo = determinizar(motor, agente_idx, vacios=vacios, rng=rng)

        # 2. Seleccionar carta desde la raíz (exploración con UCB)
        carta_elegida = _seleccionar_mejor_hijo(raiz).carta

        # 3. Simular con profundidad de agente variable
        clon = _clonar_motor(mundo)
        bots_frescos = crear_bots()
        puntos = _simular_mcts_con_profundidad(
            clon, agente_idx, carta_elegida, bots_frescos,
            profundidad_agente - 1, vacios, rng, crear_bots, rollout_tipo)

        # 4. Backpropagar: actualizar el valor como COSTE (menos = mejor)
        hijo_nodo = raiz.hijos[carta_elegida.id]
        hijo_nodo.visitas += 1
        hijo_nodo.valor_total += puntos  # acumulamos coste (menor = mejor)

    # Seleccionar la carta con menor valor medio (menor puntuación esperada)
    return min(legales, key=lambda c: raiz.hijos[c.id].valor_medio)


def _simular_mcts_con_profundidad(
    motor: MotorCorazones,
    agente_idx: int,
    primera_carta: Carta,
    bots: Dict[int, Callable],
    profundidad_restante: int,
    vacios: Optional[Dict[int, Set[int]]],
    rng: np.random.Generator,
    crear_bots: Callable[[], Dict[int, Callable]],
    rollout_tipo: str,
) -> float:
    """Simula la mano desde el turno actual, aplicando MCTS en turnos
    futuros del agente si profundidad_restante > 0.

    Args:
        motor: Estado actual (SE MODIFICA).
        agente_idx: Índice del agente.
        primera_carta: Carta que juega el agente en este turno.
        bots: Políticas de rollout para oponentes.
        profundidad_restante: Niveles adicionales de optimización MCTS.
        vacios: Voids conocidos.
        rng: Generador aleatorio.
        crear_bots: Factory para bots frescos.
        rollout_tipo: Tipo de rollout.

    Returns:
        Puntuación final del agente (0-26).
    """
    motor.jugar_carta(agente_idx, primera_carta)

    while True:
        if len(motor.mesa) == 4:
            motor.resolver_baza()

        if all(len(j.mano) == 0 for j in motor.jugadores) and len(motor.mesa) == 0:
            break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)

        if not legales:
            break

        if idx == agente_idx and profundidad_restante > 0:
            # ── Turno del agente: mini-MCTS con pocas simulaciones ──
            sims_mcts = max(10, 50 // (3 - min(profundidad_restante, 2)))
            mejor = mcts_mejor_jugada(
                motor, agente_idx, legales,
                num_simulaciones=sims_mcts,
                rng=rng, crear_bots=crear_bots,
                rollout_tipo=rollout_tipo,
                profundidad_agente=profundidad_restante,
            )
            motor.jugar_carta(agente_idx, mejor)
            profundidad_restante -= 1
        else:
            carta = bots[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)

    return motor.calcular_puntuacion_mano()[agente_idx]
