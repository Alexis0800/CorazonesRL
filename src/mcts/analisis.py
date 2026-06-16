"""
Análisis exhaustivo de manos usando PIMC con enumeración completa.

Cuando el número de mundos posibles es manejable (baza ≥ 8), genera
TODOS los mundos posibles en lugar de samplear aleatoriamente,
produciendo la decisión óptima exacta (ground truth).

Herramientas de análisis:
  - pimc_exacto(): PIMC con enumeración cuando viable, sampling cuando no
  - analizar_decision(): desglose completo de puntajes por carta
  - perfil_mano(): análisis ultra-detallado de una mano
  - evaluar_mano(): evalúa combinaciones de bots
"""

from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.mcts.pimc import (
    _clonar_motor,
    determinizar,
    simular_resto_mano,
    _puntaje_esperado_por_carta,
    pimc_mejor_jugada,
    crear_bots_rollout,
)

# ──────────────────────────────────────────────────────────────
# Umbrales de viabilidad para enumeración completa
# ──────────────────────────────────────────────────────────────

_MAX_MUNDOS_EXACTO = 100_000    # máximo de mundos a enumerar (~1s en baza 10)
_BAZA_MINIMA_EXACTA = 10        # baza mínima para intentar enumeración (35K mundos)


def _num_mundos_posibles(motor: MotorCorazones, agente_idx: int) -> int:
    """Calcula el número de distribuciones posibles de cartas desconocidas."""
    n_cartas = [len(motor.jugadores[i].mano) for i in range(4) if i != agente_idx]
    total = sum(n_cartas)
    result = 1
    remaining = total
    for n in n_cartas:
        result *= math.comb(remaining, n)
        remaining -= n
    return result


# ──────────────────────────────────────────────────────────────
# Enumeración completa de mundos
# ──────────────────────────────────────────────────────────────

def _particiones(cartas: List[Carta], n1: int, n2: int, n3: int):
    """Generador de todas las distribuciones de cartas en 3 grupos.

    Usa itertools.combinations para generar todas las formas de asignar
    n1 cartas al oponente 1, n2 al 2, y el resto al 3.

    Yields:
        Tuplas (mano1, mano2, mano3) donde cada una es una tupla de Carta.
    """
    total = n1 + n2 + n3
    assert len(cartas) == total, f"{len(cartas)} != {total}"

    indices = list(range(total))

    for idx1 in itertools.combinations(indices, n1):
        set1 = set(idx1)
        resto = [i for i in indices if i not in set1]

        for idx2 in itertools.combinations(resto, n2):
            set2 = set(idx2)
            idx3 = tuple(i for i in resto if i not in set2)

            yield (
                tuple(cartas[i] for i in idx1),
                tuple(cartas[i] for i in idx2),
                tuple(cartas[i] for i in idx3),
            )


def _aplicar_mundo(
    motor: MotorCorazones,
    agente_idx: int,
    oponentes: List[int],
    manos: Tuple[Tuple[Carta, ...], Tuple[Carta, ...], Tuple[Carta, ...]],
) -> MotorCorazones:
    """Crea un clon del motor con las manos de oponentes fijadas."""
    clon = _clonar_motor(motor)
    for i, idx in enumerate(oponentes):
        clon.jugadores[idx].mano = list(manos[i])
    return clon


def enumerar_mundos(
    motor: MotorCorazones,
    agente_idx: int,
    vacios: Optional[Dict[int, Set[int]]] = None,
    max_mundos: int = _MAX_MUNDOS_EXACTO,
) -> List[MotorCorazones]:
    """Genera TODOS los mundos posibles (distribuciones de cartas).

    Solo viable cuando el número total de mundos ≤ max_mundos.
    Para bazas tempranas (>2M mundos), retorna lista vacía.

    Args:
        motor: Estado actual.
        agente_idx: Índice del agente.
        vacios: Voids conocidos (se filtran mundos inválidos).
        max_mundos: Máximo de mundos a generar.

    Returns:
        Lista de motores clonados, cada uno con una distribución distinta.
        Vacía si hay demasiados mundos (>max_mundos).
    """
    vacios = vacios or {}
    n_mundos = _num_mundos_posibles(motor, agente_idx)

    if n_mundos > max_mundos:
        return []  # demasiados mundos

    # Cartas desconocidas
    vistas: Set[int] = set()
    for j in motor.jugadores:
        for c in j.bazas_ganadas:
            vistas.add(c.id)
    for _, c in motor.mesa:
        vistas.add(c.id)
    mano_agente_ids = {c.id for c in motor.jugadores[agente_idx].mano}

    pool = [c for c in Carta._TODAS
            if c.id not in vistas and c.id not in mano_agente_ids]

    oponentes = [i for i in range(4) if i != agente_idx]
    n_cartas = [len(motor.jugadores[i].mano) for i in oponentes]

    mundos: List[MotorCorazones] = []

    for mano1, mano2, mano3 in _particiones(pool, n_cartas[0], n_cartas[1], n_cartas[2]):
        # Filtrar por voids
        valido = True
        manos_list = [mano1, mano2, mano3]
        for i, op_idx in enumerate(oponentes):
            if op_idx in vacios:
                palos_void = vacios[op_idx]
                if any(c.palo in palos_void for c in manos_list[i]):
                    valido = False
                    break

        if valido:
            clon = _aplicar_mundo(motor, agente_idx, oponentes, (mano1, mano2, mano3))
            mundos.append(clon)

    return mundos


# ──────────────────────────────────────────────────────────────
# PIMC exacto (ground truth cuando viable)
# ──────────────────────────────────────────────────────────────

def pimc_exacto(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    vacios: Optional[Dict[int, Set[int]]] = None,
    crear_bots: Optional[Callable[[], Dict[int, Callable]]] = None,
    rollout_tipo: str = "experto",
    rng: Optional[np.random.Generator] = None,
    fallback_mundos: int = 30,
) -> Tuple[Carta, Dict[int, float], bool]:
    """PIMC con enumeración completa cuando viable, sampling cuando no.

    Args:
        motor: Estado actual.
        agente_idx: Índice del agente.
        legales: Cartas legales.
        vacios: Voids conocidos.
        crear_bots: Factory de políticas de rollout.
        rollout_tipo: Tipo de rollout si crear_bots es None.
        rng: Generador aleatorio.
        fallback_mundos: Mundos a samplear si la enumeración no es viable.

    Returns:
        Tuple de:
        - Carta óptima (menor puntuación esperada)
        - Dict[carta_id → puntuación esperada]
        - bool: True si fue exacto (enumeración), False si fue sampling
    """
    if not legales:
        raise ValueError("La lista de cartas legales no puede estar vacía")
    if len(legales) == 1:
        return legales[0], {legales[0].id: 0.0}, True

    if rng is None:
        rng = np.random.default_rng()
    if crear_bots is None:
        crear_bots = lambda: crear_bots_rollout(tipo=rollout_tipo, rng=rng)  # noqa: E731

    # Intentar enumeración completa (solo si estamos en baza avanzada)
    n_mundos_posibles = _num_mundos_posibles(motor, agente_idx)
    exacto = (motor.numero_baza >= _BAZA_MINIMA_EXACTA
              and n_mundos_posibles <= _MAX_MUNDOS_EXACTO)

    if exacto:
        mundos = enumerar_mundos(motor, agente_idx, vacios=vacios)
        exacto = len(mundos) > 0  # puede fallar si hay demasiados

    if exacto:
        # PIMC exacto: evaluar todas las cartas en todos los mundos
        acumulado: Dict[int, float] = {c.id: 0.0 for c in legales}
        bots = crear_bots()

        for mundo in mundos:
            for carta in legales:
                clon = _clonar_motor(mundo)
                puntos = simular_resto_mano(clon, agente_idx, carta, bots)
                acumulado[carta.id] += puntos

        n = len(mundos)
        scores = {cid: total / n for cid, total in acumulado.items()}
    else:
        # Fallback: PIMC con sampling
        scores = _puntaje_esperado_por_carta(
            motor, agente_idx, legales,
            vacios=vacios,
            num_mundos=fallback_mundos,
            rng=rng,
            crear_bots=crear_bots,
        )

    mejor = min(legales, key=lambda c: scores[c.id])
    return mejor, scores, exacto


# ──────────────────────────────────────────────────────────────
# Tipos para análisis detallado
# ──────────────────────────────────────────────────────────────

@dataclass
class ResultadoDecision:
    """Resultado del análisis de una decisión."""
    baza: int
    situacion: str            # "liderar", "seguir", "descartar"
    idx_jugador: int
    cartas_legales: List[str]  # nombres de cartas
    scores: Dict[str, float]   # carta → puntuación esperada
    carta_optima: str
    score_optimo: float
    carta_elegida: str         # lo que eligió el bot/política evaluada
    score_elegido: float
    coste: float               # score_elegido - score_optimo
    exacto: bool               # True si se usó enumeración completa
    n_mundos: int              # número de mundos evaluados
    modo_bot: str = ""


@dataclass
class PerfilMano:
    """Análisis completo de una mano."""
    seed: int
    num_bazas: int = 0
    decisiones: List[ResultadoDecision] = field(default_factory=list)
    puntuacion_final: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    pozo: bool = False
    q_capturador: int = -1


# ──────────────────────────────────────────────────────────────
# Análisis de una decisión individual
# ──────────────────────────────────────────────────────────────

def analizar_decision(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    carta_elegida: Carta,
    vacios: Optional[Dict[int, Set[int]]] = None,
    rollout_tipo: str = "experto",
    rng: Optional[np.random.Generator] = None,
    modo_bot: str = "",
) -> ResultadoDecision:
    """Analiza una decisión: todas las opciones, scores, y coste de la elección.

    Args:
        motor: Estado actual (NO se modifica).
        agente_idx: Jugador que decide.
        legales: Cartas legales.
        carta_elegida: La carta que el bot/política eligió.
        vacios: Voids conocidos.
        rollout_tipo: Tipo de rollout.
        rng: Generador aleatorio.
        modo_bot: Modo del bot para metadata.

    Returns:
        ResultadoDecision con el desglose completo.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determinar situación
    if not motor.mesa:
        situacion = "liderar"
    else:
        palo = motor.palo_de_salida
        if any(c.palo == palo for c in legales):
            situacion = "seguir"
        else:
            situacion = "descartar"

    _, scores, exacto = pimc_exacto(
        motor, agente_idx, legales,
        vacios=vacios,
        rollout_tipo=rollout_tipo,
        rng=rng,
    )

    _PALO_LABEL = {0: "T", 1: "D", 2: "P", 3: "C"}

    def _nombre(c: Carta) -> str:
        return f"{c.valor}{_PALO_LABEL[c.palo]}"

    scores_nombrados = {_nombre(c): scores[c.id] for c in legales}
    carta_optima = min(legales, key=lambda c: scores[c.id])

    n_mundos = _num_mundos_posibles(motor, agente_idx)
    if not exacto:
        n_mundos = 30  # fallback sampling

    return ResultadoDecision(
        baza=motor.numero_baza,
        situacion=situacion,
        idx_jugador=agente_idx,
        cartas_legales=[_nombre(c) for c in legales],
        scores=scores_nombrados,
        carta_optima=_nombre(carta_optima),
        score_optimo=round(scores[carta_optima.id], 2),
        carta_elegida=_nombre(carta_elegida),
        score_elegido=round(scores[carta_elegida.id], 2),
        coste=round(scores[carta_elegida.id] - scores[carta_optima.id], 2),
        exacto=exacto,
        n_mundos=n_mundos if exacto else 200,
        modo_bot=modo_bot,
    )


# ──────────────────────────────────────────────────────────────
# Perfil completo de una mano
# ──────────────────────────────────────────────────────────────

def perfil_mano(
    seed: int = 42,
    agente_idx: int = 0,
    politica_evaluada: Optional[Callable] = None,
    rollout_tipo: str = "experto",
    verbose: bool = True,
) -> PerfilMano:
    """Análisis ultra-detallado de una mano completa.

    Para cada decisión no trivial del agente, calcula el score PIMC
    de TODAS las opciones y compara con lo que eligió la política.

    Args:
        seed: Semilla para la mano.
        agente_idx: Jugador a analizar (default 0).
        politica_evaluada: Callable(motor, idx, legales) → Carta.
                           Si None, usa BotExperto.
        rollout_tipo: Política de rollout para PIMC.
        verbose: Imprimir resultados en tiempo real.

    Returns:
        PerfilMano con todas las decisiones analizadas.
    """
    import random
    random.seed(seed)
    rng = np.random.default_rng(seed)

    if politica_evaluada is None:
        from src.agentes.bot_experto import BotExperto
        bots_eval = [BotExperto() for _ in range(4)]
        politica_evaluada = bots_eval[agente_idx]

    # Políticas para oponentes
    from src.agentes.bot_experto import BotExperto
    bots_op = [BotExperto() for _ in range(4)]

    motor = MotorCorazones()
    motor.repartir()

    perfil = PerfilMano(seed=seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  PERFIL DE MANO — seed={seed} | agente=J{agente_idx}")
        print(f"{'='*70}")

    baza_num = 1

    while motor._mano_activa:
        if len(motor.mesa) == 4:
            motor.resolver_baza()
            if not motor._mano_activa:
                break
            baza_num += 1

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        if not legales:
            break

        if idx == agente_idx:
            # Capturar voids ANTES de que el bot modifique su estado
            if isinstance(bots_op[agente_idx], BotExperto):
                modo = bots_op[agente_idx]._modo(motor, agente_idx)
                vacios = {
                    i: set(bots_op[agente_idx]._vacios[i])
                    for i in range(1, 4)
                    if bots_op[agente_idx]._vacios[i]
                }
            else:
                modo = ""
                vacios = {}

            carta_elegida = politica_evaluada(motor, idx, legales)

            if len(legales) >= 2:
                resultado = analizar_decision(
                    motor, agente_idx, legales, carta_elegida,
                    vacios=vacios,
                    rollout_tipo=rollout_tipo,
                    rng=rng,
                    modo_bot=modo,
                )
                perfil.decisiones.append(resultado)

                if verbose:
                    _PALO = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}

                    def _nc(c: Carta) -> str:
                        _V = {11: "J", 12: "Q", 13: "K", 14: "A"}
                        v = _V.get(c.valor, str(c.valor))
                        return f"{v}{_PALO[c.palo]}"

                    exacto_str = "✓" if resultado.exacto else "~"
                    print(f"\n  ── Baza {resultado.baza} | {resultado.situacion} "
                          f"| {resultado.n_mundos:,} mundos {exacto_str} ──")
                    print(f"  Mano: {' '.join(_nc(c) for c in legales)}")

                    # Ordenar scores de menor a mayor
                    sorted_scores = sorted(
                        resultado.scores.items(), key=lambda x: x[1]
                    )
                    for nombre, score in sorted_scores:
                        marker = ""
                        if nombre == resultado.carta_optima:
                            marker = " ← ÓPTIMA"
                        if nombre == resultado.carta_elegida:
                            marker = " ← ELEGIDA"
                        print(f"    {nombre:>5}: {score:6.2f} pts{marker}")

                    if resultado.coste > 0:
                        print(f"  ⚠ Coste: {resultado.coste:+.2f} pts extras")
                    else:
                        print(f"  ✅ Decisión óptima")

            motor.jugar_carta(idx, carta_elegida)
        else:
            carta = bots_op[idx](motor, idx, legales)
            motor.jugar_carta(idx, carta)

    # Puntuación final
    perfil.puntuacion_final = motor.calcular_puntuacion_mano()
    perfil.num_bazas = baza_num

    # Detectar pozo y Q♠
    for i, j in enumerate(motor.jugadores):
        for c in j.bazas_ganadas:
            if c.es_dama_de_picas:
                perfil.q_capturador = i

    puntuacion = perfil.puntuacion_final
    if puntuacion.count(26) == 3 and puntuacion.count(0) == 1:
        perfil.pozo = True

    if verbose:
        print(f"\n  ── Resultado final ──")
        print(f"  Puntuación: {perfil.puntuacion_final}")
        if perfil.pozo:
            print(f"  🌕 SHOOTING THE MOON (J{perfil.puntuacion_final.index(0)})")
        elif perfil.q_capturador >= 0:
            print(f"  ♠ Q♠ capturada por J{perfil.q_capturador}")

        total_cost = sum(d.coste for d in perfil.decisiones if d.coste > 0)
        n_errores = sum(1 for d in perfil.decisiones if d.coste > 0)
        n_exactas = sum(1 for d in perfil.decisiones if d.exacto)
        print(f"\n  Errores: {n_errores}/{len(perfil.decisiones)} | "
              f"Coste total: {total_cost:.1f} pts | "
              f"Decisiones exactas: {n_exactas}/{len(perfil.decisiones)}")

    return perfil


# ──────────────────────────────────────────────────────────────
# Evaluación de mano con combinaciones de bots
# ──────────────────────────────────────────────────────────────

@dataclass
class ResultadoMano:
    """Resultado de una mano evaluada."""
    seed: int
    puntuacion: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    pozo: bool = False
    q_capturador: int = -1
    n_bazas: int = 0
    decisiones_exactas: int = 0
    decisiones_total: int = 0
    coste_total: float = 0.0


def evaluar_mano(
    seed: int = 42,
    agente_idx: int = 0,
    politica_evaluada: Optional[Callable] = None,
    rollout_tipo: str = "experto",
    analizar: bool = True,
) -> ResultadoMano:
    """Evalúa una mano y opcionalmente analiza cada decisión.

    Args:
        seed: Semilla para la mano.
        agente_idx: Jugador a evaluar.
        politica_evaluada: Política a evaluar (None = BotExperto).
        rollout_tipo: Tipo de rollout para PIMC.
        analizar: Si True, analiza cada decisión con PIMC.

    Returns:
        ResultadoMano con puntuación y métricas.
    """
    import random
    random.seed(seed)

    if politica_evaluada is None:
        from src.agentes.bot_experto import BotExperto
        bots = [BotExperto() for _ in range(4)]
        politica_evaluada = bots[agente_idx]
    else:
        from src.agentes.bot_experto import BotExperto
        bots = [BotExperto() for _ in range(4)]
        bots[agente_idx] = politica_evaluada  # type: ignore[assignment]

    motor = MotorCorazones()
    motor.repartir()

    resultado = ResultadoMano(seed=seed)

    while motor._mano_activa:
        if len(motor.mesa) == 4:
            motor.resolver_baza()
            if not motor._mano_activa:
                break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        if not legales:
            break

        carta = bots[idx](motor, idx, legales)  # type: ignore[operator]

        if analizar and idx == agente_idx and len(legales) >= 2:
            resultado.decisiones_total += 1
            # Solo analizamos con exactitud si baza >= 8
            if motor.numero_baza >= _BAZA_MINIMA_EXACTA:
                r = analizar_decision(
                    motor, agente_idx, legales, carta,
                    rollout_tipo=rollout_tipo,
                )
                if r.exacto:
                    resultado.decisiones_exactas += 1
                if r.coste > 0:
                    resultado.coste_total += r.coste

        motor.jugar_carta(idx, carta)

    resultado.puntuacion = motor.calcular_puntuacion_mano()
    resultado.n_bazas = motor.numero_baza

    # Detectar Q♠ y pozo
    for i, j in enumerate(motor.jugadores):
        for c in j.bazas_ganadas:
            if c.es_dama_de_picas:
                resultado.q_capturador = i

    pts = resultado.puntuacion
    if pts.count(26) == 3 and pts.count(0) == 1:
        resultado.pozo = True

    return resultado


# ──────────────────────────────────────────────────────────────
# Comparación A/B de políticas
# ──────────────────────────────────────────────────────────────

@dataclass
class ComparacionPoliticas:
    """Resultado de comparar dos políticas."""
    nombre_a: str
    nombre_b: str
    puntuacion_a: List[float] = field(default_factory=list)
    puntuacion_b: List[float] = field(default_factory=list)
    manos_a_favor_de_a: int = 0
    manos_a_favor_de_b: int = 0
    empates: int = 0
    media_a: float = 0.0
    media_b: float = 0.0


def comparar_politicas(
    politica_a: Callable,
    politica_b: Callable,
    nombre_a: str = "A",
    nombre_b: str = "B",
    num_manos: int = 100,
    seed_inicial: int = 42,
    rollout_tipo: str = "experto",
    verbose: bool = True,
) -> ComparacionPoliticas:
    """Compara dos políticas jugando las mismas manos (misma semilla).

    Cada mano se juega DOS veces con la misma semilla:
    una con política A como J0, otra con política B como J0.
    Los oponentes son siempre BotExperto.

    Args:
        politica_a: Primera política (callable).
        politica_b: Segunda política.
        nombre_a: Etiqueta para A.
        nombre_b: Etiqueta para B.
        num_manos: Número de manos a evaluar.
        seed_inicial: Semilla base.
        rollout_tipo: Tipo de rollout (no usado aquí,预留).
        verbose: Mostrar progreso.

    Returns:
        ComparacionPoliticas con los resultados.
    """
    pts_a = []
    pts_b = []
    a_favor_a = 0
    a_favor_b = 0
    empates = 0

    for i in range(num_manos):
        seed = seed_inicial + i

        ra = evaluar_mano(seed=seed, agente_idx=0,
                          politica_evaluada=politica_a, analizar=False)
        rb = evaluar_mano(seed=seed, agente_idx=0,
                          politica_evaluada=politica_b, analizar=False)

        pts_a.append(ra.puntuacion[0])
        pts_b.append(rb.puntuacion[0])

        if ra.puntuacion[0] < rb.puntuacion[0]:
            a_favor_a += 1
        elif rb.puntuacion[0] < ra.puntuacion[0]:
            a_favor_b += 1
        else:
            empates += 1

        if verbose and (i + 1) % max(1, num_manos // 10) == 0:
            ma = sum(pts_a) / len(pts_a)
            mb = sum(pts_b) / len(pts_b)
            print(f"  Mano {i+1:4d}/{num_manos} | "
                  f"{nombre_a}: {ma:.2f} | {nombre_b}: {mb:.2f} | "
                  f"Δ: {ma-mb:+.2f}")

    return ComparacionPoliticas(
        nombre_a=nombre_a,
        nombre_b=nombre_b,
        puntuacion_a=pts_a,
        puntuacion_b=pts_b,
        manos_a_favor_de_a=a_favor_a,
        manos_a_favor_de_b=a_favor_b,
        empates=empates,
        media_a=sum(pts_a) / len(pts_a),
        media_b=sum(pts_b) / len(pts_b),
    )


# ──────────────────────────────────────────────────────────────
# Estadísticas agregadas de muchas manos
# ──────────────────────────────────────────────────────────────

@dataclass
class EstadisticasManos:
    """Estadísticas agregadas de N manos."""
    total_manos: int
    puntuacion_media: float
    puntuacion_std: float
    tasa_cero: float          # fracción de manos con 0 pts
    tasa_pozo: float          # fracción de manos con pozo
    tasa_q_capturada: float   # fracción donde el agente capturó Q♠
    distribucion: Dict[int, int]  # puntuación → frecuencia


def estadisticas_manos(
    politica: Callable,
    num_manos: int = 200,
    seed_inicial: int = 42,
    agente_idx: int = 0,
    verbose: bool = True,
) -> EstadisticasManos:
    """Calcula estadísticas agregadas sobre muchas manos.

    Args:
        politica: Política a evaluar.
        num_manos: Número de manos.
        seed_inicial: Semilla base.
        agente_idx: Jugador a evaluar.
        verbose: Mostrar progreso.

    Returns:
        EstadisticasManos con todas las métricas.
    """
    puntuaciones = []
    ceros = 0
    pozos = 0
    q_capturadas = 0
    distribucion: Dict[int, int] = defaultdict(int)

    for i in range(num_manos):
        seed = seed_inicial + i
        r = evaluar_mano(seed=seed, agente_idx=agente_idx,
                         politica_evaluada=politica, analizar=False)

        pts = r.puntuacion[agente_idx]
        puntuaciones.append(pts)
        distribucion[pts] += 1

        if pts == 0:
            ceros += 1
        if r.pozo and r.puntuacion.index(0) == agente_idx:
            pozos += 1
        if r.q_capturador == agente_idx:
            q_capturadas += 1

        if verbose and (i + 1) % max(1, num_manos // 10) == 0:
            media = sum(puntuaciones) / len(puntuaciones)
            print(f"  Mano {i+1:4d}/{num_manos} | "
                  f"Media: {media:.2f} | 0s: {ceros} | "
                  f"Q♠: {q_capturadas}")

    arr = np.array(puntuaciones)
    return EstadisticasManos(
        total_manos=num_manos,
        puntuacion_media=float(np.mean(arr)),
        puntuacion_std=float(np.std(arr)),
        tasa_cero=ceros / num_manos,
        tasa_pozo=pozos / num_manos,
        tasa_q_capturada=q_capturadas / num_manos,
        distribucion=dict(distribucion),
    )
