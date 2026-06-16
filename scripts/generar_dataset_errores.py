#!/usr/bin/env python
"""
Generador de dataset para Behavioral Cloning desde errores de BotExperto.

Por cada mano:
  1. BotExperto decide cada jugada.
  2. PIMC (con enumeración exacta cuando viable) evalúa todas las opciones.
  3. Si BotExperto eligió subóptimo, se registra el par (observación, acción_óptima).
  4. Cada divergencia se clasifica en un patrón de error.

Outputs:
  - datasets/errores_v1.npz       — pares (obs, acción_óptima) para BC
  - datasets/errores_v1_patrones.md — catálogo de patrones con reglas sugeridas
  - datasets/errores_v1_rewards.md  — mapeo patrones → recompensas RL
  - datasets/errores_v1.json        — datos completos en JSON

Uso:
  python scripts/generar_dataset_errores.py --manos 500 --output errores_v1
  python scripts/generar_dataset_errores.py --manos 200 --output v2 --umbral 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_evasivo
from src.mcts.analisis import pimc_exacto, _num_mundos_posibles
from src.entorno.observacion import ObservacionBuilder
from src.entorno.recompensas import RewardConfig

# ──────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────

_PALO_LABEL = {0: "T", 1: "D", 2: "P", 3: "C"}
_TREBOL, _DIAMANTE, _PICA, _CORAZON = 0, 1, 2, 3

# ──────────────────────────────────────────────────────────────
# Tipos de datos
# ──────────────────────────────────────────────────────────────


@dataclass
class EjemploBC:
    """Un par (observación, acción_óptima) para Behavioral Cloning."""
    obs: np.ndarray          # vector de observación (194,)
    accion_optima: int       # id de la carta óptima según PIMC
    baza: int
    situacion: str
    patron: str              # categoría de error
    coste: float


@dataclass
class PatronError:
    """Un patrón de error detectado."""
    nombre: str              # identificador único
    descripcion: str         # descripción legible
    situacion: str           # liderar/seguir/descartar
    count: int = 0
    coste_total: float = 0.0
    coste_max: float = 0.0
    bazas_afectadas: List[int] = field(default_factory=list)
    ejemplos: List[Dict] = field(default_factory=list)  # hasta 5 ejemplos

    @property
    def coste_medio(self) -> float:
        return self.coste_total / self.count if self.count > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Clasificador de patrones
# ──────────────────────────────────────────────────────────────


def _valor_carta(c: Carta) -> int:
    return c.valor


def _es_alta(c: Carta) -> bool:
    return c.valor >= 11


def _es_media(c: Carta) -> bool:
    return 6 <= c.valor <= 10


def _es_baja(c: Carta) -> bool:
    return c.valor <= 5


def clasificar_error(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    carta_bot: Carta,
    carta_optima: Carta,
    situacion: str,
    modo_bot: str,
) -> str:
    """Clasifica una divergencia en un patrón de error.

    Returns:
        Nombre del patrón (string clave).
    """
    baza = motor.numero_baza
    palo_salida = motor.palo_de_salida
    puntos_mesa = sum(c.puntos for _, c in motor.mesa)

    # ── LIDERAR ──────────────────────────────────────────────
    if situacion == "liderar":
        # Lideró carta sin puntos en vez de corazón en baza tardía
        if (baza >= 8 and carta_bot.puntos == 0
                and carta_optima.es_corazon and _es_media(carta_optima)):
            return "liderar_corazon_tardio"

        # Lideró corazón alto en vez de corazón bajo en baza tardía
        if (baza >= 8 and carta_bot.es_corazon and carta_optima.es_corazon
                and carta_bot.valor > carta_optima.valor
                and _es_alta(carta_bot)):
            return "liderar_corazon_demasiado_alto"

        # Lideró alta de palo seguro en vez de baja (debía ceder lead)
        if (baza >= 9 and carta_bot.puntos == 0 and carta_optima.puntos == 0
                and carta_bot.valor > carta_optima.valor
                and carta_bot.valor >= 10):
            return "liderar_quemar_cuando_debe_ceder"

        # Lideró baja cuando debía quemar alta (safe suit burning)
        if (baza >= 5 and carta_bot.puntos == 0 and carta_optima.puntos == 0
                and carta_bot.valor < carta_optima.valor
                and _es_alta(carta_optima)):
            return "liderar_no_quema_maxima"

        # Lideró pica (no Q♠) cuando debía liderar corazón o trébol/diamante
        if (carta_bot.palo == _PICA and not carta_bot.es_dama_de_picas
                and carta_optima.palo != _PICA):
            return "liderar_pica_innecesaria"

        # Lideró Q♠ mal (debía/no debía)
        if carta_bot.es_dama_de_picas:
            return "liderar_q_espadas_mal_momento"
        if carta_optima.es_dama_de_picas:
            return "liderar_no_suelta_q_espadas"

        return "liderar_otro"

    # ── SEGUIR PALO ──────────────────────────────────────────
    if situacion == "seguir":
        # Jugó baja en vez de quemar A/K del palo en baza sin puntos
        if (puntos_mesa == 0 and carta_optima.puntos == 0
                and _es_alta(carta_optima) and not _es_alta(carta_bot)
                and carta_bot.palo == carta_optima.palo):
            return "seguir_no_quema_alta_en_baza_limpia"

        # Jugó alta que gana en vez de baja que pierde (con puntos en mesa)
        if puntos_mesa > 0 and carta_bot.valor > carta_optima.valor:
            return "seguir_gana_baza_con_puntos"

        # Jugó Q♠ cuando debía jugar otra pica
        if carta_bot.es_dama_de_picas:
            return "seguir_q_espadas_mal_momento"

        # Jugó A♠/K♠ cuando debía jugar Q♠ (dump)
        if (carta_optima.es_dama_de_picas
                and carta_bot.palo == _PICA and carta_bot.valor >= 13):
            return "seguir_no_suelta_q_con_altas"

        # Jugó baja en vez de alta cuando todas ganan (debía quemar)
        if (carta_bot.valor < carta_optima.valor
                and carta_bot.palo == carta_optima.palo
                and puntos_mesa == 0):
            return "seguir_no_quema_maxima_forzada"

        return "seguir_otro"

    # ── DESCARTAR ────────────────────────────────────────────
    if situacion == "descartar":
        # Descartó corazón cuando debía descartar Q♠
        if carta_optima.es_dama_de_picas:
            return "descarte_no_suelta_q_espadas"

        # Descartó Q♠ cuando no debía
        if carta_bot.es_dama_de_picas:
            return "descarte_q_espadas_prematuro"

        # Descartó A♣/A♦ en vez de corazón alto
        if (carta_bot.valor == 14 and carta_bot.palo in (_TREBOL, _DIAMANTE)
                and carta_optima.es_corazon and _es_alta(carta_optima)):
            return "descarte_ases_en_vez_de_corazones"

        # Descartó corazón bajo en vez de K♠/A♠ (con Q♠ activa)
        if (carta_bot.es_corazon and carta_optima.palo == _PICA
                and carta_optima.valor >= 13 and puntos_mesa == 0):
            return "descarte_no_suelta_picas_altas"

        # Descartó sin puntos en vez de corazón (corazones rotos)
        if (carta_bot.puntos == 0 and carta_optima.es_corazon
                and motor.corazones_rotos and puntos_mesa == 0):
            return "descarte_no_suelta_corazon_bajo"

        return "descarte_otro"

    return "sin_clasificar"


# ──────────────────────────────────────────────────────────────
# Análisis de una mano con captura de dataset
# ──────────────────────────────────────────────────────────────


def _crear_bots_rollout():
    return {i: bot_evasivo for i in range(4)}


def _situacion(motor: MotorCorazones, agente: int) -> str:
    if not motor.mesa:
        return "liderar"
    palo = motor.palo_de_salida
    if any(c.palo == palo for c in motor.jugadores[agente].mano):
        return "seguir"
    return "descartar"


def _nombre(c: Carta) -> str:
    return f"{c.valor}{_PALO_LABEL[c.palo]}"


def analizar_mano_con_dataset(
    motor: MotorCorazones,
    seed: int,
    num_mundos_fallback: int,
    rng: np.random.Generator,
    umbral: float,
    obs_builder: ObservacionBuilder,
    puntuaciones_historicas: List[int],
    puntos_mano: List[int],
) -> Tuple[List[EjemploBC], Dict[str, PatronError]]:
    """Ejecuta una mano y captura ejemplos BC + patrones de error."""
    ejemplos: List[EjemploBC] = []
    patrones: Dict[str, PatronError] = {}

    while motor._mano_activa:
        if len(motor.mesa) == 4:
            motor.resolver_baza()
            if not motor._mano_activa:
                break

        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        if not legales:
            break

        if idx == 0:
            sit = _situacion(motor, 0)
            # Usar BotExperto para decidir (creamos uno nuevo; el análisis
            # de patrones no requiere estado persistente entre bazas)
            bot = BotExperto()
            carta_bot = bot(motor, 0, legales)
            modo = bot._modo(motor, 0)

            if len(legales) >= 2:
                # Capturar observación actual
                obs = obs_builder.construir(
                    motor, 0,
                    vacios=[set() for _ in range(4)],
                    puntuacion_historica=puntuaciones_historicas,
                    puntos_mano_actual=puntos_mano,
                    dama_picas_en=None,
                )

                # PIMC exacto
                _, scores, exacto = pimc_exacto(
                    motor, 0, legales,
                    rng=rng,
                    crear_bots=_crear_bots_rollout,
                    fallback_mundos=num_mundos_fallback,
                )
                carta_optima = min(legales, key=lambda c: scores[c.id])
                score_bot = scores[carta_bot.id]
                score_opt = scores[carta_optima.id]
                coste = score_bot - score_opt

                if coste > umbral:
                    # Clasificar el patrón
                    patron = clasificar_error(
                        motor, 0, legales, carta_bot, carta_optima, sit, modo,
                    )

                    # Guardar ejemplo BC (observación → acción óptima)
                    ejemplos.append(EjemploBC(
                        obs=obs.copy(),
                        accion_optima=carta_optima.id,
                        baza=motor.numero_baza,
                        situacion=sit,
                        patron=patron,
                        coste=coste,
                    ))

                    # Acumular en patrón
                    if patron not in patrones:
                        patrones[patron] = PatronError(
                            nombre=patron,
                            descripcion="",
                            situacion=sit,
                        )
                    p = patrones[patron]
                    p.count += 1
                    p.coste_total += coste
                    p.coste_max = max(p.coste_max, coste)
                    p.bazas_afectadas.append(motor.numero_baza)
                    if len(p.ejemplos) < 5:
                        p.ejemplos.append({
                            "baza": motor.numero_baza,
                            "carta_bot": _nombre(carta_bot),
                            "carta_optima": _nombre(carta_optima),
                            "coste": round(coste, 2),
                            "exacto": exacto,
                        })

            motor.jugar_carta(0, carta_bot)
        else:
            # Oponentes: jugar primera carta legal (rápido, no afecta análisis)
            motor.jugar_carta(idx, legales[0])

    return ejemplos, patrones


# ──────────────────────────────────────────────────────────────
# Descripciones de patrones
# ──────────────────────────────────────────────────────────────

DESCRIPCIONES_PATRONES = {
    "liderar_corazon_tardio": (
        "En baza ≥8, lideró carta sin puntos cuando debió liderar corazón medio. "
        "PIMC dice que dump de corazón es más seguro que guardarlo."
    ),
    "liderar_corazon_demasiado_alto": (
        "En baza tardía, lideró corazón alto (J/Q/K/A) cuando debió liderar "
        "corazón más bajo. El corazón alto es útil para control, el bajo para dump."
    ),
    "liderar_quemar_cuando_debe_ceder": (
        "En baza ≥9, lideró carta alta de palo seguro (gana la baza) cuando "
        "debía liderar baja (ceder el lead). Ganar fuerza a liderar de nuevo."
    ),
    "liderar_no_quema_maxima": (
        "Tenía la máxima de un palo seguro y no la lideró. Perdió oportunidad "
        "de ganar baza limpia y quemar liability."
    ),
    "liderar_pica_innecesaria": (
        "Lideró pica (no Q♠) cuando había mejores opciones. Las picas son "
        "peligrosas mientras Q♠ está activa."
    ),
    "liderar_q_espadas_mal_momento": (
        "Lideró Q♠ en momento inadecuado (demasiado pronto o cuando era "
        "máxima en picas)."
    ),
    "liderar_no_suelta_q_espadas": (
        "No lideró Q♠ cuando debía (baza tardía con K♠/A♠ en circulación)."
    ),
    "liderar_otro": (
        "Error al liderar no clasificado en categorías específicas."
    ),

    "seguir_no_quema_alta_en_baza_limpia": (
        "En baza sin puntos, jugó carta baja/media en vez de A/K del palo. "
        "Debía quemar la carta alta en baza limpia para eliminar liability futura."
    ),
    "seguir_gana_baza_con_puntos": (
        "Jugó carta que gana la baza cuando había puntos en mesa, pudiendo "
        "jugar una perdedora. Capturó puntos innecesariamente."
    ),
    "seguir_q_espadas_mal_momento": (
        "Jugó Q♠ siguiendo picas cuando no era óptimo."
    ),
    "seguir_no_suelta_q_con_altas": (
        "Tenía Q♠ y A♠/K♠, y jugó A♠/K♠ en vez de Q♠. PIMC dice que "
        "era mejor soltar Q♠ para que otro la capture."
    ),
    "seguir_no_quema_maxima_forzada": (
        "Forzado a ganar la baza (todas sus cartas > ganadora actual), "
        "pero jugó la más baja en vez de la más alta. Debía quemar la máxima."
    ),
    "seguir_otro": (
        "Error al seguir palo no clasificado en categorías específicas."
    ),

    "descarte_no_suelta_q_espadas": (
        "Pudiendo descartar Q♠ en baza con puntos, no lo hizo. Perdió "
        "oportunidad de endosar 13 pts a un rival."
    ),
    "descarte_q_espadas_prematuro": (
        "Descartó Q♠ en momento inadecuado (baza sin puntos, demasiado pronto)."
    ),
    "descarte_ases_en_vez_de_corazones": (
        "Descartó A♣/A♦ en vez de corazón alto. Los ases de palos seguros "
        "ganan bazas y atraen corazones descartados."
    ),
    "descarte_no_suelta_picas_altas": (
        "Con Q♠ activa, no descartó K♠/A♠ en baza limpia. Estas cartas "
        "son peligrosas: ganan bazas de ♠ donde Q♠ puede caer encima."
    ),
    "descarte_no_suelta_corazon_bajo": (
        "Con corazones rotos, no descartó corazón bajo en baza limpia. "
        "Los corazones bajos pueden forzar ganar bazas de ♥ con puntos."
    ),
    "descarte_otro": (
        "Error al descartar no clasificado en categorías específicas."
    ),

    "sin_clasificar": (
        "Error no clasificado. Requiere análisis manual."
    ),
}


# ──────────────────────────────────────────────────────────────
# Mapeo patrones → posibles rewards/penalties RL
# ──────────────────────────────────────────────────────────────

REWARDS_SUGERIDOS = {
    "liderar_corazon_tardio": {
        "tipo": "recompensa",
        "nombre": "REWARD_LIDERAR_CORAZON_TARDIO",
        "valor_sugerido": 1.0,
        "condicion": "baza ≥ 8 AND corazones_rotos AND se lidera corazón medio (≤10)",
        "justificacion": "Dump seguro de corazones en final de mano."
    },
    "liderar_corazon_demasiado_alto": {
        "tipo": "penalización",
        "nombre": "PENALTY_LIDERAR_CORAZON_ALTO_TARDIO",
        "valor_sugerido": -1.5,
        "condicion": "baza ≥ 9 AND se lidera corazón J/Q/K/A teniendo corazones ≤10",
        "justificacion": "Corazones altos son para control, no para dump."
    },
    "liderar_quemar_cuando_debe_ceder": {
        "tipo": "penalización",
        "nombre": "PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD",
        "valor_sugerido": -2.0,
        "condicion": "baza ≥ 9 AND se gana baza sin puntos con carta alta de palo seguro",
        "justificacion": "Ganar baza en final de mano fuerza a liderar de nuevo."
    },
    "liderar_no_quema_maxima": {
        "tipo": "recompensa",
        "nombre": "REWARD_QUEMAR_MAXIMA_PALO_SEGURO",
        "valor_sugerido": 0.5,
        "condicion": "baza ≥ 6 AND se lidera la máxima de ♣/♦",
        "justificacion": "Quemar máxima en palo seguro garantiza baza limpia."
    },
    "liderar_pica_innecesaria": {
        "tipo": "penalización",
        "nombre": "PENALTY_LIDERAR_PICA_CON_Q_ACTIVA",
        "valor_sugerido": -1.0,
        "condicion": "Q♠ activa AND se lidera pica sin ser la máxima",
        "justificacion": "Liderar picas con Q♠ activa es peligroso."
    },
    "liderar_q_espadas_mal_momento": {
        "tipo": "penalización",
        "nombre": "PENALTY_LIDERAR_Q_EQUIVOCADO",
        "valor_sugerido": -3.0,
        "condicion": "Se lidera Q♠ cuando soy máxima en picas o baza < 7",
        "justificacion": "Liderar Q♠ sin K♠/A♠ en circulación = auto-13pts."
    },
    "liderar_no_suelta_q_espadas": {
        "tipo": "recompensa",
        "nombre": "REWARD_LIDERAR_Q_DUMP_SEGURO",
        "valor_sugerido": 3.0,
        "condicion": "baza ≥ 7 AND K♠/A♠ en circulación AND se lidera Q♠",
        "justificacion": "Dump de Q♠ cuando hay cobertura de K♠/A♠."
    },
    "seguir_no_quema_alta_en_baza_limpia": {
        "tipo": "recompensa",
        "nombre": "REWARD_QUEMAR_ALTA_SIGUIENDO_PALO",
        "valor_sugerido": 0.5,
        "condicion": "puntos_mesa == 0 AND se juega A/K del palo",
        "justificacion": "Ya cubierto parcialmente. Refuerzo adicional."
    },
    "seguir_gana_baza_con_puntos": {
        "tipo": "penalización",
        "nombre": "PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE",
        "valor_sugerido": -3.0,
        "condicion": "puntos_mesa > 0 AND se juega carta > ganadora teniendo perdedora",
        "justificacion": "Capturar puntos cuando había alternativa de perder."
    },
    "seguir_no_suelta_q_con_altas": {
        "tipo": "recompensa",
        "nombre": "REWARD_DUMP_Q_SIGUIENDO_PICAS",
        "valor_sugerido": 2.0,
        "condicion": "siguiendo ♠ AND se juega Q♠ AND K♠/A♠ en circulación",
        "justificacion": "Soltar Q♠ cuando hay cobertura."
    },
    "seguir_no_quema_maxima_forzada": {
        "tipo": "recompensa",
        "nombre": "REWARD_QUEMAR_MAXIMA_FORZADA",
        "valor_sugerido": 0.3,
        "condicion": "Todas las cartas ganan AND se juega la máxima",
        "justificacion": "Si vas a ganar igual, quema la más alta."
    },
    "descarte_no_suelta_q_espadas": {
        "tipo": "recompensa",
        "nombre": "REWARD_DESCARTAR_DAMA_SEGURO",
        "valor_sugerido": 5.0,  # ya existe, subir de 5.0
        "condicion": "YA EXISTE. Subir valor a 8.0.",
        "justificacion": "Descartar Q♠ en baza con puntos es crítico."
    },
    "descarte_q_espadas_prematuro": {
        "tipo": "penalización",
        "nombre": "PENALTY_DESCARTAR_Q_PREMATURO",
        "valor_sugerido": -3.0,
        "condicion": "baza < 6 AND se descarta Q♠ en baza sin puntos",
        "justificacion": "Q♠ debe guardarse para bazas con puntos."
    },
    "descarte_ases_en_vez_de_corazones": {
        "tipo": "recompensa",
        "nombre": "REWARD_DESCARTAR_ASES_SEGUROS",
        "valor_sugerido": 1.0,
        "condicion": "Se descarta A♣/A♦ en baza limpia antes que corazones",
        "justificacion": "Ases ganan bazas y atraen descartes de corazones."
    },
    "descarte_no_suelta_picas_altas": {
        "tipo": "recompensa",
        "nombre": "REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA",
        "valor_sugerido": 2.0,
        "condicion": "Q♠ activa AND se descarta K♠/A♠ en baza limpia",
        "justificacion": "K♠/A♠ son liability con Q♠ activa."
    },
    "descarte_no_suelta_corazon_bajo": {
        "tipo": "recompensa",
        "nombre": "REWARD_DESCARTAR_CORAZON_BAJO_ROTO",
        "valor_sugerido": 0.5,
        "condicion": "corazones_rotos AND se descarta corazón en baza limpia",
        "justificacion": "Con corazones rotos, cualquier corazón es peligroso."
    },
}


# ──────────────────────────────────────────────────────────────
# Generación de reportes
# ──────────────────────────────────────────────────────────────

def generar_patrones_md(
    patrones: Dict[str, PatronError],
    total_manos: int,
    output_path: str,
) -> str:
    """Genera un reporte Markdown con el catálogo de patrones."""
    lines = [
        f"# Catálogo de Patrones de Error — BotExperto",
        f"",
        f"**Manos analizadas:** {total_manos}  ",
        f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M')}  ",
        f"**Total patrones detectados:** {len(patrones)}  ",
        f"",
        f"---",
        f"",
    ]

    # Ordenar por coste total descendente
    sorted_patrones = sorted(
        patrones.items(), key=lambda x: -x[1].coste_total
    )

    for nombre, p in sorted_patrones:
        desc = DESCRIPCIONES_PATRONES.get(nombre, "Sin descripción.")
        lines.append(f"## `{nombre}`")
        lines.append(f"")
        lines.append(f"- **Ocurrencias:** {p.count}")
        lines.append(f"- **Coste total:** {p.coste_total:.1f} pts")
        lines.append(f"- **Coste medio:** {p.coste_medio:.2f} pts")
        lines.append(f"- **Coste máximo:** {p.coste_max:.2f} pts")
        lines.append(f"- **Situación:** {p.situacion}")
        lines.append(f"")
        lines.append(f"**Descripción:** {desc}")
        lines.append(f"")

        if p.ejemplos:
            lines.append(f"**Ejemplos:**")
            lines.append(f"")
            lines.append(f"| Baza | Bot | PIMC | Coste | Exacto |")
            lines.append(f"|------|-----|------|-------|--------|")
            for ej in p.ejemplos:
                exacto_str = "✓" if ej["exacto"] else "~"
                lines.append(
                    f"| {ej['baza']} | {ej['carta_bot']} | {ej['carta_optima']} "
                    f"| {ej['coste']:.1f} | {exacto_str} |"
                )
            lines.append(f"")

    md = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


def generar_rewards_md(output_path: str) -> str:
    """Genera reporte de mapeo patrones → recompensas RL."""
    lines = [
        f"# Mapeo Patrones → Recompensas/Castigos RL",
        f"",
        f"Se sugiere añadir las siguientes señales al `RewardConfig` y",
        f"a `CalculadoraRecompensas` para que el modelo RL aprenda estas",
        f"distinciones que el BotExperto actualmente no hace.",
        f"",
        f"---",
        f"",
    ]

    for patron, info in sorted(REWARDS_SUGERIDOS.items()):
        tipo_icon = "🟢" if info["tipo"] == "recompensa" else "🔴"
        lines.append(f"## {tipo_icon} `{info['nombre']}`")
        lines.append(f"")
        lines.append(f"- **Patrón asociado:** `{patron}`")
        lines.append(f"- **Tipo:** {info['tipo']}")
        lines.append(f"- **Valor sugerido:** {info['valor_sugerido']:+.1f}")
        lines.append(f"- **Condición:** `{info['condicion']}`")
        lines.append(f"- **Justificación:** {info['justificacion']}")
        lines.append(f"")

    # Tabla resumen
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Tabla Resumen")
    lines.append(f"")
    lines.append(f"| Reward/Penalty | Tipo | Valor | Patrón |")
    lines.append(f"|---------------|------|-------|--------|")
    for patron, info in sorted(REWARDS_SUGERIDOS.items()):
        lines.append(
            f"| `{info['nombre']}` | {info['tipo']} | {info['valor_sugerido']:+.1f} "
            f"| `{patron}` |"
        )

    md = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    return md


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset BC y catálogo de patrones desde errores de BotExperto"
    )
    parser.add_argument("--manos", type=int, default=300,
                        help="Manos a analizar (default: 300)")
    parser.add_argument("--mundos", type=int, default=30,
                        help="Mundos PIMC fallback por decisión (default: 30)")
    parser.add_argument("--umbral", type=float, default=0.3,
                        help="Coste mínimo para registrar (default: 0.3 pts)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="errores_v1",
                        help="Nombre base para archivos de salida")
    parser.add_argument("--output-dir", default="datasets",
                        help="Directorio de salida")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    obs_builder = ObservacionBuilder(dim=194)

    output_dir = os.path.join(_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, args.output)

    print(f"{'='*65}")
    print(f"  GENERADOR DE DATASET + CATÁLOGO DE PATRONES")
    print(f"{'='*65}")
    print(f"  Manos: {args.manos} | Mundos fallback: {args.mundos}")
    print(f"  Umbral: {args.umbral} pts | Seed: {args.seed}")
    print(f"  Output: {base}_*")
    print()

    todos_ejemplos: List[EjemploBC] = []
    todos_patrones: Dict[str, PatronError] = {}
    total_decisiones = 0
    exactas_count = 0

    t0 = time.time()

    for i in range(args.manos):
        seed = args.seed + i
        motor = MotorCorazones()
        motor.repartir()

        puntuaciones = [0, 0, 0, 0]
        puntos_mano = [0, 0, 0, 0]

        ejemplos, patrones_mano = analizar_mano_con_dataset(
            motor, seed, args.mundos, rng, args.umbral,
            obs_builder, puntuaciones, puntos_mano,
        )

        todos_ejemplos.extend(ejemplos)

        # Merge de patrones
        for nombre, p in patrones_mano.items():
            if nombre not in todos_patrones:
                todos_patrones[nombre] = PatronError(
                    nombre=nombre,
                    descripcion=DESCRIPCIONES_PATRONES.get(nombre, ""),
                    situacion=p.situacion,
                )
            tp = todos_patrones[nombre]
            tp.count += p.count
            tp.coste_total += p.coste_total
            tp.coste_max = max(tp.coste_max, p.coste_max)
            tp.bazas_afectadas.extend(p.bazas_afectadas)
            # Mantener hasta 5 ejemplos en total
            while len(tp.ejemplos) < 5 and p.ejemplos:
                tp.ejemplos.append(p.ejemplos.pop(0))

        if (i + 1) % max(1, args.manos // 10) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.manos - i - 1) / rate
            print(f"  Mano {i+1:4d}/{args.manos} | "
                  f"Ejemplos BC: {len(todos_ejemplos):5d} | "
                  f"Patrones: {len(todos_patrones):2d} | "
                  f"ETA: {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  ✅ Completado en {elapsed:.1f}s")
    print(f"  Total ejemplos BC: {len(todos_ejemplos):,}")
    print(f"  Total patrones: {len(todos_patrones)}")

    # ── Guardar dataset NPZ ──────────────────────────────────
    if todos_ejemplos:
        obs_array = np.stack([e.obs for e in todos_ejemplos])
        acc_array = np.array([e.accion_optima for e in todos_ejemplos])
        npz_path = f"{base}.npz"
        np.savez_compressed(npz_path, obs=obs_array, acciones=acc_array)
        print(f"  📦 Dataset BC: {npz_path} ({obs_array.shape[0]:,} ejemplos, "
              f"obs_dim={obs_array.shape[1]})")

    # ── Guardar JSON ─────────────────────────────────────────
    json_data = {
        "meta": {
            "manos": args.manos,
            "mundos_fallback": args.mundos,
            "umbral": args.umbral,
            "seed": args.seed,
            "total_ejemplos": len(todos_ejemplos),
            "elapsed_s": round(elapsed, 1),
        },
        "patrones": {
            nombre: {
                "count": p.count,
                "coste_total": round(p.coste_total, 1),
                "coste_medio": round(p.coste_medio, 2),
                "coste_max": round(p.coste_max, 2),
                "situacion": p.situacion,
                "descripcion": DESCRIPCIONES_PATRONES.get(nombre, ""),
            }
            for nombre, p in sorted(
                todos_patrones.items(), key=lambda x: -x[1].coste_total
            )
        },
    }
    json_path = f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  📋 JSON: {json_path}")

    # ── Generar reportes Markdown ────────────────────────────
    md_path = f"{base}_patrones.md"
    generar_patrones_md(todos_patrones, args.manos, md_path)
    print(f"  📄 Patrones: {md_path}")

    rewards_path = f"{base}_rewards.md"
    generar_rewards_md(rewards_path)
    print(f"  📄 Rewards RL: {rewards_path}")

    # ── Resumen en consola ───────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TOP 10 PATRONES POR COSTE TOTAL")
    print(f"{'='*65}")
    sorted_p = sorted(
        todos_patrones.items(), key=lambda x: -x[1].coste_total
    )[:10]
    for nombre, p in sorted_p:
        desc = DESCRIPCIONES_PATRONES.get(nombre, "")
        desc_short = desc[:80] + "..." if len(desc) > 80 else desc
        print(f"  {nombre:40s} | {p.count:4d} err | "
              f"total={p.coste_total:6.1f} | med={p.coste_medio:.2f}")
        print(f"    → {desc_short}")

    print(f"\n  ✅ Todo generado en {output_dir}/")
    print(f"     {args.output}.npz      — Dataset BC (obs → acción óptima)")
    print(f"     {args.output}.json     — Datos estructurados")
    print(f"     {args.output}_patrones.md — Catálogo de patrones + reglas")
    print(f"     {args.output}_rewards.md  — Mapeo a recompensas RL")


if __name__ == "__main__":
    main()
