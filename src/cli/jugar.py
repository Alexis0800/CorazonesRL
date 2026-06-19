"""
Interfaz de línea de comandos — Juego interactivo humano vs modelo RL.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.dominio.carta import Carta
from src.entorno.dimensiones import DIM_ENTORNO
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
from src.torneo.normalizacion import normalizar_obs_desde_archivo

# ----------------------------------------------------------------
# Helpers de visualización
# ----------------------------------------------------------------

_PALO_SIMBOLO = {0: "♣", 1: "♦", 2: "♠", 3: "♥"}
_VALOR_SIMBOLO = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
                  9: "9", 10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
NOMBRES_PALOS = {"♣": 0, "♦": 1, "♠": 2, "♥": 3}
_NOMBRE_VALOR = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                 "9": 9, "10": 10, "j": 11, "J": 11, "q": 12, "Q": 12,
                 "k": 13, "K": 13, "a": 14, "A": 14}


def carta_a_str(c: Carta) -> str:
    return f"{_VALOR_SIMBOLO[c.valor]}{_PALO_SIMBOLO[c.palo]}"


def parsear_carta(texto: str) -> int:
    """Convierte 'A♥' a ID 0-51."""
    texto = texto.strip()
    if not texto:
        raise ValueError("Texto vacío")
    palo_simbolo = texto[-1]
    if palo_simbolo not in NOMBRES_PALOS:
        raise ValueError(f"Palo no reconocido: {palo_simbolo}")
    palo = NOMBRES_PALOS[palo_simbolo]
    valor_str = texto[:-1]
    if valor_str not in _NOMBRE_VALOR:
        raise ValueError(f"Valor no reconocido: {valor_str}")
    valor = _NOMBRE_VALOR[valor_str]
    return palo * 13 + (valor - 2)


def mostrar_mano(mano: List[Carta]) -> None:
    """Muestra la mano del jugador humano."""
    grupos: Dict[int, List[Carta]] = {0: [], 1: [], 2: [], 3: []}
    for c in sorted(mano, key=lambda x: (x.palo, x.valor)):
        grupos[c.palo].append(c)

    for palo in range(4):
        if grupos[palo]:
            print(f"  {_PALO_SIMBOLO[palo]}: ", end="")
            cartas_str = [carta_a_str(c) for c in sorted(
                grupos[palo], key=lambda x: x.valor
            )]
            print("  ".join(cartas_str))


def ejecutar_juego(
    modelo_path: str,
    modelo_idx: int = 1,
    vecnorm_path: Optional[str] = None,
) -> None:
    """Ejecuta una partida interactiva humano vs modelo + bots."""
    from stable_baselines3 import MaskablePPO
    from src.entorno import CorazonesEnv

    print(f"Cargando modelo: {modelo_path}")
    model = MaskablePPO.load(modelo_path)

    bots = [bot_conservador, bot_agresivo, bot_evasivo]
    humano_idx = 0

    politicas = {}
    for i in range(4):
        if i == modelo_idx:
            politicas[i] = lambda motor, idx, legales, m=model, vp=vecnorm_path: _modelo_politica(
                m, idx, legales, vp)
        elif i != humano_idx:
            politicas[i] = bots[(i + modelo_idx) % 3]

    env = CorazonesEnv(agente_idx=0)  # Placeholder
    print("⚠️  Juego interactivo no implementado en versión refactorizada.")
    print("   Usa 'python jugar_contra_modelo.py' para la implementación completa.")


def _modelo_politica(model, idx, legales, vecnorm_path):
    """Política del modelo RL como callable."""
    from src.entorno.observacion import ObservacionBuilder
    obs_builder = ObservacionBuilder(dim=DIM_ENTORNO)
    obs = obs_builder.construir_desde_motor(None, idx)  # Stub
    mask = np.zeros(52, dtype=np.bool_)
    for c in legales:
        mask[c.id] = True
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return Carta._TODAS[int(action)]
