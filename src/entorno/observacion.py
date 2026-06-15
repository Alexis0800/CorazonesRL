"""
Constructores del vector de observación para el entorno Corazones.

Extraído de entorno.py para cumplir Single Responsibility Principle (SRP).
La lógica de construcción de observación está separada del ciclo de vida
del entorno Gymnasium.

Vector de observación (v6, 194 dimensiones):
    [0:52]    Mano del agente (one-hot)
    [52:104]  Mesa actual / baza en curso (one-hot)
    [104:156] Cementerio / cartas jugadas en bazas anteriores (one-hot)
    [156:172] Vacíos conocidos (4 jugadores × 4 palos)
    [172:176] Puntajes históricos globales (normalizados /100)
    [176:180] Puntos acumulados en la mano actual (normalizados /26)
    [180]     Corazones rotos (0.0 o 1.0)
    [181]     Posición en la baza actual (0.0, 0.33, 0.66, 1.0)
    [182:187] Rastreador de la Dama de Picas (one-hot, 5 estados)
    [187]     pozo_viable — ¿es viable intentar shooting the moon?
    [188]     debo_arriesgar — ¿estoy tan atrás que debo arriesgarme?
    [189]     puedo_alimentar — ¿puedo darle puntos a un rival?
    [190:194] all_void_X — ¿los 3 rivales son void en el palo X?
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class ObservacionBuilder:
    """Construye vectores de observación desde el estado del motor y entorno.

    Centraliza TODA la lógica de construcción de observación en un solo lugar,
    eliminando la duplicación que existía en entorno.py, PoliticaSB3, y
    entorno_multi.py.
    """

    DIM_V5 = 190
    DIM_V6 = 194

    def __init__(self, dim: int = 194):
        self.dim = dim

    def construir(
        self,
        motor,
        agente_idx: int,
        vacios: List[set],
        puntuacion_historica: List[int],
        puntos_mano_actual: List[int],
        dama_picas_en: Optional[int],
        pozo_viable: bool = False,
        debo_arriesgar: bool = False,
        puedo_alimentar: bool = False,
    ) -> np.ndarray:
        """Construye el vector de observación completo de `dim` dimensiones.

        Args:
            motor: Instancia de MotorCorazones.
            agente_idx: Índice del agente (0-3).
            vacios: Lista de sets de palos void para cada jugador.
            puntuacion_historica: Puntuación acumulada de cada jugador.
            puntos_mano_actual: Puntos en la mano actual de cada jugador.
            dama_picas_en: Índice del jugador que tiene la Dama de Picas (o None).
            pozo_viable: Si es viable intentar shooting the moon.
            debo_arriesgar: Si el agente debe arriesgarse.
            puedo_alimentar: Si el agente puede alimentar puntos a un rival.

        Returns:
            Array np.float32 de shape (dim,).
        """
        obs = np.zeros(self.dim, dtype=np.float32)
        a = agente_idx

        # --- Mano del agente [0:52] ---
        for c in motor.jugadores[a].mano:
            obs[c.id] = 1.0

        # --- Mesa actual [52:104] ---
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0

        # --- Cementerio [104:156] ---
        for i in range(4):
            for c in motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        # --- Vacíos [156:172] ---
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            for palo in vacios[jug_idx]:
                obs[156 + rel * 4 + palo] = 1.0

        # --- Puntajes históricos [172:176] ---
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[172 + rel] = min(puntuacion_historica[jug_idx] / 100.0, 1.0)

        # --- Puntos mano actual [176:180] ---
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[176 + rel] = min(puntos_mano_actual[jug_idx] / 26.0, 1.0)

        # --- Corazones rotos [180] ---
        obs[180] = 1.0 if motor.corazones_rotos else 0.0

        # --- Posición en la baza [181] ---
        posiciones = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
        obs[181] = posiciones.get(len(motor.mesa), 0.0)

        # --- Rastreador Dama de Picas [182:187] ---
        if dama_picas_en is None:
            obs[182] = 1.0
        else:
            rel = (dama_picas_en - a) % 4
            obs[183 + rel] = 1.0

        # --- Features v5 [187:190] ---
        if self.dim >= 190:
            obs[187] = 1.0 if pozo_viable else 0.0
            obs[188] = 1.0 if debo_arriesgar else 0.0
            obs[189] = 1.0 if puedo_alimentar else 0.0

        # --- Features all_void v6 [190:194] ---
        if self.dim >= 194:
            for palo in range(4):
                todos_vacios = all(
                    palo in vacios[j] for j in range(4) if j != a
                )
                obs[190 + palo] = 1.0 if todos_vacios else 0.0

        return obs

    def construir_desde_motor(
        self, motor, jugador_idx: int
    ) -> np.ndarray:
        """Construye una observación mínima desde la perspectiva de un jugador.

        Solo incluye información disponible desde el motor (mano, mesa, bazas).
        Las features estratégicas y all_void quedan en 0. Útil para bots y
        snapshots que juegan como oponentes en self-play.

        Args:
            motor: Instancia de MotorCorazones.
            jugador_idx: Índice del jugador.

        Returns:
            Array np.float32 de shape (dim,).
        """
        obs = np.zeros(self.dim, dtype=np.float32)
        a = jugador_idx

        for c in motor.jugadores[a].mano:
            obs[c.id] = 1.0
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0
        for i in range(4):
            for c in motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        return obs


# Instancia por defecto para compatibilidad
_observacion_v6 = ObservacionBuilder(dim=194)

__all__ = ["ObservacionBuilder"]
