"""
Constructores del vector de observación para el entorno Corazones.

Extraído de entorno.py para cumplir Single Responsibility Principle (SRP).
La lógica de construcción de observación está separada del ciclo de vida
del entorno Gymnasium.

Vector de observación (v11, 224 dimensiones):
    [0:52]    Mano del agente (one-hot)
    [52:104]  Mesa actual / baza en curso (one-hot)
    [104:156] Cementerio / cartas jugadas en bazas anteriores (one-hot)
    [156:172] Vacíos conocidos (4 jugadores × 4 palos)
    [172:176] Puntajes históricos globales (normalizados /100)
    [176:180] Puntos acumulados en la mano actual (normalizados /26)
    [180]     Corazones rotos (0.0 o 1.0)
    [181]     Posición en la baza actual (0.0, 0.33, 0.66, 1.0)
    [182:187] Rastreador de la Dama de Picas (one-hot, 5 estados)
    [187]     moon_prob_agente — P(Moon del agente) continuo [0, 1]
    [188]     moon_prob_rival  — max P(Moon) entre los 3 rivales [0, 1]
    [189]     puedo_alimentar — ¿puedo darle puntos a un rival?
    [190:194] all_void_X — ¿los 3 rivales son void en el palo X?
    [194]     Número de baza / 13.0
    [195]     Jugadores cerca de 100 pts / 3.0
    [196]     Q♠ ya fue capturada
    [197]     Agente es líder de puntaje
    [198]     Mano terminal posible (score ≥74)
    [199:203] Cartas restantes por palo / 13.0
    [203:207] Cartas altas (J/Q/K/A) restantes por palo / 4.0
    [207:211] Probabilidad Q♠ por jugador relativo
    [211:215] Corazones capturados esta mano / 13.0
    [215:219] Alerta pozo por jugador (≥6 corazones)
    [219]     Palo de salida (-1.0 si None, else palo/3.0)
    [220:224] quien_jugo_mesa — 4 flags: ¿el jugador relativo ya jugó en esta baza?
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.entorno.dimensiones import DIM_V5, DIM_V6, DIM_V10, DIM_V11, DIM_ENTRENAMIENTO


class ObservacionBuilder:
    """Construye vectores de observación desde el estado del motor y entorno.

    Centraliza TODA la lógica de construcción de observación en un solo lugar,
    eliminando la duplicación que existía en entorno.py, PoliticaSB3, y
    entorno_multi.py.

    Soporta 3 dimensiones (definidas en dimensiones.py):
        DIM_V5 = 190 (features básicas + flags estratégicos)
        DIM_V6 = 194 (+ all_void por palo)
        DIM_V10 = 220 (+ bloque v9 de features avanzadas)
    """

    DIM_V5: int = DIM_V5
    DIM_V6: int = DIM_V6
    DIM_V10: int = DIM_V10

    def __init__(self, dim: int = DIM_ENTRENAMIENTO):
        self.dim = dim

    def construir(
        self,
        motor,
        agente_idx: int,
        vacios: List[set],
        puntuacion_historica: List[int],
        puntos_mano_actual: List[int],
        dama_picas_en: Optional[int],
        moon_prob_agente: float = 0.0,
        moon_prob_rival: float = 0.0,
        puedo_alimentar: bool = False,
    ) -> np.ndarray:
        """Construye el vector de observación completo de `dim` dimensiones.

        Args:
            motor: Instancia de MotorCorazones.
            agente_idx: Índice del agente (0-3).
            vacios: Lista de sets de palos void para cada jugador.
            puntuacion_historica: Puntuación acumulada de cada jugador.
            puntos_mano_actual: Puntos en la mano actual de cada jugador.
            dama_picas_en: Índice del jugador con la Dama de Picas (o None).
            moon_prob_agente: P(Moon del agente) continuo [0, 1].
            moon_prob_rival: max P(Moon) entre los 3 rivales [0, 1].
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
            obs[187] = float(moon_prob_agente)   # P(Moon agente) continuo [0,1]
            obs[188] = float(moon_prob_rival)    # max P(Moon rival) continuo [0,1]
            obs[189] = 1.0 if puedo_alimentar else 0.0

        # --- Features all_void v6 [190:194] ---
        if self.dim >= 194:
            for palo in range(4):
                todos_vacios = all(
                    palo in vacios[j] for j in range(4) if j != a
                )
                obs[190 + palo] = 1.0 if todos_vacios else 0.0

        # --- Bloque v9 [194:220] ---
        if self.dim >= 220:
            self._construir_bloque_v9(
                obs, a, motor, puntuacion_historica, dama_picas_en, vacios
            )

        # --- Bloque v11 [220:224]: quien_jugo_mesa ---
        if self.dim >= 224:
            self._construir_bloque_v11(obs, a, motor)

        return obs

    def _construir_bloque_v9(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        puntuacion_historica: List[int],
        dama_picas_en: Optional[int],
        vacios: List[set],
    ) -> None:
        """Añade los 26 features avanzadas (v9) al vector obs [194:220].

        Args:
            obs: Array de observación a modificar in-place.
            a: Índice del agente.
            motor: Instancia de MotorCorazones.
            puntuacion_historica: Puntuación acumulada de cada jugador.
            dama_picas_en: Índice del jugador con Q♠, o None.
        """
        # [194] baza_numero / 13.0
        obs[194] = min(motor.numero_baza / 13.0, 1.0)

        # [195] jugadores_cerca_de_100 / 3.0
        cerca = sum(1 for p in puntuacion_historica if p >= 85)
        obs[195] = cerca / 3.0

        # [196] Q♠ ya fue capturada
        obs[196] = 1.0 if dama_picas_en is not None else 0.0

        # [197] soy líder en puntaje
        mi_pts = puntuacion_historica[a]
        obs[197] = 1.0 if all(
            mi_pts <= puntuacion_historica[j] for j in range(4)
        ) else 0.0

        # [198] mano terminal posible (algún jugador ≥74)
        obs[198] = 1.0 if any(p >= 74 for p in puntuacion_historica) else 0.0

        # [199:203] cartas restantes por palo / 13.0
        cementerio_por_palo = [0, 0, 0, 0]
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                cementerio_por_palo[c.palo] += 1
        for palo in range(4):
            obs[199 + palo] = max(0.0,
                                  (13.0 - cementerio_por_palo[palo]) / 13.0)

        # [203:207] cartas altas (J/Q/K/A) restantes por palo / 4.0
        altas_cementerio = [0, 0, 0, 0]
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                if c.valor >= 11:
                    altas_cementerio[c.palo] += 1
        for palo in range(4):
            obs[203 + palo] = max(0.0, (4.0 - altas_cementerio[palo]) / 4.0)

        # [207:211] probabilidad Q♠ por jugador relativo
        q_prob = self._calcular_prob_q_picas(a, motor, vacios, dama_picas_en)
        for r in range(4):
            obs[207 + r] = q_prob[r]

        # [211:215] corazones capturados esta mano / 13.0
        for j in range(4):
            rel = (j - a) % 4
            corazones = sum(
                1 for c in motor.jugadores[j].bazas_ganadas if c.es_corazon
            )
            obs[211 + rel] = min(corazones / 13.0, 1.0)

        # [215:219] alerta pozo: ≥6 corazones esta mano
        for j in range(4):
            rel = (j - a) % 4
            corazones = sum(
                1 for c in motor.jugadores[j].bazas_ganadas if c.es_corazon
            )
            obs[215 + rel] = 1.0 if corazones >= 6 else 0.0

        # [219] palo_salida: 0.0 si None, else palo/3.0
        # (0.0 es válido porque palo=0 es trébol — la distinción se pierde
        #  pero palo_salida=None solo ocurre al inicio de cada mano)
        obs[219] = (
            0.0 if motor.palo_de_salida is None
            else motor.palo_de_salida / 3.0
        )

    def _construir_bloque_v11(
        self,
        obs: np.ndarray,
        a: int,
        motor,
    ) -> None:
        """Añade las 4 features v11: quien_jugo_mesa [220:224].

        Para cada jugador relativo al agente (0=self, 1=izq, 2=frente, 3=der),
        indica si ya jugó en la baza actual.

        Args:
            obs: Array de observación a modificar in-place.
            a: Índice del agente.
            motor: Instancia de MotorCorazones.
        """
        # Determinar qué jugadores ya jugaron en esta baza
        jugadores_que_jugaron = {idx for idx, _ in motor.mesa}
        for rel in range(4):
            abs_idx = (a + rel) % 4
            obs[220 + rel] = 1.0 if abs_idx in jugadores_que_jugaron else 0.0

    def _calcular_prob_q_picas(
        self,
        a: int,
        motor,
        vacios: List[set],
        dama_picas_en: Optional[int],
    ) -> List[float]:
        """Distribuye probabilidad de Q♠ entre jugadores por eliminación.

        Usa voids conocidos para excluir jugadores que son void en picas.
        """
        _PICA = 2

        if dama_picas_en is not None:
            return [0.0, 0.0, 0.0, 0.0]

        mi_mano = motor.jugadores[a].mano
        if any(c.es_dama_de_picas for c in mi_mano):
            return [1.0, 0.0, 0.0, 0.0]

        for jug_idx, carta in motor.mesa:
            if carta.es_dama_de_picas:
                rel = (jug_idx - a) % 4
                result = [0.0, 0.0, 0.0, 0.0]
                result[rel] = 1.0
                return result

        # Q♠ solo puede estar en rivales no-void en picas
        candidatos = [
            r for r in range(1, 4)
            if _PICA not in vacios[(a + r) % 4]
        ]
        if not candidatos:
            return [0.0, 0.0, 0.0, 0.0]
        prob = 1.0 / len(candidatos)
        result = [0.0, 0.0, 0.0, 0.0]
        for r in candidatos:
            result[r] = prob
        return result

    def construir_desde_motor(
        self, motor, jugador_idx: int
    ) -> np.ndarray:
        """Construye una observación mínima desde la perspectiva de un jugador.

        Solo incluye información disponible desde el motor (mano, mesa, bazas).
        Las features estratégicas y avanzadas quedan en 0. Útil para bots y
        snapshots que juegan como oponentes en self-play.

        Args:
            motor: Instancia de MotorCorazones.
            jugador_idx: Índice del jugador.

        Returns:
            Array np.float32 de shape (dim,).
        """
        obs = np.zeros(self.dim, dtype=np.float32)

        for c in motor.jugadores[jugador_idx].mano:
            obs[c.id] = 1.0
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0
        for i in range(4):
            for c in motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        # Bloque v11: quien_jugo_mesa
        if self.dim >= 224:
            jugadores_que_jugaron = {idx for idx, _ in motor.mesa}
            for rel in range(4):
                abs_idx = (jugador_idx + rel) % 4
                obs[220 + rel] = 1.0 if abs_idx in jugadores_que_jugaron else 0.0

        return obs


# Instancia por defecto
_observacion_default = ObservacionBuilder(dim=DIM_V11)

__all__ = ["ObservacionBuilder"]
