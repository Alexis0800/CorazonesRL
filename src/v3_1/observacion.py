"""
Constructor de observación v3.1 — 228 dimensiones depuradas.

Extiende ObservacionBuilder (220-d) eliminando duplicados,
features inferibles, y añadiendo patrones de rivales.

Bloques (todos relativos al agente, orden: [agente, rival+1, rival+2, rival+3]):
  [0:52]     Mano del agente (one-hot)
  [52:104]   Mesa actual (one-hot)
  [104:156]  Cementerio (one-hot)
  [156:172]  Vacíos (4 jug × 4 palos)
  [172:176]  Puntaje histórico /100
  [176:180]  Puntos mano actual (raw 0-26)
  [180]      Corazones rotos
  [181]      Posición en baza (0, .33, .66, 1)
  [182:187]  Q♠ tracker (5 estados: desconocida, jug0, jug1, jug2, jug3)
  [187]      pozo_viable
  [188]      Número de baza /13
  [189:193]  Cartas restantes por palo (raw, 0-13)
  [193]      Palo de salida (0 si None, else palo/3)
  [194:198]  Peligro Q♠ por palo
  [198]      Peligro Q♠ inminente (1.0 si tengo Q♠ + otra ♠)
  [199:203]  Prob Q♠ por jugador (usa voids)
  [203:207]  Altas en mi mano (J/Q/K/A) por palo
  [207:211]  Máxima absoluta por palo (1.0 si tengo la más alta viva)
  [211:215]  Control de palo: mis altas / total altas vivas
  [215:219]  ♥ altos (J/Q/K/A) capturados por rival
  [219:223]  ♠ altas (J/Q/K/A) jugadas por rival
  [223:227]  ¿Ya jugó ♥ cada rival?
  [227]      Forzado (1 sola carta legal)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.entorno.observacion import ObservacionBuilder
from src.v3_1.dimensiones import DIM_V3_1


class ObservacionBuilderV31(ObservacionBuilder):
    """Constructor de observación depurada para v3.1.

    Extiende ObservacionBuilder base (220-d) eliminando:
    - Duplicados: cartas restantes /13, prob Q♠ v1, puntos /26,
      bazas restantes, Q♠ capturada, all_void_X, riesgo baza
    - Inferibles: cerca de 100, soy líder, altas restantes /4,
      corazones capturados /13, alerta pozo, posición en baza,
      oportunidad descarte, liderazgo duplicado
    - Bajo valor: mano terminal, debo_arriesgar, puedo_alimentar,
      lidero picas forzado, puedo quemar palo

    Añade 12 dimensiones de patrones de rivales [215:227]:
    - ♥ altos capturados por cada rival (detección de pozo)
    - ♠ altas jugadas por cada rival (detección de evasión Q♠)
    - ¿Ya jugó ♥ cada rival? (quién rompió corazones)
    """

    def __init__(self, dim: int = DIM_V3_1):
        super().__init__(dim=dim)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

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
        """Construye el vector de observación completo de 228 dimensiones.

        Args:
            motor: Instancia de MotorCorazones.
            agente_idx: Índice del agente (0-3).
            vacios: Lista de sets de palos void para cada jugador.
            puntuacion_historica: Puntuación acumulada de cada jugador.
            puntos_mano_actual: Puntos en la mano actual de cada jugador.
            dama_picas_en: Índice del jugador que tiene Q♠ (o None).
            pozo_viable: Si es viable intentar shooting the moon.
            debo_arriesgar: Ignorado en v3.1 (no usado).
            puedo_alimentar: Ignorado en v3.1 (no usado).

        Returns:
            Array np.float32 de shape (228,).
        """
        obs = np.zeros(self.dim, dtype=np.float32)
        a = agente_idx

        # ── Bloque 1: Cartas [0:156] ──
        self._fill_cartas(obs, a, motor)

        # ── Bloque 2: Vacíos [156:172] ──
        self._fill_vacios(obs, a, vacios)

        # ── Bloque 3: Puntuaciones [172:181] ──
        self._fill_puntuaciones(
            obs, a, motor, puntuacion_historica, puntos_mano_actual,
        )

        # ── Bloque 4: Q♠ + posición [181:188] ──
        self._fill_qs_posicion(obs, a, motor, dama_picas_en, pozo_viable)

        # ── Bloque 5: Estado de la mano [188:194] ──
        self._fill_estado_mano(obs, a, motor)

        # ── Bloque 6: Q♠ enriquecido [194:203] ──
        self._fill_qs_enriquecido(obs, a, motor, vacios, dama_picas_en)

        # ── Bloque 7: Control de palo [203:215] ──
        self._fill_control_palo(obs, a, motor)

        # ── Bloque 8: Patrones de rivales [215:227] ──
        self._fill_patrones_rivales(obs, a, motor)

        # ── Bloque 9: Forzado [227] ──
        self._fill_forzado(obs, a, motor)

        return obs

    def construir_desde_motor(
        self, motor, jugador_idx: int
    ) -> np.ndarray:
        """Construye observación mínima desde la perspectiva de un jugador.

        Solo llena los bloques de cartas y vacíos básicos.
        El resto queda en cero (usado para snapshots en self-play).

        Args:
            motor: Instancia de MotorCorazones.
            jugador_idx: Índice del jugador.

        Returns:
            Array np.float32 de shape (228,).
        """
        obs = np.zeros(self.dim, dtype=np.float32)
        self._fill_cartas(obs, jugador_idx, motor)
        return obs

    # ------------------------------------------------------------------
    # Bloque 1: Cartas [0:156]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_cartas(obs: np.ndarray, a: int, motor) -> None:
        """Llena los 3 bloques one-hot de cartas.

        [0:52]    Mano del agente
        [52:104]  Mesa actual
        [104:156] Cementerio (bazas ganadas de todos los jugadores)
        """
        # Mano [0:52]
        for c in motor.jugadores[a].mano:
            obs[c.id] = 1.0

        # Mesa [52:104]
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0

        # Cementerio [104:156]
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                obs[104 + c.id] = 1.0

    # ------------------------------------------------------------------
    # Bloque 2: Vacíos [156:172]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_vacios(obs: np.ndarray, a: int, vacios: List[set]) -> None:
        """[156:172] Vacíos conocidos: 4 jugadores × 4 palos.

        Orden: [agente_p0, agente_p1, agente_p2, agente_p3,
                rival1_p0, rival1_p1, rival1_p2, rival1_p3,
                rival2_p0, ..., rival3_p3]
        """
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            for palo in vacios[jug_idx]:
                obs[156 + rel * 4 + palo] = 1.0

    # ------------------------------------------------------------------
    # Bloque 3: Puntuaciones [172:181]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_puntuaciones(
        obs: np.ndarray,
        a: int,
        motor,
        puntuacion_historica: List[int],
        puntos_mano_actual: List[int],
    ) -> None:
        """[172:181] Puntajes históricos, puntos mano actual, corazones rotos.

        [172:176] Puntaje histórico /100 (4 jugadores, relativo al agente)
        [176:180] Puntos mano actual raw 0-26 (4 jugadores, relativo)
        [180]     Corazones rotos (0.0 o 1.0)
        """
        # Histórico [172:176]
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[172 + rel] = min(puntuacion_historica[jug_idx] / 100.0, 1.0)

        # Mano actual [176:180] — valores raw, sin normalizar
        for jug_idx in range(4):
            rel = (jug_idx - a) % 4
            obs[176 + rel] = float(puntos_mano_actual[jug_idx])

        # Corazones rotos [180]
        obs[180] = 1.0 if motor.corazones_rotos else 0.0

    # ------------------------------------------------------------------
    # Bloque 4: Q♠ + Posición [181:188]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_qs_posicion(
        obs: np.ndarray,
        a: int,
        motor,
        dama_picas_en: Optional[int],
        pozo_viable: bool,
    ) -> None:
        """[181:188] Posición en baza, Q♠ tracker, pozo_viable.

        [181]     Posición en baza: 0.0, 0.33, 0.66, 1.0
        [182:187] Q♠ tracker: one-hot 5 estados
                  [0]=desconocida, [1]=agente, [2]=rival1, [3]=rival2, [4]=rival3
        [187]     pozo_viable: 1.0 si es viable intentar moon
        """
        # Posición [181]
        posiciones = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
        obs[181] = posiciones.get(len(motor.mesa), 0.0)

        # Q♠ tracker [182:187]
        if dama_picas_en is None:
            obs[182] = 1.0  # desconocida
        else:
            rel = (dama_picas_en - a) % 4
            obs[183 + rel] = 1.0  # jugador específico (relativo)

        # Pozo viable [187]
        obs[187] = 1.0 if pozo_viable else 0.0

    # ------------------------------------------------------------------
    # Bloque 5: Estado de la mano [188:194]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_estado_mano(
        obs: np.ndarray, a: int, motor
    ) -> None:
        """[188:194] Número de baza, cartas restantes, palo de salida.

        [188]     Número de baza /13
        [189:193] Cartas restantes por palo (raw 0-13)
                  = cartas que NO están en mi mano ni en el cementerio/mesa
        [193]     Palo de salida: 0.0 si None, else palo/3.0
        """
        # Baza [188]
        obs[188] = min(motor.numero_baza / 13.0, 1.0)

        # Cartas restantes [189:193]
        # Contar cartas fuera (cementerio + mesa)
        fuera_por_palo = [0, 0, 0, 0]
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                fuera_por_palo[c.palo] += 1
        for _, c in motor.mesa:
            fuera_por_palo[c.palo] += 1

        # Contar cartas en mi mano
        en_mano = [0, 0, 0, 0]
        for c in motor.jugadores[a].mano:
            en_mano[c.palo] += 1

        for palo in range(4):
            restantes = 13 - fuera_por_palo[palo] - en_mano[palo]
            obs[189 + palo] = float(max(0, restantes))

        # Palo de salida [193]
        obs[193] = (
            0.0 if motor.palo_de_salida is None
            else motor.palo_de_salida / 3.0
        )

    # ------------------------------------------------------------------
    # Bloque 6: Q♠ enriquecido [194:203]
    # ------------------------------------------------------------------

    def _fill_qs_enriquecido(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        vacios: List[set],
        dama_picas_en: Optional[int],
    ) -> None:
        """[194:203] Peligro Q♠, peligro inminente, probabilidad Q♠.

        [194:198] Peligro Q♠ por palo
        [198]     Peligro Q♠ inminente (tengo Q♠ + otra ♠)
        [199:203] Probabilidad Q♠ por jugador (usa voids)
        """
        self._fill_peligro_qs(obs, a, motor, dama_picas_en)
        self._fill_peligro_qs_inminente(obs, a, motor, dama_picas_en)
        self._fill_prob_qs(obs, a, motor, vacios, dama_picas_en)

    @staticmethod
    def _fill_peligro_qs(
        obs: np.ndarray, a: int, motor, dama_picas_en: Optional[int]
    ) -> None:
        """[194:198] Peligro Q♠ por palo.

        Combina probabilidad de Q♠ en el palo × peligrosidad.
        Si Q♠ capturada → todo 0.
        """
        _PICA = 2

        if dama_picas_en is not None:
            for palo in range(4):
                obs[194 + palo] = 0.0
            return

        # Q♠ en mi mano → peligro máximo en picas
        if any(c.es_dama_de_picas for c in motor.jugadores[a].mano):
            obs[194 + _PICA] = 1.0
            return

        # Q♠ en mesa → peligro en picas
        for _, c in motor.mesa:
            if c.es_dama_de_picas:
                obs[194 + _PICA] = 1.0
                return

        # Q♠ en rivales → peligro proporcional
        altas_rest = ObservacionBuilderV31._contar_altas_restantes(motor)
        for palo in range(4):
            peligro_base = 1.0 if palo == _PICA else 0.3
            obs[194 + palo] = peligro_base * (altas_rest[palo] / 4.0)

    @staticmethod
    def _fill_peligro_qs_inminente(
        obs: np.ndarray, a: int, motor, dama_picas_en: Optional[int]
    ) -> None:
        """[198] 1.0 si el agente tiene Q♠ Y otra ♠ en mano."""
        if dama_picas_en is not None:
            obs[198] = 0.0
            return

        mi_mano = motor.jugadores[a].mano
        tiene_qs = any(c.es_dama_de_picas for c in mi_mano)
        if not tiene_qs:
            obs[198] = 0.0
            return

        otras_picas = [
            c for c in mi_mano
            if c.palo == 2 and not c.es_dama_de_picas
        ]
        obs[198] = 1.0 if len(otras_picas) >= 1 else 0.0

    @staticmethod
    def _fill_prob_qs(
        obs: np.ndarray,
        a: int,
        motor,
        vacios: List[set],
        dama_picas_en: Optional[int],
    ) -> None:
        """[199:203] Probabilidad Q♠ por jugador (relativo al agente).

        Usa voids y cartas restantes de ♠ para distribuir probabilidad.

        Orden: [agente, rival+1, rival+2, rival+3].
        """
        _PICA = 2

        # Q♠ ya capturada → todo 0
        if dama_picas_en is not None:
            return  # ya están en 0

        # Agente tiene Q♠ → prob 1.0 para agente
        mi_mano = motor.jugadores[a].mano
        if any(c.es_dama_de_picas for c in mi_mano):
            obs[199] = 1.0
            return

        # Q♠ en mesa → el jugador que la jugó
        for jug_idx, c in motor.mesa:
            if c.es_dama_de_picas:
                rel = (jug_idx - a) % 4
                obs[199 + rel] = 1.0
                return

        # Q♠ desconocida → distribuir proporcional a ♠ restantes por jugador
        # Primero, contar ♠ restantes totales
        fuera_picas = 0
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                if c.palo == _PICA:
                    fuera_picas += 1
        for _, c in motor.mesa:
            if c.palo == _PICA:
                fuera_picas += 1

        picas_en_mi_mano = sum(1 for c in mi_mano if c.palo == _PICA)
        picas_restantes = 13 - fuera_picas - picas_en_mi_mano

        if picas_restantes <= 0:
            return

        # Capacidad de cada rival (n_cartas en mano, 0 si void en ♠)
        capacidad = []
        for r in range(1, 4):
            rival_idx = (a + r) % 4
            n_cartas = len(motor.jugadores[rival_idx].mano)
            if n_cartas == 0:
                capacidad.append(0)
            elif _PICA in vacios[rival_idx]:
                capacidad.append(0)
            else:
                capacidad.append(max(0, n_cartas))

        total_cap = sum(capacidad)
        if total_cap > 0:
            for i, cap in enumerate(capacidad):
                obs[200 + i] = cap / total_cap
        else:
            # Distribución uniforme
            for i in range(1, 4):
                obs[199 + i] = 1.0 / 3.0

    # ------------------------------------------------------------------
    # Bloque 7: Control de palo [203:215]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_control_palo(
        obs: np.ndarray, a: int, motor
    ) -> None:
        """[203:215] Altas en mano, máxima absoluta, dominancia.

        [203:207] Altas en mi mano (J/Q/K/A) por palo — raw count 0-4
        [207:211] Máxima absoluta por palo — 1.0 si tengo la más alta viva
        [211:215] Control de palo — mis altas / total altas vivas
        """
        # Altas en mano [203:207]
        altas_mano = [0, 0, 0, 0]
        for c in motor.jugadores[a].mano:
            if c.valor >= 11:  # J=11, Q=12, K=13, A=14
                altas_mano[c.palo] += 1
        for palo in range(4):
            obs[203 + palo] = float(altas_mano[palo])

        # Altas restantes (en manos rivales, no jugadas aún)
        altas_rest = ObservacionBuilderV31._contar_altas_restantes(motor)

        # Restar las altas que tiene el agente en su mano
        # (_contar_altas_restantes cuenta todas las altas no jugadas,
        #  incluyendo las del agente — hay que quitarlas)
        for palo in range(4):
            altas_rest[palo] = max(0, altas_rest[palo] - altas_mano[palo])

        # Máxima absoluta [207:211]
        for palo in range(4):
            if altas_mano[palo] > 0 and altas_rest[palo] == 0:
                # Tengo al menos un alta y no quedan altas en rivales
                # → tengo LA máxima (asumiendo que la más alta de mi mano es A)
                obs[207 + palo] = 1.0
            else:
                obs[207 + palo] = 0.0

        # Control de palo [211:215]
        for palo in range(4):
            total_altas = altas_mano[palo] + altas_rest[palo]
            if total_altas > 0:
                obs[211 + palo] = altas_mano[palo] / total_altas
            else:
                obs[211 + palo] = 0.0

    # ------------------------------------------------------------------
    # Bloque 8: Patrones de rivales [215:227]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_patrones_rivales(
        obs: np.ndarray, a: int, motor
    ) -> None:
        """[215:227] Patrones de comportamiento de rivales.

        [215:219] ♥ altos (J/Q/K/A) capturados por rival esta mano
                  Orden: [rival+1, rival+2, rival+3, agente]
                  → ¿Alguien está juntando corazones? (posible pozo)

        [219:223] ♠ altas (J/Q/K/A) jugadas por rival
                  Orden: [rival+1, rival+2, rival+3, agente]
                  → ¿Alguien está botando ♠? (no quiere Q♠)

        [223:227] ¿Ya jugó ♥ cada rival?
                  Orden: [rival+1, rival+2, rival+3, agente]
                  → ¿Quién rompió corazones?
        """
        # Inicializar contadores
        corazones_altos_capt = [0, 0, 0, 0]  # por jugador absoluto
        picas_altas_jugadas = [0, 0, 0, 0]   # por jugador absoluto
        ya_jugo_corazon = [0, 0, 0, 0]       # por jugador absoluto

        # Recorrer bazas ganadas de cada jugador
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                # ♥ altos capturados
                if c.es_corazon and c.valor >= 11:
                    corazones_altos_capt[j] += 1
                # ♠ altas
                if c.palo == 2 and c.valor >= 11:
                    picas_altas_jugadas[j] += 1
                # ¿Ya jugó ♥?
                if c.es_corazon:
                    ya_jugo_corazon[j] = 1

        # Mesa: cartas en juego actual
        for jug_idx, c in motor.mesa:
            if c.es_corazon and c.valor >= 11:
                corazones_altos_capt[jug_idx] += 1
            if c.palo == 2 and c.valor >= 11:
                picas_altas_jugadas[jug_idx] += 1
            if c.es_corazon:
                ya_jugo_corazon[jug_idx] = 1

        # Rellenar en orden relativo: [rival+1, rival+2, rival+3, agente]
        for r in range(4):
            jug_idx = (a + r + 1) % 4  # empezar por rival+1

            # ♥ altos [215:219] — normalizado /4
            obs[215 + r] = min(corazones_altos_capt[jug_idx] / 4.0, 1.0)

            # ♠ altas [219:223] — normalizado /4
            obs[219 + r] = min(picas_altas_jugadas[jug_idx] / 4.0, 1.0)

            # ¿Ya jugó ♥? [223:227]
            obs[223 + r] = float(ya_jugo_corazon[jug_idx])

    # ------------------------------------------------------------------
    # Bloque 9: Forzado [227]
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_forzado(obs: np.ndarray, a: int, motor) -> None:
        """[227] 1.0 si solo hay 1 carta legal (jugada forzada).

        Feature clave para credit assignment: cuando el agente no tiene
        alternativa, los castigos por capturar puntos no deberían
        penalizarlo igual que cuando eligió mal teniendo opciones.
        """
        legales = motor.obtener_jugadas_legales(a)
        obs[227] = 1.0 if len(legales) <= 1 else 0.0

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _contar_altas_restantes(motor) -> List[int]:
        """Cuenta J/Q/K/A que quedan sin jugar por palo (en manos rivales).

        Args:
            motor: Instancia de MotorCorazones.

        Returns:
            Lista de 4 enteros con el conteo de altas restantes.
        """
        altas_totales = [4, 4, 4, 4]  # J, Q, K, A por palo

        # Restar altas ya jugadas (cementerio + mesa)
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                if c.valor >= 11:
                    altas_totales[c.palo] -= 1
        for _, c in motor.mesa:
            if c.valor >= 11:
                altas_totales[c.palo] -= 1

        # Restar altas en mi mano
        # (asumimos agente=0 para esta utilidad, el caller ajusta)
        # NOTA: Esta utilidad cuenta altas en TODAS las manos rivales.
        # El caller debe restar las altas en mano del agente si es necesario.
        return [max(0, a) for a in altas_totales]
