"""
Constructor de observación enriquecida para v3 (260 dimensiones).

Extiende el ObservacionBuilder base (220-d) con 40 features explícitas
de tracking que el MLP no puede inferir por sí mismo:

    [220:224] cartas_restantes — Cartas sin jugar por palo (valor raw, 0-13)
    [224:228] peligro_qs — Probabilidad Q♠ × peligrosidad del palo
    [228:232] cartas_altas_mano — J/Q/K/A en mi mano por palo
    [232:236] riesgo_baza — Puntos esperados si gano esta baza
    [236:240] control_palo — Dominancia por palo (altas en mano / altas restantes)
    [240:244] oportunidad_descarte — ¿Puedo vaciarme de este palo esta baza?
    [244]     bazas_restantes — 13 - baza_actual
    [245:249] puntos_rivales — Puntos acumulados esta mano por cada jugador (rel)
    [249]     peligro_qs_inminente — 1.0 si tengo Q♠ + otra ♠ (riesgo de jugarla)
    [250]     soy_lider — 1.0 si la mesa está vacía (agente lidera)
    [251]     lidero_picas_forzado — 1.0 si debo liderar y solo tengo ♠
    [252:256] maxima_absoluta_palo — 1.0 si tengo la carta más alta viva del palo
    [256:260] puedo_quemar_palo — 1.0 si liderar mi máxima da baza limpia (0 pts)

Estas features resuelven parcialmente el problema de "risk_assessment"
que el diagnóstico PIMC identificó como la causa raíz #1 (46% de errores).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.entorno.observacion import ObservacionBuilder
from src.entorno.dimensiones import DIM_V10

# --- Dimensionalidad v3 ---
DIM_V3: int = 265  # 220 base + 30 enriquecidas + 10 liderazgo + 4 prob_qs + 1 forzado


class ObservacionBuilderV3(ObservacionBuilder):
    """Constructor de observación enriquecida para v3.

    Extiende ObservacionBuilder (220-d) añadiendo 40 dimensiones con
    features de tracking explícito. Total: 260 dimensiones.

    Las features base [0:220] se construyen exactamente igual que en v1/v2,
    usando la herencia de ObservacionBuilder. Las features [220:260] son
    nuevas y específicas de v3.
    """

    def __init__(self, dim: int = DIM_V3):
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
        """Construye el vector de observación completo de 250 dimensiones.

        Args:
            motor: Instancia de MotorCorazones.
            agente_idx: Índice del agente (0-3).
            vacios: Lista de sets de palos void para cada jugador.
            puntuacion_historica: Puntuación acumulada de cada jugador.
            puntos_mano_actual: Puntos en la mano actual de cada jugador.
            dama_picas_en: Índice del jugador que tiene Q♠ (o None).
            pozo_viable: Si es viable intentar shooting the moon.
            debo_arriesgar: Si el agente debe arriesgarse.
            puedo_alimentar: Si el agente puede alimentar puntos a un rival.

        Returns:
            Array np.float32 de shape (250,).
        """
        # Construir base 220-d usando el método de la clase padre
        obs = super().construir(
            motor=motor,
            agente_idx=agente_idx,
            vacios=vacios,
            puntuacion_historica=puntuacion_historica,
            puntos_mano_actual=puntos_mano_actual,
            dama_picas_en=dama_picas_en,
            pozo_viable=pozo_viable,
            debo_arriesgar=debo_arriesgar,
            puedo_alimentar=puedo_alimentar,
        )

        # Añadir features enriquecidas [220:250]
        self._construir_bloque_v3(
            obs, agente_idx, motor, vacios,
            puntos_mano_actual, dama_picas_en,
        )

        return obs

    def construir_desde_motor(
        self, motor, jugador_idx: int
    ) -> np.ndarray:
        """Construye observación mínima desde la perspectiva de un jugador.

        Features enriquecidas se dejan en 0 para oponentes en self-play
        (igual que en v1/v2). Solo el agente principal recibe el vector completo.

        Args:
            motor: Instancia de MotorCorazones.
            jugador_idx: Índice del jugador.

        Returns:
            Array np.float32 de shape (250,).
        """
        obs = super().construir_desde_motor(motor, jugador_idx)
        # Rellenar con ceros las 40 dims extra
        # (ya están en cero porque np.zeros las inicializa)
        return obs

    # ------------------------------------------------------------------
    # Bloque enriquecido [220:250]
    # ------------------------------------------------------------------

    def _construir_bloque_v3(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        vacios: List[set],
        puntos_mano_actual: List[int],
        dama_picas_en: Optional[int],
    ) -> None:
        """Añade las 30 features enriquecidas de v3 al vector obs [220:250].

        Args:
            obs: Array de observación a modificar in-place (ya tiene 220 dims).
            a: Índice del agente.
            motor: Instancia de MotorCorazones.
            vacios: Lista de sets de palos void por jugador.
            puntos_mano_actual: Puntos acumulados en la mano actual.
            dama_picas_en: Índice del jugador con Q♠, o None.
        """
        # [220:224] Cartas restantes por palo (valor raw, no normalizado)
        self._fill_cartas_restantes(obs, a, motor)

        # [224:228] Peligro Q♠ por palo
        self._fill_peligro_qs(obs, a, motor, vacios, dama_picas_en)

        # [228:232] Cartas altas (J/Q/K/A) en mi mano por palo
        self._fill_cartas_altas_mano(obs, a, motor)

        # [232:236] Riesgo por baza: puntos esperados si gano
        self._fill_riesgo_baza(obs, motor)

        # [236:240] Control de palo
        self._fill_control_palo(obs, a, motor)

        # [240:244] Oportunidad de descarte (¿puedo vaciarme?)
        self._fill_oportunidad_descarte(obs, a, motor, vacios)

        # [244] Bazas restantes
        obs[244] = float(13 - motor.numero_baza)

        # [245:249] Puntos de cada jugador esta mano (relativo al agente)
        self._fill_puntos_rivales(obs, a, puntos_mano_actual)

        # [249] Peligro inminente de Q♠: 1.0 si tengo Q♠ + otra ♠
        self._fill_peligro_qs_inminente(obs, a, motor,
                                        dama_picas_en)

        # [250:260] Features de liderazgo (PIMC P2a)
        self._construir_bloque_liderazgo(
            obs, a, motor, dama_picas_en,
        )

        # [260:264] Probabilidad Q♠ por jugador (relativo al agente)
        self._fill_prob_qs_por_jugador(
            obs, a, motor, vacios, dama_picas_en,
        )

        # [264] Forzado: 1.0 si el agente no tiene alternativa (1 sola carta legal)
        self._fill_forzado(obs, a, motor)

    # ------------------------------------------------------------------
    # Sub-funciones de filling
    # ------------------------------------------------------------------

    def _fill_peligro_qs_inminente(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        dama_picas_en: Optional[int],
    ) -> None:
        """[249] Peligro inminente de Q♠: 1.0 si el agente tiene Q♠ y otra ♠.

        Indicador binario del error #1 del PIMC: jugar Q♠ voluntariamente
        cuando hay alternativas seguras en el mismo palo. Solo se activa
        cuando Q♠ está en la mano del agente Y tiene al menos otra ♠.

        0.0 en todos los demás casos (Q♠ capturada, agente no tiene Q♠,
        o Q♠ es su única ♠).
        """
        _PICA = 2

        # Q♠ ya capturada → sin peligro inminente
        if dama_picas_en is not None:
            obs[249] = 0.0
            return

        # Verificar si el agente tiene Q♠
        mi_mano = motor.jugadores[a].mano
        tiene_qs = any(c.es_dama_de_picas for c in mi_mano)

        if not tiene_qs:
            obs[249] = 0.0
            return

        # Verificar si tiene otra ♠ además de Q♠
        otras_picas = [
            c for c in mi_mano
            if c.palo == _PICA and not c.es_dama_de_picas
        ]
        obs[249] = 1.0 if len(otras_picas) >= 1 else 0.0

    def _fill_prob_qs_por_jugador(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        vacios: List[set],
        dama_picas_en: Optional[int],
    ) -> None:
        """[260:264] Probabilidad de que cada jugador (relativo al agente) tenga Q♠.

        Orden: [agente, rival_1, rival_2, rival_3].

        Lógica:
        - Q♠ capturada → todos 0.0
        - Q♠ en mesa → el jugador que la jugó = 1.0
        - Q♠ en mi mano → agente = 1.0
        - Q♠ desconocida → distribución proporcional a ♠ restantes
          en manos de cada jugador (excluyendo voids conocidos).

        Feature clave para reducir captura de Q♠ (error #1 del PIMC,
        modelos RL capturan 35-42% vs BotExperto 31%).
        """
        _PICA = 2
        OFFSET = 260

        # Q♠ ya capturada → sin incertidumbre
        if dama_picas_en is not None:
            for i in range(4):
                obs[OFFSET + i] = 0.0
            return

        # ¿Q♠ en mi mano?
        mi_mano = motor.jugadores[a].mano
        if any(c.es_dama_de_picas for c in mi_mano):
            obs[OFFSET + 0] = 1.0  # agente
            for i in range(1, 4):
                obs[OFFSET + i] = 0.0
            return

        # ¿Q♠ en la mesa actual?
        for jug_idx, c in motor.mesa:
            if c.es_dama_de_picas:
                rel = (jug_idx - a) % 4
                for i in range(4):
                    obs[OFFSET + i] = 1.0 if i == rel else 0.0
                return

        # Q♠ está en un rival desconocido → distribuir probabilidad
        # basada en cuántas ♠ puede tener cada jugador
        # Contar ♠ vistas (cementerio + mesa)
        picas_vistas = 0
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                if c.palo == _PICA:
                    picas_vistas += 1
        for _, c in motor.mesa:
            if c.palo == _PICA:
                picas_vistas += 1
        # ♠ en mi mano
        picas_en_mi_mano = sum(1 for c in mi_mano if c.palo == _PICA)
        # ♠ restantes (desconocidas)
        picas_restantes = 13 - picas_vistas - picas_en_mi_mano

        # Contar ♠ que cada rival puede tener (espacio en su mano)
        rivales = [(a + i) % 4 for i in range(1, 4)]
        capacidad = []
        for r in rivales:
            # Cartas que puede tener: tamaño de su mano
            n_cartas = len(motor.jugadores[r].mano)
            # Si es void en ♠, capacidad 0
            if r in vacios and _PICA in vacios[r]:
                capacidad.append(0)
            else:
                capacidad.append(max(0, n_cartas))

        total_cap = sum(capacidad)
        if total_cap > 0 and picas_restantes > 0:
            for i, cap in enumerate(capacidad):
                obs[OFFSET + i + 1] = cap / total_cap
        else:
            # Distribución uniforme entre los 3 rivales
            for i in range(1, 4):
                obs[OFFSET + i] = 1.0 / 3.0

        obs[OFFSET + 0] = 0.0  # agente no tiene Q♠ (ya verificado)
        _PICA = 2

        # Q♠ ya capturada → sin peligro inminente
        if dama_picas_en is not None:
            obs[249] = 0.0
            return

        # Verificar si el agente tiene Q♠
        mi_mano = motor.jugadores[a].mano
        tiene_qs = any(c.es_dama_de_picas for c in mi_mano)

        if not tiene_qs:
            obs[249] = 0.0
            return

        # Verificar si tiene otra ♠ además de Q♠
        otras_picas = [
            c for c in mi_mano
            if c.palo == _PICA and not c.es_dama_de_picas
        ]
        obs[249] = 1.0 if len(otras_picas) >= 1 else 0.0

    def _fill_cartas_restantes(
        self, obs: np.ndarray, a: int, motor
    ) -> None:
        """[220:224] Cartas restantes por palo — en manos rivales.

        Cuenta cartas que NO están en mi mano ni en el cementerio/mesa.
        Es decir, cartas en manos de los rivales.
        """
        # Cartas fuera (cementerio + mesa)
        fuera_por_palo = [0, 0, 0, 0]
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                fuera_por_palo[c.palo] += 1
        for _, c in motor.mesa:
            fuera_por_palo[c.palo] += 1

        # Cartas en mi mano
        en_mano_por_palo = [0, 0, 0, 0]
        for c in motor.jugadores[a].mano:
            en_mano_por_palo[c.palo] += 1

        for palo in range(4):
            restantes = 13 - fuera_por_palo[palo] - en_mano_por_palo[palo]
            obs[220 + palo] = float(max(0, restantes))

    def _fill_peligro_qs(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        vacios: List[set],
        dama_picas_en: Optional[int],
    ) -> None:
        """[224:228] Peligro de Q♠ por palo.

        Combina la probabilidad de que Q♠ esté en ese palo con la
        peligrosidad de jugar ese palo (número de cartas altas restantes).
        Valor = prob_qs_en_palo × (altas_restantes / 4.0).

        Si Q♠ ya fue capturada, todos los valores son 0.
        """
        _PICA = 2

        if dama_picas_en is not None:
            # Q♠ ya capturada → sin peligro
            for palo in range(4):
                obs[224 + palo] = 0.0
            return

        # ¿Tengo Q♠ yo?
        mi_mano = motor.jugadores[a].mano
        if any(c.es_dama_de_picas for c in mi_mano):
            # Q♠ en mi mano → peligro máximo en picas
            obs[224 + _PICA] = 1.0
            for palo in range(4):
                if palo != _PICA:
                    obs[224 + palo] = 0.0
            return

        # ¿Q♠ en la mesa?
        for _, c in motor.mesa:
            if c.es_dama_de_picas:
                obs[224 + _PICA] = 1.0
                for palo in range(4):
                    if palo != _PICA:
                        obs[224 + palo] = 0.0
                return

        # Q♠ está en algún rival — peligro proporcional a altas restantes
        altas_restantes = self._contar_altas_restantes(motor)
        for palo in range(4):
            peligro_base = 1.0 if palo == _PICA else 0.3
            obs[224 + palo] = peligro_base * (altas_restantes[palo] / 4.0)

    def _fill_cartas_altas_mano(
        self, obs: np.ndarray, a: int, motor
    ) -> None:
        """[228:232] Cartas altas (J=11, Q=12, K=13, A=14) en mi mano por palo.

        Valor raw: 0-4 por palo. Feature clave para risk_assessment.
        """
        altas = [0, 0, 0, 0]
        for c in motor.jugadores[a].mano:
            if c.valor >= 11:
                altas[c.palo] += 1
        for palo in range(4):
            obs[228 + palo] = float(altas[palo])

    def _fill_riesgo_baza(
        self, obs: np.ndarray, motor
    ) -> None:
        """[232:236] Riesgo de ganar esta baza por palo.

        Estima los puntos que recibiría si ganara la baza actual
        jugando una carta de cada palo. Solo considera cartas ya en la mesa.

        Feature clave para el error #1 (LIDERAR mal).
        """
        puntos_en_mesa_por_palo = [0, 0, 0, 0]
        for _, c in motor.mesa:
            puntos = 0
            if c.es_corazon:
                puntos += 1
            if c.es_dama_de_picas:
                puntos += 13
            puntos_en_mesa_por_palo[c.palo] += puntos

        for palo in range(4):
            # Normalizado a [0, 1] (max 13 pts por baza)
            obs[232 + palo] = min(puntos_en_mesa_por_palo[palo] / 13.0, 1.0)

    def _fill_control_palo(
        self, obs: np.ndarray, a: int, motor
    ) -> None:
        """[236:240] Control de palo: ¿soy dominante?

        control = altas_en_mi_mano / max(altas_restantes_total, 1).
        Rango [0, 1]. 1.0 = tengo todas las altas restantes.
        """
        altas_en_mano = [0, 0, 0, 0]
        for c in motor.jugadores[a].mano:
            if c.valor >= 11:
                altas_en_mano[c.palo] += 1

        altas_rest = self._contar_altas_restantes(motor)

        for palo in range(4):
            total_altas = altas_en_mano[palo] + altas_rest[palo]
            if total_altas > 0:
                obs[236 + palo] = altas_en_mano[palo] / total_altas
            else:
                obs[236 + palo] = 0.0

    def _fill_oportunidad_descarte(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        vacios: List[set],
    ) -> None:
        """[240:244] Oportunidad de descarte por palo.

        Vale 1.0 si tengo cartas de este palo Y el palo de salida
        es diferente (puedo descartar). 0.0 si no tengo o debo seguir.
        """
        palo_salida = motor.palo_de_salida
        mi_mano = motor.jugadores[a].mano

        tengo_por_palo = [False, False, False, False]
        for c in mi_mano:
            tengo_por_palo[c.palo] = True

        for palo in range(4):
            if not tengo_por_palo[palo]:
                obs[240 + palo] = 0.0  # No tengo cartas de este palo
            elif palo_salida is not None and palo != palo_salida:
                # Puedo descartar si no tengo que seguir el palo
                if tengo_por_palo[palo_salida] if palo_salida is not None else True:
                    obs[240 + palo] = 0.0  # Tengo que seguir palo de salida
                else:
                    # Soy void en palo de salida → puedo descartar
                    obs[240 + palo] = 1.0
            elif palo_salida is None:
                obs[240 + palo] = 0.0  # Soy mano (el que lidera)
            else:
                obs[240 + palo] = 0.0  # Mismo palo que salida

    def _fill_puntos_rivales(
        self,
        obs: np.ndarray,
        a: int,
        puntos_mano_actual: List[int],
    ) -> None:
        """[245:249] Puntos acumulados esta mano por cada jugador (relativo).

        El orden es [agente, rival_rel_1, rival_rel_2, rival_rel_3].
        Valores raw (0-26), sin normalizar.
        """
        for r in range(4):
            jug_idx = (a + r) % 4
            obs[245 + r] = float(puntos_mano_actual[jug_idx])

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _fill_forzado(
        self,
        obs: np.ndarray,
        a: int,
        motor,
    ) -> None:
        """[264] Indicador de jugada forzada: 1.0 si solo hay 1 carta legal.

        Feature clave para credit assignment: cuando el agente no tiene
        alternativa, los castigos por capturar puntos no deberian
        penalizarlo igual que cuando eligio mal teniendo opciones.

        El modelo puede aprender a ignorar/discountear recompensas
        negativas cuando esta feature esta activa.
        """
        legales = motor.obtener_jugadas_legales(a)
        obs[264] = 1.0 if len(legales) <= 1 else 0.0

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _contar_altas_restantes(motor) -> List[int]:
        """Cuenta J/Q/K/A que quedan sin jugar por palo.

        Args:
            motor: Instancia de MotorCorazones.

        Returns:
            Lista de 4 enteros con el conteo de altas restantes.
        """
        # Total de altas por palo: 4 (J, Q, K, A)
        altas_totales = [4, 4, 4, 4]

        # Restar altas ya jugadas (cementerio + mesa)
        for j in range(4):
            for c in motor.jugadores[j].bazas_ganadas:
                if c.valor >= 11:
                    altas_totales[c.palo] -= 1
        for _, c in motor.mesa:
            if c.valor >= 11:
                altas_totales[c.palo] -= 1

        return [max(0, x) for x in altas_totales]

    # ------------------------------------------------------------------
    # Bloque liderazgo [250:260]
    # ------------------------------------------------------------------

    def _construir_bloque_liderazgo(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        dama_picas_en,
    ) -> None:
        """Añade 10 features de liderazgo [250:260].

        Args:
            obs: Array de observación a modificar in-place.
            a: Índice del agente.
            motor: Instancia de MotorCorazones.
            dama_picas_en: Índice del jugador con Q♠, o None si activa.
        """
        # [250] soy_lider
        self._fill_soy_lider(obs, motor)

        # [251] lidero_picas_forzado
        self._fill_lidero_picas_forzado(obs, a, motor)

        # [252:256] maxima_absoluta_palo
        self._fill_maxima_absoluta_palo(obs, a, motor)

        # [256:260] puedo_quemar_palo
        self._fill_puedo_quemar_palo(obs, a, motor, dama_picas_en)

    # ── [250] soy_lider ──

    def _fill_soy_lider(
        self, obs: np.ndarray, motor
    ) -> None:
        """[250] 1.0 si la mesa está vacía → el agente lidera esta baza."""
        obs[250] = 1.0 if not motor.mesa else 0.0

    # ── [251] lidero_picas_forzado ──

    def _fill_lidero_picas_forzado(
        self, obs: np.ndarray, a: int, motor
    ) -> None:
        """[251] 1.0 si el agente debe liderar y solo tiene ♠ en mano."""
        _PICA = 2
        if motor.mesa:
            obs[251] = 0.0
            return

        mi_mano = motor.jugadores[a].mano
        palos_en_mano = {c.palo for c in mi_mano}
        obs[251] = 1.0 if palos_en_mano == {_PICA} else 0.0

    # ── [252:256] maxima_absoluta_palo ──

    def _fill_maxima_absoluta_palo(
        self, obs: np.ndarray, a: int, motor
    ) -> None:
        """[252:256] 1.0 si el agente tiene la carta más alta viva del palo.

        Determina, para cada palo, si la carta más alta que queda sin jugar
        está en la mano del agente. Solo se consideran cartas en manos
        (ni bazas_ganadas ni mesa — ésas ya no están vivas).
        """
        # Encontrar la máxima viva por palo (solo en manos)
        max_viva = [-1, -1, -1, -1]  # valor máximo por palo
        poseedor_max = [-1, -1, -1, -1]  # quién la tiene

        for j in range(4):
            for c in motor.jugadores[j].mano:
                if c.valor > max_viva[c.palo]:
                    max_viva[c.palo] = c.valor
                    poseedor_max[c.palo] = j

        for palo in range(4):
            obs[252 + palo] = 1.0 if poseedor_max[palo] == a else 0.0

    # ── [256:260] puedo_quemar_palo ──

    def _fill_puedo_quemar_palo(
        self,
        obs: np.ndarray,
        a: int,
        motor,
        dama_picas_en,
    ) -> None:
        """[256:260] 1.0 si liderar mi máxima del palo da baza limpia (0 pts).

        Condiciones:
          - Tengo la máxima absoluta del palo (entre cartas vivas en manos).
          - El palo no tiene puntos en juego (♥=1pt, Q♠=13pt).
          - Para ♠: Q♠ debe estar capturada (no activa).
        """
        _PICA = 2

        # Determinar máxima por palo (solo cartas en manos)
        max_viva = [-1, -1, -1, -1]
        poseedor_max = [-1, -1, -1, -1]
        for j in range(4):
            for c in motor.jugadores[j].mano:
                if c.valor > max_viva[c.palo]:
                    max_viva[c.palo] = c.valor
                    poseedor_max[c.palo] = j

        qs_capturada = dama_picas_en is not None

        for palo in range(4):
            # ¿Tengo la máxima?
            tengo_max = poseedor_max[palo] == a

            # ¿El palo es seguro? (sin puntos)
            if palo == _PICA:
                palo_seguro = qs_capturada  # ♠ solo seguro si Q♠ capturada
            else:
                palo_seguro = True  # ♣/♦ no tienen puntos

            obs[256 + palo] = 1.0 if (tengo_max and palo_seguro) else 0.0


__all__ = ["ObservacionBuilderV3", "DIM_V3"]
