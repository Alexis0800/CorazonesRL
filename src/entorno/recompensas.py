"""
Configuración de recompensas para el entorno Corazones.

Extraído de entorno.py para cumplir Single Responsibility Principle (SRP).
Centraliza todas las constantes de recompensa y la lógica de cálculo
asociada a eventos de baza y fin de mano.

v12 (consolidación — SSOT):
  - Valores unificados con los que se usaban en CorazonesEnv (v12).
  - Magnitudes reducidas respecto a v9 para evitar reward hacking.
  - BAZA_TARDIA = 7 (phase-gating).
  - Nuevos métodos tácticos: liderar_q_equivocado, quemar_maxima_palo_seguro,
    quemar_maxima_forzada, dump_q_siguiendo_picas, ganar_baza_con_puntos_evitable,
    descartar_k_a_picas, quemar_alta_siguiendo_palo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RewardConfig:
    """Configuración inmutable de todas las constantes de recompensa (v12 SSOT).

    Estos valores son el Single Source of Truth. CorazonesEnv DEBE usar
    CalculadoraRecompensas para toda lógica de recompensa.
    """

    # --- Recompensas por evento (siempre activas) ---
    REWARD_CORAZON: float = -1.0
    REWARD_DAMA_PICAS: float = -6.0            # v12: reducido -10.0→-6.0
    REWARD_SHOOTING_MOON: float = 50.0
    REWARD_CORAZON_POZO: float = 1.5           # positivo cuando pozo_viable=True

    # --- Recompensa final ---
    REWARD_PRIMERO: float = 500.0
    REWARD_SEGUNDO: float = 200.0
    REWARD_TERCERO: float = -200.0
    REWARD_CUARTO: float = -500.0

    # --- Recompensas densas (solo en bazas ≥ BAZA_TARDIA para evitar/ganar) ---
    REWARD_NO_GANAR_BAZA_CON_PUNTOS: float = 0.3   # v12: 1.5→0.3
    REWARD_DESCARTAR_CORAZON_SEGURO: float = 0.2   # v12: 0.3→0.2
    REWARD_DESCARTAR_DAMA_SEGURO: float = 2.0      # v12: 5.0→2.0
    REWARD_GANAR_BAZA_SIN_PUNTOS: float = -0.1     # v12: -0.5→-0.1

    # --- Recompensas de fin de mano ---
    REWARD_PERDER_MANO: float = -2.0
    REWARD_GANAR_MANO: float = 2.0
    REWARD_POR_PUNTO_EN_MANO: float = -0.1         # v12: -0.2→-0.1

    # --- Correcciones estratégicas (siempre activas) ---
    REWARD_Q_SPADES_SIN_POZO: float = -5.0          # v12: -10.0→-5.0
    REWARD_GANAR_BAZA_CON_CORAZON: float = -2.0     # v12: -3.0→-2.0

    # --- Recompensas estratégicas ---
    REWARD_BLOQUEAR_POZO: float = 15.0
    REWARD_ALIMENTAR_EXITOSO: float = 10.0

    # --- Correcciones tácticas (v12: magnitudes reducidas) ---
    PENALTY_LIDERAR_PICA_CON_Q_ACTIVA: float = -2.0     # v12: -3.0→-2.0
    REWARD_QUEMAR_MAXIMA_PALO_SEGURO: float = 1.0       # v12: 2.5→1.0
    PENALTY_LIDERAR_Q_EQUIVOCADO: float = -6.0          # v12: -8.0→-6.0
    REWARD_DUMP_Q_SIGUIENDO_PICAS: float = 1.0          # v12: 5.0→1.0
    PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE: float = -2.0  # v12: -8.0→-2.0
    REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA: float = 1.0  # v12: 5.0→1.0
    REWARD_QUEMAR_MAXIMA_FORZADA: float = 0.5           # v12: 1.5→0.5
    REWARD_QUEMAR_ALTA_SIGUIENDO_PALO: float = 1.0     # v12: 2.5→1.0

    # --- Correcciones tácticas (v10 Fase B, v12: magnitudes reducidas) ---
    PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD: float = -1.0  # v12: -5.0→-1.0
    REWARD_LIDERAR_Q_DUMP_SEGURO: float = 2.0          # v12: 8.0→2.0
    REWARD_DESCARTAR_CORAZON_BAJO_ROTO: float = 0.5    # v12: 2.5→0.5

    # --- Umbrales ---
    PUNTUACION_MAXIMA: float = 100.0
    BAZA_TARDIA: int = 7            # v12: 9→7 (phase-gating más temprano)
    SCORE_RIVAL_CERCA: int = 85     # umbral para "rival cerca de 100"
    CORAZONES_ALERTA_POZO: int = 10  # corazones capturados para detectar moon

    # --- v9: señales baza-level por diagnóstico PIMC ---
    # Penalización inmediata al capturar Q♠ sin intención de luna.
    # Complementa la señal terminal para mejorar crédito-asignación.
    Q_SPADES_BAZA_PENALTY: float = -5.0
    # Recompensa por corazón capturado durante intento Moon activo
    # (moon_prob_agente >= moon_prob_threshold). Incentiva perseguir la luna.
    MOON_HEARTS_STEP_REWARD: float = 0.4
    # Bonus por descartar K♠/A♠ sin ganar la baza, con Q♠ aún activa.
    # Corrige el error más costoso del v8: retener K♠ en lugar de descartarlo.
    DESCARTAR_REY_PICAS_REWARD: float = 1.5


class CalculadoraRecompensas:
    """Calcula recompensas para eventos dentro de una partida de Corazones."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()

    def recompensa_baza_ganada(
        self,
        cartas_baza: list,
        agente_idx: int,
        ganador: int,
        pozo_viable: bool,
        numero_baza: int = 1,
        en_modo_pozo: bool = False,
    ) -> float:
        """Recompensa cuando el agente ganó la baza.

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            ganador: Índice del ganador de la baza.
            pozo_viable: Si es viable shooting the moon.
            numero_baza: Número de baza actual (1-13). Dense rewards solo ≥9.
            en_modo_pozo: Si el agente está ejecutando shooting the moon.
        """
        if ganador != agente_idx:
            return 0.0

        reward = 0.0
        gano_corazon = False
        gano_q_spades = False
        puntos_baza = 0

        for c in cartas_baza:
            if c.es_corazon:
                if pozo_viable or en_modo_pozo:
                    reward += self.cfg.REWARD_CORAZON_POZO  # positivo en pozo
                else:
                    reward += self.cfg.REWARD_CORAZON
                gano_corazon = True
            if c.es_dama_de_picas:
                reward += self.cfg.REWARD_DAMA_PICAS
                gano_q_spades = True
            puntos_baza += c.puntos

        # Penalizaciones estratégicas (siempre activas)
        if gano_q_spades and not pozo_viable:
            reward += self.cfg.REWARD_Q_SPADES_SIN_POZO
        if gano_corazon and not pozo_viable and not en_modo_pozo:
            reward += self.cfg.REWARD_GANAR_BAZA_CON_CORAZON

        # Penalización por ganar baza sin puntos (solo bazas tardías)
        if puntos_baza == 0 and numero_baza >= self.cfg.BAZA_TARDIA:
            reward += self.cfg.REWARD_GANAR_BAZA_SIN_PUNTOS

        return reward

    def recompensa_baza_evitada(
        self,
        cartas_baza: list,
        agente_idx: int,
        idx_agente_en_mesa: Optional[int],
        numero_baza: int = 1,
    ) -> float:
        """Recompensa cuando el agente NO ganó la baza (evitó puntos).

        Args:
            cartas_baza: Lista de cartas en la baza.
            agente_idx: Índice del agente.
            idx_agente_en_mesa: Posición del agente en la mesa, o None.
            numero_baza: Número de baza actual (1-13). Dense rewards solo ≥9.
        """
        reward = 0.0
        puntos_baza = sum(c.puntos for c in cartas_baza)
        es_baza_tardia = numero_baza >= self.cfg.BAZA_TARDIA

        # Recompensa por evitar puntos (solo bazas tardías)
        if puntos_baza > 0 and es_baza_tardia:
            reward += self.cfg.REWARD_NO_GANAR_BAZA_CON_PUNTOS * \
                min(puntos_baza, 3)

        # Recompensa por descartes estratégicos (siempre activa para Q♠)
        if idx_agente_en_mesa is not None:
            carta_agente = cartas_baza[idx_agente_en_mesa]
            if carta_agente.es_corazon and es_baza_tardia:
                reward += self.cfg.REWARD_DESCARTAR_CORAZON_SEGURO
            if carta_agente.es_dama_de_picas:  # descarte de Q♠: siempre recompensar
                reward += self.cfg.REWARD_DESCARTAR_DAMA_SEGURO

        return reward

    def recompensa_fin_mano(
        self,
        agente_idx: int,
        puntuaciones_mano: List[int],
        pleno_jugador: Optional[int],
    ) -> float:
        """Recompensa al final de una mano."""
        reward = 0.0

        if pleno_jugador is None:
            puntos_agente = puntuaciones_mano[agente_idx]
            puntos_otros = [puntuaciones_mano[i]
                            for i in range(4) if i != agente_idx]

            if puntos_agente < min(puntos_otros):
                reward += self.cfg.REWARD_GANAR_MANO
            elif puntos_agente > max(puntos_otros):
                reward += self.cfg.REWARD_PERDER_MANO

            reward += puntos_agente * self.cfg.REWARD_POR_PUNTO_EN_MANO
        elif pleno_jugador == agente_idx:
            reward += self.cfg.REWARD_GANAR_MANO * 2

        return reward

    def recompensa_bloquear_pozo(
        self,
        agente_idx: int,
        ganador_q_espadas: Optional[int],
        corazones_por_jugador: List[int],
        pleno_jugador: Optional[int],
    ) -> float:
        """Recompensa si el agente bloqueó un intento de shooting the moon.

        Condición: algún rival acumuló ≥CORAZONES_ALERTA_POZO corazones esta mano
        pero el pozo no fue exitoso, Y el agente fue quien capturó Q♠.

        Args:
            agente_idx: Índice del agente.
            ganador_q_espadas: Quién capturó Q♠ (o None si no fue jugada).
            corazones_por_jugador: Corazones capturados por cada jugador esta mano.
            pleno_jugador: Quién hizo shooting the moon (None si nadie).
        """
        if pleno_jugador is not None:
            return 0.0  # pozo exitoso, no hubo bloqueo

        # ¿Hubo algún rival que intentó el moon (≥ umbral corazones)?
        hay_intento = any(
            j != agente_idx and corazones_por_jugador[j] >= self.cfg.CORAZONES_ALERTA_POZO
            for j in range(4)
        )
        if not hay_intento:
            return 0.0

        # El agente bloqueó si capturó Q♠ (impidió que el shooter acumulara todos los pts)
        if ganador_q_espadas == agente_idx:
            return self.cfg.REWARD_BLOQUEAR_POZO

        return 0.0

    def recompensa_alimentar(
        self,
        agente_idx: int,
        puntuaciones_mano: List[int],
        puntuaciones_historicas: List[int],
    ) -> float:
        """Recompensa si el agente "alimentó" al rival objetivo (cerca de 100).

        Condición: existe un rival con puntuacion_historica ≥ SCORE_RIVAL_CERCA
        Y ese rival recibió puntos esta mano.

        Args:
            agente_idx: Índice del agente.
            puntuaciones_mano: Puntos recibidos esta mano por cada jugador.
            puntuaciones_historicas: Puntuación acumulada ANTES de esta mano.
        """
        for j in range(4):
            if j == agente_idx:
                continue
            if (puntuaciones_historicas[j] >= self.cfg.SCORE_RIVAL_CERCA
                    and puntuaciones_mano[j] > 0):
                return self.cfg.REWARD_ALIMENTAR_EXITOSO

        return 0.0

    def recompensa_liderar_pica(
        self,
        agente_idx: int,
        carta_jugada,  # Carta
        mano_agente: list,  # List[Carta] después de jugar
        posicion_en_baza: int,  # 0=first, ..., 3=last (ANTES de jugar)
        dama_picas_activa: bool,
    ) -> float:
        """Penaliza liderar pica no-máxima con Q♠ aún activa.

        Solo aplica si el agente lideró (posición 0) una pica que no es
        su máxima pica, y Q♠ sigue en juego. Si el agente es último en
        jugar (posición 3), liderar cualquier cosa es seguro.
        """
        if posicion_en_baza != 0:
            return 0.0
        if not dama_picas_activa:
            return 0.0
        if carta_jugada.palo != 2:  # no es pica
            return 0.0
        if carta_jugada.es_dama_de_picas:  # es Q♠, se maneja aparte
            return 0.0

        # ¿Era la máxima pica del agente?
        picas_en_mano = [c for c in mano_agente if c.palo == 2]
        if not picas_en_mano:
            return 0.0  # ya no tiene picas, era la última
        if carta_jugada.valor >= max(c.valor for c in picas_en_mano):
            return 0.0  # era su máxima pica → liderar es correcto

        return self.cfg.PENALTY_LIDERAR_PICA_CON_Q_ACTIVA

    def recompensa_ganar_baza_tardia(
        self,
        numero_baza: int,
        puntos_baza: int,
        palo_salida: int,
        carta_jugada,  # Carta
        es_maxima_en_mano: bool,
    ) -> float:
        """Penaliza ganar una baza sin puntos en fase tardía con carta alta de palo seguro.

        En bazas ≥9, ganar una baza te fuerza a liderar la siguiente. Si ganas
        con la máxima de un palo seguro (♣/♦) sin puntos en juego, pierdes la
        oportunidad de ceder el lead y quedas expuesto a recibir puntos después.

        Returns:
            PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD si aplica, 0.0 si no.
        """
        if numero_baza < 9:
            return 0.0
        if puntos_baza != 0:
            return 0.0  # ya hay otras penalizaciones por ganar con puntos
        if palo_salida not in (0, 1):  # solo ♣/♦ son palos seguros
            return 0.0
        if not es_maxima_en_mano:
            return 0.0
        if carta_jugada.valor < 13:  # debe ser A o K
            return 0.0
        return self.cfg.PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD

    def recompensa_liderar_q_dump(
        self,
        numero_baza: int,
        carta_jugada,  # Carta
        es_maxima_en_picas: bool,
        hay_altas_en_circulacion: bool,
    ) -> float:
        """Recompensa liderar Q♠ como dump estratégico en bazas tardías.

        Condiciones:
        - Baza ≥7 (fase media-tardía)
        - La carta es Q♠
        - NO soy máxima en picas (si lo soy, liderar Q♠ = auto-13pts)
        - K♠/A♠ aún en circulación (alguien las tiene y cubrirá Q♠)

        Returns:
            REWARD_LIDERAR_Q_DUMP_SEGURO si aplica, 0.0 si no.
        """
        if numero_baza < 7:
            return 0.0
        if not carta_jugada.es_dama_de_picas:
            return 0.0
        if es_maxima_en_picas:
            return 0.0  # auto-13pts, no es seguro
        if not hay_altas_en_circulacion:
            return 0.0  # sin cobertura, Q♠ queda expuesta
        return self.cfg.REWARD_LIDERAR_Q_DUMP_SEGURO

    def recompensa_descartar_corazon_bajo(
        self,
        corazones_rotos: bool,
        carta_descartada,  # Carta
        es_descarte: bool,
        puntos_baza: int,
    ) -> float:
        """Recompensa descartar un corazón en baza limpia cuando los corazones están rotos.

        Con corazones rotos, cualquier corazón es un liability: un rival puede
        liderar corazones y forzarte a ganar una baza con puntos. Descartarlos
        en bazas limpias (sin puntos) elimina ese riesgo.

        Returns:
            REWARD_DESCARTAR_CORAZON_BAJO_ROTO si aplica, 0.0 si no.
        """
        if not corazones_rotos:
            return 0.0
        if not es_descarte:
            # solo aplica cuando es descarte (void en palo de salida)
            return 0.0
        if puntos_baza != 0:
            return 0.0  # no descartar corazones en bazas con puntos
        if not carta_descartada.es_corazon:
            return 0.0
        return self.cfg.REWARD_DESCARTAR_CORAZON_BAJO_ROTO

    def recompensa_liderar_q_equivocado(
        self,
        numero_baza: int,
        carta_jugada,  # Carta
        es_maxima_picas: bool,
    ) -> float:
        """Penaliza liderar Q♠ en mal momento.

        Condiciones:
        - Baza temprana (< BAZA_TARDIA): liderar Q♠ es peligroso, nadie ha
          descartado picas aún.
        - Soy máxima en picas: si lidero Q♠ siendo máxima, la recuperaré
          yo mismo → auto-13 puntos garantizados.
        """
        if not carta_jugada.es_dama_de_picas:
            return 0.0
        baza_temprana = numero_baza < self.cfg.BAZA_TARDIA
        if baza_temprana or es_maxima_picas:
            return self.cfg.PENALTY_LIDERAR_Q_EQUIVOCADO
        return 0.0

    def recompensa_quemar_maxima_palo_seguro(
        self,
        posicion_en_baza: int,
        carta_jugada,  # Carta
        es_maxima: bool,
    ) -> float:
        """Recompensa liderar la máxima de un palo seguro (♣/♦).

        Solo aplica si el agente es el líder de la baza y la carta es
        la máxima que tiene en ese palo.
        """
        if posicion_en_baza != 0:
            return 0.0
        if carta_jugada.palo not in (0, 1):  # solo ♣ y ♦
            return 0.0
        if not es_maxima:
            return 0.0
        return self.cfg.REWARD_QUEMAR_MAXIMA_PALO_SEGURO

    def recompensa_quemar_maxima_forzada(
        self,
        posicion_en_baza: int,
        forzado_a_ganar: bool,
        carta_jugada,  # Carta
    ) -> float:
        """Recompensa jugar la máxima cuando se está forzado a ganar.

        Si el agente va a ganar sí o sí (todas sus cartas del palo superan
        la máxima en mesa), jugar la más alta es correcto.
        """
        if posicion_en_baza == 0:
            return 0.0  # solo cuando sigue el palo, no cuando lidera
        if not forzado_a_ganar:
            return 0.0
        return self.cfg.REWARD_QUEMAR_MAXIMA_FORZADA

    def recompensa_dump_q_siguiendo_picas(
        self,
        carta_jugada,  # Carta
        gano_baza: bool,
        palo_salida: Optional[int],
        numero_baza: int,
    ) -> float:
        """Recompensa soltar Q♠ siguiendo el palo de picas.

        Si otro jugador lideró picas y el agente juega Q♠ (obligado
        por seguir el palo), es un dump forzado que evita recibir
        los 13 puntos más tarde.
        """
        if numero_baza < self.cfg.BAZA_TARDIA:
            return 0.0
        if not carta_jugada.es_dama_de_picas:
            return 0.0
        if gano_baza:
            return 0.0  # si ganó la baza, no fue un dump
        if palo_salida != 2:  # no es pica
            return 0.0
        return self.cfg.REWARD_DUMP_Q_SIGUIENDO_PICAS

    def recompensa_ganar_baza_con_puntos_evitable(
        self,
        numero_baza: int,
        puntos_baza: int,
        pozo_viable: bool,
    ) -> float:
        """Penaliza ganar baza con puntos cuando era evitable.

        Condiciones:
        - Baza ≥ BAZA_TARDIA
        - La baza tiene puntos
        - No es modo pozo
        """
        if numero_baza < self.cfg.BAZA_TARDIA:
            return 0.0
        if puntos_baza <= 0:
            return 0.0
        if pozo_viable:
            return 0.0
        return self.cfg.PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE

    def recompensa_descartar_k_a_picas(
        self,
        carta_jugada,  # Carta
        puntos_baza: int,
        dama_picas_activa: bool,
        es_descarte: bool,
    ) -> float:
        """Recompensa descartar K♠/A♠ en baza limpia con Q♠ aún activa.

        Descartar cartas altas de picas cuando Q♠ sigue en juego reduce
        el riesgo de verse forzado a capturarla después.
        """
        if not dama_picas_activa:
            return 0.0
        if puntos_baza != 0:
            return 0.0
        if not es_descarte:
            return 0.0
        if carta_jugada.palo != 2:
            return 0.0
        if carta_jugada.valor < 13:  # K=13, A=14
            return 0.0
        return self.cfg.REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA

    def recompensa_quemar_alta_siguiendo_palo(
        self,
        puntos_baza: int,
        carta_jugada,  # Carta
        es_mismo_palo: bool,
    ) -> float:
        """Recompensa jugar carta alta (A/K) siguiendo el palo en baza limpia.

        Quemar cartas altas en bazas sin puntos libera al agente de
        tener que ganar bazas con puntos más tarde.
        """
        if puntos_baza != 0:
            return 0.0
        if not es_mismo_palo:
            return 0.0
        if carta_jugada.valor < 13:
            return 0.0
        return self.cfg.REWARD_QUEMAR_ALTA_SIGUIENDO_PALO

    def recompensa_fin_partida(
        self, agente_idx: int, puntuacion_historica: List[int]
    ) -> float:
        """Recompensa al final de la partida según posición."""
        ranking = sorted(range(4), key=lambda i: puntuacion_historica[i])
        posicion = ranking.index(agente_idx)

        if posicion == 0:
            return self.cfg.REWARD_PRIMERO
        elif posicion == 1:
            return self.cfg.REWARD_SEGUNDO
        elif posicion == 2:
            return self.cfg.REWARD_TERCERO
        else:
            return self.cfg.REWARD_CUARTO


# Instancia por defecto para compatibilidad
_recompensas = CalculadoraRecompensas()

__all__ = ["RewardConfig", "CalculadoraRecompensas"]
