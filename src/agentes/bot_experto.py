"""
Bot Experto para el juego de Corazones.

Razonamiento con estado completo por mano:
  - Cementerio: todas las cartas vistas (bazas + mesa)
  - Voids acumulados entre bazas (infiere de cada baza completa)
  - Posición de Q♠ por eliminación (sabe quién NO puede tenerla)
  - Detecta intento de pozo ajeno y lo bloquea
  - Quema palos en etapa final cuando tiene la máxima
  - Descarte con malicia: da Q♠ al objetivo más rentable

Modos: MINIMIZAR | POZO | BLOQUEAR_POZO | ALIMENTAR
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones

_TREBOL, _DIAMANTE, _PICA, _CORAZON = 0, 1, 2, 3

# Umbrales para detección de pozo ajeno
_CORAZONES_PARA_SOSPECHAR = 6   # nro de corazones capturados por un rival
_CORAZONES_PARA_ALERTAR = 9     # nro sin Q♠ que ya es alarma roja


class BotExperto:
    """Bot heurístico con razonamiento completo e inferencia de estado global."""

    def __init__(self) -> None:
        self._reset_estado_mano()

    # ──────────────────────────────────────────────────────────────
    # Estado intramano
    # ──────────────────────────────────────────────────────────────

    def _reset_estado_mano(self) -> None:
        """Reinicia todo el estado al inicio de una nueva mano."""
        # Voids persistentes entre bazas: jugador → set de palos donde es void
        self._vacios: Dict[int, Set[int]] = {i: set() for i in range(4)}
        # Palo de salida de la BAZA ANTERIOR (para reconstruir voids)
        self._palo_salida_anterior: Optional[int] = None
        # Quién inició la BAZA ANTERIOR (para reconstruir orden de juego)
        self._starter_anterior: int = 0
        # Palo de salida de la baza actual (lo guardamos al decidir o al ver la mesa)
        self._palo_salida_baza_actual: Optional[int] = None
        # Número de baza en la última llamada
        self._ultimo_baza_num: int = 0
        # Jugador sospechoso de intentar shooting the moon
        self._sospecha_pozo: Optional[int] = None

    def __call__(self, motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta:
        self._actualizar_estado(motor, idx)
        carta = self._decidir(motor, idx, legales)
        # Registrar el palo que lideramos (cuando somos los primeros en jugar)
        if not motor.mesa:
            self._palo_salida_baza_actual = carta.palo
        return carta

    # ──────────────────────────────────────────────────────────────
    # Actualización de estado
    # ──────────────────────────────────────────────────────────────

    def _actualizar_estado(self, motor: MotorCorazones, idx: int) -> None:
        # Nueva mano: resetear todo
        if motor.numero_baza == 1 and self._ultimo_baza_num > 1:
            self._reset_estado_mano()

        # Nueva baza: procesar la baza anterior que ya fue resuelta
        if motor.numero_baza > self._ultimo_baza_num:
            self._procesar_baza_anterior(motor)
            # Guardar datos de la baza que ACABA de empezar
            self._starter_anterior = motor.indice_jugador_inicial
            self._palo_salida_anterior = self._palo_salida_baza_actual
            self._palo_salida_baza_actual = motor.palo_de_salida  # puede ser None
            self._ultimo_baza_num = motor.numero_baza

        # Actualizar palo de la baza actual si ya fue establecido por el primer jugador
        if motor.palo_de_salida is not None:
            self._palo_salida_baza_actual = motor.palo_de_salida

        # Inferir voids desde los jugadores que ya jugaron en ESTA baza
        palo = motor.palo_de_salida
        if palo is not None and motor.mesa:
            # el primero en la mesa es el líder
            starter_baza = motor.mesa[0][0]
            for jug_idx, carta in motor.mesa:
                if jug_idx != starter_baza and carta.palo != palo:
                    self._vacios[jug_idx].add(palo)

        # Actualizar sospecha de pozo ajeno
        self._actualizar_sospecha_pozo(motor, idx)

    def _procesar_baza_anterior(self, motor: MotorCorazones) -> None:
        """
        Reconstruye la baza anterior (ya resuelta) para inferir voids de los 4 jugadores.

        Después de resolver_baza(), el ganador tiene las 4 cartas en bazas_ganadas[-4:],
        en el orden en que fueron jugadas (starter primero). Usamos _starter_anterior
        para saber qué jugador jugó qué carta.
        """
        if self._ultimo_baza_num == 0:
            return  # sin baza anterior
        palo = self._palo_salida_anterior
        if palo is None:
            # no conocemos el palo de salida (no debería ocurrir normalmente)
            return

        ganador = motor.indice_jugador_inicial  # ganador de la baza anterior
        bazas = motor.jugadores[ganador].bazas_ganadas
        if len(bazas) < 4:
            return

        cartas_baza = bazas[-4:]
        for i, carta in enumerate(cartas_baza):
            jugador_i = (self._starter_anterior + i) % 4
            # El líder (i=0) siempre juega el palo de salida, no puede ser void
            if i > 0 and carta.palo != palo:
                self._vacios[jugador_i].add(palo)

    def _actualizar_sospecha_pozo(self, motor: MotorCorazones, mi_idx: int) -> None:
        """Detecta si algún rival está acumulando suficientes puntos para el pozo."""
        for i, j in enumerate(motor.jugadores):
            if i == mi_idx:
                continue
            corazones = sum(1 for c in j.bazas_ganadas if c.es_corazon)
            tiene_q = any(c.es_dama_de_picas for c in j.bazas_ganadas)
            # Alerta: tiene Q♠ + muchos corazones → podría completar el pozo
            if corazones >= _CORAZONES_PARA_SOSPECHAR and tiene_q:
                self._sospecha_pozo = i
                return
            # Alerta sin Q♠: acumula muchos corazones, Q♠ puede venir
            if corazones >= _CORAZONES_PARA_ALERTAR:
                self._sospecha_pozo = i
                return
        self._sospecha_pozo = None

    # ──────────────────────────────────────────────────────────────
    # Consultas de estado del juego
    # ──────────────────────────────────────────────────────────────

    def _cementerio(self, motor: MotorCorazones) -> Set[Carta]:
        """Todas las cartas ya jugadas (bazas ganadas + en mesa actual)."""
        jugado: Set[Carta] = set()
        for j in motor.jugadores:
            jugado.update(j.bazas_ganadas)
        jugado.update(c for _, c in motor.mesa)
        return jugado

    def _cartas_restantes_por_palo(
        self, motor: MotorCorazones, mi_mano: List[Carta]
    ) -> Dict[int, int]:
        """Cartas por palo que no están en mi mano ni en el cementerio."""
        jugadas = self._cementerio(motor)
        mi_set = set(mi_mano)
        restantes = {0: 0, 1: 0, 2: 0, 3: 0}
        for c in Carta._TODAS:
            if c not in jugadas and c not in mi_set:
                restantes[c.palo] += 1
        return restantes

    def _cartas_altas_restantes(
        self, motor: MotorCorazones, mi_mano: List[Carta]
    ) -> Dict[int, int]:
        """Cartas J/Q/K/A por palo que no están en mi mano ni jugadas."""
        jugadas = self._cementerio(motor)
        mi_set = set(mi_mano)
        altas = {0: 0, 1: 0, 2: 0, 3: 0}
        for c in Carta._TODAS:
            if c.valor >= 11 and c not in jugadas and c not in mi_set:
                altas[c.palo] += 1
        return altas

    def _q_activa(self, motor: MotorCorazones) -> bool:
        """True si Q♠ no ha sido ganada en ninguna baza todavía."""
        for j in motor.jugadores:
            if any(c.es_dama_de_picas for c in j.bazas_ganadas):
                return False
        return True

    def _q_posibles(self, motor: MotorCorazones, idx: int) -> Set[int]:
        """
        Jugadores que PUEDEN tener la Q♠ en mano (por eliminación).

        Si un jugador es void en picas, definitivamente no la tiene.
        """
        if not self._q_activa(motor):
            return set()
        # Si Q♠ está en la mesa actual, no está en ninguna mano ahora mismo
        if any(c.es_dama_de_picas for _, c in motor.mesa):
            return set()

        mi_mano = motor.jugadores[idx].mano
        if any(c.es_dama_de_picas for c in mi_mano):
            return {idx}

        posibles: Set[int] = set()
        for i in range(4):
            if i != idx and _PICA not in self._vacios[i]:
                posibles.add(i)
        return posibles

    def _tengo_maxima_del_palo(self, motor: MotorCorazones, idx: int, palo: int) -> bool:
        """
        True si mi carta más alta de este palo ganaría la baza si la lidero.
        Equivale a: ningún rival tiene carta más alta de este palo.
        """
        mi_mano = motor.jugadores[idx].mano
        mis_del_palo = [c for c in mi_mano if c.palo == palo]
        if not mis_del_palo:
            return False

        jugadas = self._cementerio(motor)
        mi_set = set(mi_mano)
        restantes_rivales = [
            c for c in Carta._TODAS
            if c.palo == palo and c not in jugadas and c not in mi_set
        ]
        if not restantes_rivales:
            return True  # soy el único con cartas de este palo

        mi_max = max(mis_del_palo, key=lambda c: c.valor)
        max_rival = max(restantes_rivales, key=lambda c: c.valor)
        return mi_max.valor > max_rival.valor

    def _objetivo_q(self, motor: MotorCorazones, idx: int) -> Optional[int]:
        """
        Jugador óptimo para recibir Q♠ cuando decidimos descargarla.
        Preferimos al rival con MENOS puntos acumulados (el líder), que tenga
        la mayor capacidad de absorber daño sin cerrar el juego inmediatamente.
        """
        puntos = [motor.jugadores[i].puntuacion_historica for i in range(4)]
        candidatos = [i for i in range(4) if i != idx]
        # Si alguien está en modo pozo sospechoso, no darle la Q♠ (se la facilitamos)
        if self._sospecha_pozo is not None:
            candidatos = [i for i in candidatos if i != self._sospecha_pozo]
        if not candidatos:
            candidatos = [i for i in range(4) if i != idx]
        # Dar la Q♠ al que tiene MENOS puntos (mayor impacto relativo)
        return min(candidatos, key=lambda i: puntos[i])

    # ──────────────────────────────────────────────────────────────
    # Modos de juego
    # ──────────────────────────────────────────────────────────────

    def _pozo_viable(self, motor: MotorCorazones, idx: int) -> bool:
        """True si tengo condiciones mínimas para shooting the moon."""
        if motor.corazones_rotos:
            return False
        if motor.jugadores[idx].puntuacion_historica >= 85:
            return False
        mi_mano = motor.jugadores[idx].mano
        corazones = [c for c in mi_mano if c.es_corazon]
        altos = [c for c in corazones if c.valor >= 11]
        return len(corazones) >= 6 and len(altos) >= 3

    def _modo(self, motor: MotorCorazones, idx: int) -> str:
        """Determina el modo de juego: MINIMIZAR | POZO | BLOQUEAR_POZO | ALIMENTAR."""
        if self._pozo_viable(motor, idx):
            return "POZO"
        if self._sospecha_pozo is not None:
            return "BLOQUEAR_POZO"
        mi_pts = motor.jugadores[idx].puntuacion_historica
        for i, j in enumerate(motor.jugadores):
            if i != idx and j.puntuacion_historica >= 85 and mi_pts < j.puntuacion_historica:
                return "ALIMENTAR"
        return "MINIMIZAR"

    # ──────────────────────────────────────────────────────────────
    # Toma de decisiones
    # ──────────────────────────────────────────────────────────────

    def _decidir(self, motor: MotorCorazones, idx: int, legales: List[Carta]) -> Carta:
        modo = self._modo(motor, idx)

        if not motor.mesa:
            return self._liderar(motor, idx, legales, modo)

        palo = motor.palo_de_salida
        mismo_palo = [c for c in legales if c.palo == palo]

        if mismo_palo:
            return self._seguir_palo(motor, idx, mismo_palo, modo)
        return self._descartar(motor, idx, legales, modo)

    def _liderar(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
        modo: str,
    ) -> Carta:
        baza = motor.numero_baza

        # Modo pozo: liderar con el corazón más alto para barrer la mesa
        if modo == "POZO":
            corazones = [c for c in legales if c.es_corazon]
            if corazones:
                return max(corazones, key=lambda c: c.valor)

        # Modo bloquear pozo: liderar con corazón bajo para ganar una baza
        # y quitarle corazones al rival sospechoso
        if modo == "BLOQUEAR_POZO" and motor.corazones_rotos:
            corazones = [c for c in legales if c.es_corazon]
            if corazones:
                return min(corazones, key=lambda c: c.valor)

        # ── Liderar Q♠ como dump cuando hay picas más altas en circulación ──────
        # En bazas tardías (≥7): si K♠/A♠ siguen en manos rivales, liderar Q♠
        # fuerza al poseedor a ganar la baza llevándose los 13 pts.
        # Solo si algún rival no es void confirmado en ♠.
        mi_q_en_mano = [c for c in legales if c.es_dama_de_picas]
        if mi_q_en_mano and baza >= 7 and not self._tengo_maxima_del_palo(motor, idx, _PICA):
            alguno_puede_seguir_picas = any(
                _PICA not in self._vacios[i] for i in range(4) if i != idx
            )
            if alguno_puede_seguir_picas:
                return mi_q_en_mano[0]

        # Nunca liderar con Q♠ si hay alternativa (fuera del caso anterior)
        sin_q = [c for c in legales if not c.es_dama_de_picas]
        candidatos = sin_q if sin_q else legales

        # ── Final de mano (baza ≥ 9): dar el lead, dump de corazones/picas ─
        # Con pocas cartas restantes, ganar una baza con A♣/A♦ fuerza a liderar
        # de nuevo, quedando atrapado con corazones/picas. Mejor ceder el lead.
        if baza >= 9:
            # Liderar corazón para dump: alguien más lo ganará (1 pt para él)
            if motor.corazones_rotos:
                corazones_legales = [c for c in candidatos if c.es_corazon]
                if corazones_legales:
                    # Liderar corazón medio: bajo enough para no ganar, alto enough para dump
                    # Preferir corazones ≤10 (no J/Q/K/A que son más controlables)
                    medios = [c for c in corazones_legales if c.valor <= 10]
                    if medios:
                        return max(medios, key=lambda c: c.valor)
                    return min(corazones_legales, key=lambda c: c.valor)

            # Liderar pica (no Q♠) para dump si Q♠ ya fue capturada
            if not self._q_activa(motor):
                picas_legales = [c for c in candidatos if c.palo == _PICA]
                if picas_legales:
                    return max(picas_legales, key=lambda c: c.valor)

            # En bazas 11+: evitar ganar la baza a toda costa
            if baza >= 11:
                sin_puntos_tardios = [c for c in candidatos if c.puntos == 0]
                if sin_puntos_tardios:
                    return min(sin_puntos_tardios, key=lambda c: c.valor)
                return min(candidatos, key=lambda c: c.puntos * 100 + c.valor)

        # ── Quemar palos (desde baza 6) ──────────────────────────────────────
        # Si tengo la máxima de un palo seguro (♣/♦), liderarla garantiza ganar
        # una baza limpia ahora, mejor que ser forzado a ganar una con puntos después.
        if baza >= 6:
            for palo in (_TREBOL, _DIAMANTE):
                mis_del_palo = [c for c in candidatos if c.palo == palo]
                if mis_del_palo and self._tengo_maxima_del_palo(motor, idx, palo):
                    return max(mis_del_palo, key=lambda c: c.valor)

            # Quemar ♠ también, pero solo si Q♠ ya fue capturada
            if not self._q_activa(motor):
                mis_picas = [c for c in candidatos if c.palo == _PICA]
                if mis_picas and self._tengo_maxima_del_palo(motor, idx, _PICA):
                    return max(mis_picas, key=lambda c: c.valor)

        # ── Preferir cartas sin puntos, la más baja posible ────────────────
        sin_puntos = [c for c in candidatos if c.puntos == 0]
        if sin_puntos:
            return min(sin_puntos, key=lambda c: c.valor)
        return min(candidatos, key=lambda c: c.puntos * 100 + c.valor)

    def _seguir_palo(
        self,
        motor: MotorCorazones,
        idx: int,
        mismo_palo: List[Carta],
        modo: str,
    ) -> Carta:
        q_en_mesa = any(c.es_dama_de_picas for _, c in motor.mesa)
        puntos_mesa = sum(c.puntos for _, c in motor.mesa)
        ganadora = self._carta_ganadora_actual(motor)

        # Bloquear pozo: ganar la baza aunque tenga corazones, para cortarle el pozo
        if modo == "BLOQUEAR_POZO" and puntos_mesa > 0:
            gana = [c for c in mismo_palo
                    if ganadora is None or c.valor > ganadora.valor]
            if gana:
                return min(gana, key=lambda c: c.valor)  # ganar con el mínimo

        # ── Dump Q♠ siguiendo picas (baza ≥5): si todas mis otras picas
        # ganan la baza, jugar Q♠ da la oportunidad de que K♠/A♠ la cubran.
        # Solo cuando no soy la máxima y no tengo picas perdedoras.
        if (motor.palo_de_salida == _PICA and not q_en_mesa and modo != "POZO"
                and motor.numero_baza >= 5):
            mi_q = [c for c in mismo_palo if c.es_dama_de_picas]
            if mi_q and not self._tengo_maxima_del_palo(motor, idx, _PICA):
                otras = [c for c in mismo_palo if not c.es_dama_de_picas]
                if otras and ganadora is not None:
                    # Solo si TODAS mis otras picas ganan la baza
                    if all(c.valor > ganadora.valor for c in otras):
                        return mi_q[0]

        # Evitar capturar Q♠: jugar la carta más alta que no gane la baza
        if q_en_mesa and ganadora is not None:
            no_gana = [c for c in mismo_palo if c.valor <= ganadora.valor]
            if no_gana:
                return max(no_gana, key=lambda c: c.valor)
            # Todas ganan (todas > Q♠): jugar la más baja para minimizar daño futuro
            return min(mismo_palo, key=lambda c: c.valor)

        # Evitar ganar baza con corazones: la carta más alta que no gana
        if puntos_mesa > 0 and ganadora is not None:
            no_gana = [c for c in mismo_palo if c.valor <= ganadora.valor]
            if no_gana:
                return max(no_gana, key=lambda c: c.valor)
            return min(mismo_palo, key=lambda c: c.valor)

        # Sin puntos en mesa: quemar la carta alta de forma segura.
        # Si no puedo ganar la baza, el ganador ya está determinado → jugar la
        # más alta que pierda (elimina liability futura sin riesgo).
        # Si todas mis cartas ganan, ganar con la MÁXIMA: quemar la carta alta
        # ahora (0 pts) es mejor que arriesgarse a ganar con ella después (con puntos).
        if ganadora is not None:
            no_gana = [c for c in mismo_palo if c.valor <= ganadora.valor]
            if no_gana:
                # ── Quemar A♠/K♠ con Q♠ activa aunque pueda perder ──────
                # Si Q♠ sigue en circulación, A♠/K♠ son liability: en una
                # futura baza de ♠ forzarán ganar y Q♠ puede caer encima.
                # Quemarlas ahora en baza limpia elimina ese riesgo.
                if self._q_activa(motor):
                    picas_altas_quemables = [
                        c for c in mismo_palo
                        if c.palo == _PICA and c.valor >= 13 and not c.es_dama_de_picas
                    ]
                    if picas_altas_quemables:
                        return picas_altas_quemables[0]  # A♠ o K♠
                return max(no_gana, key=lambda c: c.valor)
            # Todas mis cartas ganan la baza → quemar la más alta
            return max(mismo_palo, key=lambda c: c.valor)
        return min(mismo_palo, key=lambda c: c.valor)

    def _descartar(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
        modo: str,
    ) -> Carta:
        """Descartar cuando soy void en el palo liderado."""
        puntos_mesa = sum(c.puntos for _, c in motor.mesa)
        ganador_baza = self._ganador_actual(motor)

        q = [c for c in legales if c.es_dama_de_picas]
        corazones_altos = sorted(
            [c for c in legales if c.es_corazon and c.valor >= 11],
            key=lambda c: -c.valor,
        )
        corazones_todos = sorted(
            [c for c in legales if c.es_corazon],
            key=lambda c: -c.valor,
        )
        sin_puntos = [c for c in legales if c.puntos ==
                      0 and not c.es_dama_de_picas]

        # ── Modo bloquear pozo ────────────────────────────────────────────
        if modo == "BLOQUEAR_POZO" and ganador_baza == self._sospecha_pozo:
            # No dar más puntos al sospechoso; descartar algo neutro
            if sin_puntos:
                return max(sin_puntos, key=lambda c: c.valor)
            if corazones_todos:
                return min(corazones_todos, key=lambda c: c.valor)
            return legales[0]

        # ── Baza con puntos: descartar con malicia ─────────────────────────
        if puntos_mesa > 0:
            if q:
                # ¿Quién va ganando la baza? ¿Es nuestro objetivo?
                objetivo = self._objetivo_q(motor, idx)
                if ganador_baza == objetivo or ganador_baza is None:
                    return q[0]  # dar Q♠ al objetivo o a quien sea
                # El ganador actual no es nuestro objetivo óptimo;
                # aún así descargar Q♠ si no hay mejor momento
                # (preferable a guardarla para una baza peor)
                return q[0]

            # Sin Q♠: dar el corazón más alto (más puntos para el ganador)
            if corazones_todos:
                return corazones_todos[0]
            return max(sin_puntos or legales, key=lambda c: c.valor)

        # ── Baza sin puntos: deshacerse de cartas peligrosas ─────────────
        baza = motor.numero_baza

        # Desde la mitad de la mano (baza ≥6), Q♠ vale 13 pts y cada baza adicional
        # reduce las oportunidades de descargarla estratégicamente.
        if q and baza >= 6 and modo != "POZO":
            return q[0]

        # Corazones J/Q/K/A♥ son peligrosos (pueden ganar bazas futuras con puntos)
        if corazones_altos:
            return corazones_altos[0]

        # K♠/A♠ con Q♠ activa: si alguien lidera ♠, podemos ganar la baza y
        # recibir Q♠ descartada encima (13 pts). Mejor descargarlos en baza limpia.
        if self._q_activa(motor):
            picas_altas = [c for c in legales
                           if c.palo == _PICA and c.valor > 12 and not c.es_dama_de_picas]
            if picas_altas:
                return max(picas_altas, key=lambda c: c.valor)

        # Ases de palos seguros (A♣/A♦): siempre ganan su palo, lo que atrae
        # descartados de corazones de rivales void. Mejor liberarlos ahora.
        ases_seguros = [c for c in sin_puntos if c.valor == 14]
        if ases_seguros:
            return ases_seguros[0]

        # Con corazones rotos, los corazones bajos también son peligrosos:
        # los rivales pueden liderarlos y forzarnos a ganar bazas con puntos después.
        if corazones_todos and motor.corazones_rotos:
            return corazones_todos[0]

        # Sin corazones altos ni urgencia: carta sin puntos más alta (liberar mano)
        if sin_puntos:
            return max(sin_puntos, key=lambda c: c.valor)

        # Solo corazones bajos (sin puntos en mesa y sin rotos) o Q♠ sin umbral
        if corazones_todos:
            return corazones_todos[0]
        return q[0] if q else min(legales, key=lambda c: c.puntos)

    # ──────────────────────────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────────────────────────

    def _carta_ganadora_actual(self, motor: MotorCorazones) -> Optional[Carta]:
        """Retorna la carta que actualmente gana la baza en curso."""
        if not motor.mesa:
            return None
        palo = motor.palo_de_salida
        ganadora: Optional[Carta] = None
        for _, c in motor.mesa:
            if c.palo == palo:
                if ganadora is None or c.valor > ganadora.valor:
                    ganadora = c
        return ganadora

    def _ganador_actual(self, motor: MotorCorazones) -> Optional[int]:
        """Retorna el índice del jugador que actualmente gana la baza."""
        ganadora = self._carta_ganadora_actual(motor)
        if ganadora is None:
            return None
        for jug_idx, carta in motor.mesa:
            if carta == ganadora:
                return jug_idx
        return None
