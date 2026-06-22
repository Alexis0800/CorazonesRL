"""
Tests unitarios para BotCastigador — oponente que explota debilidades con Q♠.

El BotCastigador está diseñado para presionar al agente RL durante self-play:
  - Lidera ♠ cuando Q♠ está activa para forzar al portador a jugarla.
  - Sigue ♠ agresivamente (carta más alta) para forzar Q♠ del rival.
  - En otros palos, juega conservadoramente (minimizar puntos).

Estrategia TDD (Red-Green-Refactor):
  RED   → Escribir tests primero (deben fallar).
  GREEN → Implementar el código mínimo para que pasen.
  REFACTOR → Mejorar sin romper tests.
"""

from __future__ import annotations

import pytest

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.agentes.bot_castigador import BotCastigador

# ─── Constantes ────────────────────────────────────────────────────────────
_TREBOL, _DIAMANTE, _PICA, _CORAZON = 0, 1, 2, 3


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _crear_motor_con_mano(
    cartas_j0: list[Carta],
    cartas_j1: list[Carta],
    cartas_j2: list[Carta],
    cartas_j3: list[Carta],
    *,
    baza: int = 1,
    corazones_rotos: bool = False,
    indice_inicial: int | None = None,
) -> MotorCorazones:
    """Crea un motor con manos predefinidas para testing determinístico.

    Args:
        cartas_j0..j3: Manos de cada jugador.
        baza: Número de baza (default 1). Usar ≥2 para evitar la regla 2♣.
        corazones_rotos: Si True, se permite liderar corazones.
        indice_inicial: Jugador que inicia la baza. Si None, se infiere del
                       poseedor de 2♣ (baza 1) o se asigna J0.
    """
    m = MotorCorazones()
    m.jugadores[0].mano = list(cartas_j0)
    m.jugadores[1].mano = list(cartas_j1)
    m.jugadores[2].mano = list(cartas_j2)
    m.jugadores[3].mano = list(cartas_j3)
    m.corazones_rotos = corazones_rotos
    m.numero_baza = baza
    m.mesa = []
    m.palo_de_salida = None
    m._mano_activa = True
    if indice_inicial is not None:
        m.indice_jugador_inicial = indice_inicial
    elif baza == 1:
        for i, jug in enumerate(m.jugadores):
            if any(c.es_dos_de_treboles for c in jug.mano):
                m.indice_jugador_inicial = i
                return m
        m.indice_jugador_inicial = 0
    else:
        m.indice_jugador_inicial = 0
    return m


def _carta_por_id(carta_id: int) -> Carta:
    """Helper: obtener carta por su ID."""
    return Carta._TODAS[carta_id]


def _qs() -> Carta:
    """Retorna Q♠ (id 36)."""
    return _carta_por_id(36)


def _as_picas() -> Carta:
    """Retorna A♠ (id 38)."""
    return _carta_por_id(38)


def _rey_picas() -> Carta:
    """Retorna K♠ (id 37)."""
    return _carta_por_id(37)


def _diez_picas() -> Carta:
    """Retorna 10♠ (id 34)."""
    return _carta_por_id(34)


def _dos_picas() -> Carta:
    """Retorna 2♠ (id 26)."""
    return _carta_por_id(26)


def _as_trebol() -> Carta:
    return _carta_por_id(12)


def _dos_trebol() -> Carta:
    return _carta_por_id(0)


def _as_corazon() -> Carta:
    return _carta_por_id(51)


def _dos_corazon() -> Carta:
    return _carta_por_id(39)


# ═══════════════════════════════════════════════════════════════════════════
# Clase 1 — Instanciación y firma
# ═══════════════════════════════════════════════════════════════════════════


class TestBotCastigadorInstanciacion:
    """Verifica que BotCastigador cumple con la interfaz PoliticaJuego."""

    def test_se_instancia_sin_errores(self) -> None:
        """Instanciación básica sin argumentos."""
        bot = BotCastigador()
        assert bot is not None
        assert callable(bot)

    def test_es_callable_con_tres_argumentos(self) -> None:
        """Debe aceptar (motor, idx, legales) → Carta."""
        bot = BotCastigador()
        m = MotorCorazones()
        m.repartir()
        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)
        assert isinstance(carta, Carta)
        assert carta in legales


# ═══════════════════════════════════════════════════════════════════════════
# Clase 2 — Comportamiento al liderar
# ═══════════════════════════════════════════════════════════════════════════


class TestBotCastigadorLidera:
    """Comportamiento cuando el bot es mano (primer jugador de la baza)."""

    def test_lidera_picas_cuando_qs_activa_y_no_la_tiene(self) -> None:
        """Si Q♠ está activa y el bot no la tiene, debe preferir liderar ♠."""
        bot = BotCastigador()

        # Baza 2+: sin restricción de 2♣. J0 lidera con A♠, 2♣.
        # J1 tiene Q♠.
        m = _crear_motor_con_mano(
            cartas_j0=[_as_picas(), _dos_trebol()],
            cartas_j1=[_qs(), _dos_corazon()],
            cartas_j2=[_carta_por_id(1), _carta_por_id(2)],
            cartas_j3=[_carta_por_id(3), _carta_por_id(4)],
            baza=2,
            corazones_rotos=True,
        )

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # Debe liderar ♠ (A♠ en este caso)
        assert carta.palo == _PICA, (
            f"BotCastigador debe liderar ♠ con Q♠ activa, "
            f"eligió palo {carta.palo}"
        )

    def test_no_lidera_picas_si_qs_ya_fue_capturada(self) -> None:
        """Si Q♠ ya fue capturada, no hay razón para liderar ♠ agresivamente."""
        bot = BotCastigador()

        # J0 = BotCastigador con A♠, 2♣
        # J1 ya capturó Q♠ (simulado poniéndola en bazas_ganadas)
        m = _crear_motor_con_mano(
            cartas_j0=[_as_picas(), _dos_trebol()],
            cartas_j1=[_dos_corazon()],
            cartas_j2=[_carta_por_id(1)],
            cartas_j3=[_carta_por_id(2)],
        )
        m.jugadores[1].bazas_ganadas.append(_qs())

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # Con Q♠ capturada, no necesita forzar ♠
        # Puede liderar ♣ (conservador) o ♠, pero ♠ no es obligatorio
        # Verificamos que el bot no crashea y juega legal
        assert carta in legales

    def test_lidera_pica_mas_alta_para_forzar_qs(self) -> None:
        """Cuando lidera ♠ con Q♠ activa, debe elegir la ♠ MÁS ALTA."""
        bot = BotCastigador()

        # Baza 2+: J0 con A♠, 2♠, 2♣
        m = _crear_motor_con_mano(
            cartas_j0=[_as_picas(), _dos_picas(), _dos_trebol()],
            cartas_j1=[_qs(), _dos_corazon()],
            cartas_j2=[_carta_por_id(1), _carta_por_id(2)],
            cartas_j3=[_carta_por_id(3), _carta_por_id(4)],
            baza=2,
            corazones_rotos=True,
        )

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # Debe jugar la ♠ más alta (A♠ > 2♠)
        assert carta is _as_picas(), (
            f"BotCastigador debe liderar la ♠ más alta (A♠), "
            f"eligió {carta}"
        )

    def test_no_lidera_picas_si_bot_tiene_qs(self) -> None:
        """Si el bot mismo tiene Q♠, no debe liderar ♠ (se castigaría a sí mismo)."""
        bot = BotCastigador()

        # J0 = BotCastigador TIENE Q♠ y A♠
        m = _crear_motor_con_mano(
            cartas_j0=[_qs(), _as_picas(), _dos_trebol()],
            cartas_j1=[_dos_corazon()],
            cartas_j2=[_carta_por_id(1)],
            cartas_j3=[_carta_por_id(2)],
        )

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # Teniendo Q♠, no debe liderar ♠ voluntariamente
        # (a menos que sea forzado porque es su único palo)
        if carta.palo == _PICA:
            # Solo es aceptable si no tiene cartas de otros palos
            otros_palos = [c for c in legales if c.palo != _PICA]
            assert len(otros_palos) == 0, (
                f"BotCastigador lideró ♠ teniendo Q♠ y alternativas: {carta}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Clase 3 — Comportamiento siguiendo ♠
# ═══════════════════════════════════════════════════════════════════════════


class TestBotCastigadorSiguePicas:
    """Comportamiento cuando el bot debe seguir el palo ♠."""

    def test_juega_alta_siguiendo_picas_con_qs_activa(self) -> None:
        """Siguiendo ♠ con Q♠ activa, juega la más alta para presionar."""
        bot = BotCastigador()

        # J1 lidera 3♠ en baza 2+. J0 (BotCastigador) sigue con alta.
        m = _crear_motor_con_mano(
            cartas_j0=[_diez_picas(), _dos_picas()],
            cartas_j1=[_carta_por_id(28), _dos_corazon()],  # 3♠, 2♥
            cartas_j2=[_carta_por_id(1)],
            cartas_j3=[_carta_por_id(2)],
            baza=2,
            corazones_rotos=True,
            indice_inicial=1,  # J1 lidera
        )

        # J1 lidera 3♠
        m.jugar_carta(1, _carta_por_id(28))  # 3♠

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # Debe jugar 10♠ (más alta) para forzar al que tiene Q♠
        assert carta is _diez_picas(), (
            f"BotCastigador debe jugar la ♠ más alta siguiendo palo, "
            f"eligió {carta}"
        )

        # Debe jugar 10♠ (más alta) para forzar al que tiene Q♠
        assert carta is _diez_picas(), (
            f"BotCastigador debe jugar la ♠ más alta siguiendo palo, "
            f"eligió {carta}"
        )

    def test_juega_baja_siguiendo_picas_con_qs_en_mesa(self) -> None:
        """Si Q♠ ya está en la mesa, jugar baja para no ganar la baza."""
        bot = BotCastigador()

        # J1 lidera Q♠ en baza 2+. J0 debe seguir con ♠ baja.
        m = _crear_motor_con_mano(
            cartas_j0=[_as_picas(), _dos_picas()],
            cartas_j1=[_qs(), _dos_corazon()],
            cartas_j2=[_carta_por_id(1)],
            cartas_j3=[_carta_por_id(2)],
            baza=2,
            corazones_rotos=True,
            indice_inicial=1,  # J1 lidera
        )

        # Simular: J1 lideró con Q♠
        m.jugar_carta(1, _qs())

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # Con Q♠ en mesa, no queremos ganar la baza → jugar baja
        assert carta is _dos_picas(), (
            f"BotCastigador con Q♠ en mesa debe evitar ganar, "
            f"eligió {carta}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Clase 4 — Comportamiento general (no ♠)
# ═══════════════════════════════════════════════════════════════════════════


class TestBotCastigadorGeneral:
    """Comportamiento en situaciones que no involucran ♠."""

    def test_juega_conservador_en_otros_palos(self) -> None:
        """En palos que no son ♠, debe jugar conservador (carta más baja)."""
        bot = BotCastigador()

        m = _crear_motor_con_mano(
            cartas_j0=[_as_trebol(), _dos_trebol()],
            cartas_j1=[_dos_corazon()],
            cartas_j2=[_carta_por_id(1)],
            cartas_j3=[_carta_por_id(2)],
        )

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # En ♣ sin Q♠ involucrada, debe jugar conservador (2♣)
        assert carta is _dos_trebol(), (
            f"BotCastigador en palos seguros debe ser conservador, "
            f"eligió {carta}"
        )

    def test_descarta_corazones_si_es_void(self) -> None:
        """Si está void en el palo de salida, descarta corazones (castigo)."""
        bot = BotCastigador()

        # J0 = BotCastigador solo tiene ♥
        # J1 lidera ♣
        m = _crear_motor_con_mano(
            cartas_j0=[_as_corazon(), _dos_corazon()],
            cartas_j1=[_dos_trebol(), _carta_por_id(1)],
            cartas_j2=[_carta_por_id(2), _carta_por_id(3)],
            cartas_j3=[_carta_por_id(4), _carta_por_id(5)],
        )

        # J1 lidera 2♣
        m.jugar_carta(1, _dos_trebol())

        legales = m.obtener_jugadas_legales(0)
        carta = bot(m, 0, legales)

        # Debe descartar A♥ (la más alta) para "pintar" — castigar
        assert carta.es_corazon, (
            f"BotCastigador void debe descartar corazones, eligió {carta}"
        )

    def test_seleccion_es_deterministica(self) -> None:
        """Dos llamadas con el mismo estado deben dar la misma carta."""
        bot = BotCastigador()

        m1 = _crear_motor_con_mano(
            cartas_j0=[_as_picas(), _dos_picas()],
            cartas_j1=[_qs()],
            cartas_j2=[_carta_por_id(1)],
            cartas_j3=[_carta_por_id(2)],
        )

        m2 = _crear_motor_con_mano(
            cartas_j0=[_as_picas(), _dos_picas()],
            cartas_j1=[_qs()],
            cartas_j2=[_carta_por_id(1)],
            cartas_j3=[_carta_por_id(2)],
        )

        legales1 = m1.obtener_jugadas_legales(0)
        legales2 = m2.obtener_jugadas_legales(0)

        assert bot(m1, 0, legales1) is bot(m2, 0, legales2)


# ═══════════════════════════════════════════════════════════════════════════
# Clase 5 — Integración con MotorCorazones
# ═══════════════════════════════════════════════════════════════════════════


class TestBotCastigadorIntegracion:
    """Pruebas de integración: partidas completas con MotorCorazones."""

    def test_no_crashea_en_partida_completa(self) -> None:
        """Ejecutar una partida completa con BotCastigador no debe crashear."""
        bot = BotCastigador()
        m = MotorCorazones()
        m.repartir()

        for _ in range(52):
            idx = m.obtener_jugador_actual()
            legales = m.obtener_jugadas_legales(idx)
            carta = bot(m, idx, legales)
            assert carta in legales, f"Carta ilegal: {carta}"
            m.jugar_carta(idx, carta)
            if len(m.mesa) == 4:
                m.resolver_baza()

        # Si llegamos aquí sin excepciones, el test pasa
        assert True

    def test_estado_se_resetea_entre_manos(self) -> None:
        """El estado interno (voids, Q♠ tracking) debe resetearse entre manos."""
        bot = BotCastigador()

        # Primera mano
        m1 = MotorCorazones()
        m1.repartir()
        for _ in range(52):
            idx = m1.obtener_jugador_actual()
            legales = m1.obtener_jugadas_legales(idx)
            m1.jugar_carta(idx, bot(m1, idx, legales))
            if len(m1.mesa) == 4:
                m1.resolver_baza()

        # Segunda mano — debe funcionar sin carry-over de estado
        m2 = MotorCorazones()
        m2.repartir()
        for _ in range(52):
            idx = m2.obtener_jugador_actual()
            legales = m2.obtener_jugadas_legales(idx)
            carta = bot(m2, idx, legales)
            assert carta in legales


# ═══════════════════════════════════════════════════════════════════════════
# Clase 6 — Self-play (interacción con snapshot RL)
# ═══════════════════════════════════════════════════════════════════════════


class TestBotCastigadorSelfPlay:
    """Verifica que el bot puede integrarse como oponente en self-play."""

    def test_devuelve_siempre_carta_legal(self) -> None:
        """Fuzzing: 100 manos con reparto aleatorio, nunca carta ilegal."""
        bot = BotCastigador()

        for _ in range(100):
            m = MotorCorazones()
            m.repartir()
            for _ in range(52):
                idx = m.obtener_jugador_actual()
                legales = m.obtener_jugadas_legales(idx)
                if not legales:
                    break
                carta = bot(m, idx, legales)
                assert carta in legales, (
                    f"Carta {carta} no está en legales {[str(c) for c in legales]}"
                )
                m.jugar_carta(idx, carta)
                if len(m.mesa) == 4:
                    m.resolver_baza()
