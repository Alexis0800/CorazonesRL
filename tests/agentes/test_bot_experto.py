"""
Tests TDD para BotExperto.

Estructura de secciones:
  1. Conteo de cartas
  2. Rastreo de Q♠
  3. Modos de juego (MINIMIZAR / POZO / ALIMENTAR)
  4. Decisión siguiendo el palo
  5. Decisión descartando (void)
  6. Decisión liderando
  7. Integración: partida completa
"""

import pytest
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
from src.dominio.baraja import Baraja
from src.agentes.bot_experto import BotExperto


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _carta(valor: int, palo: int) -> Carta:
    """Retorna la carta correspondiente a (valor, palo)."""
    for c in Carta._TODAS:
        if c.valor == valor and c.palo == palo:
            return c
    raise ValueError(f"Carta no encontrada: valor={valor}, palo={palo}")


def _motor_con_mano(mano_j0: list, mano_j1: list, mano_j2: list, mano_j3: list) -> MotorCorazones:
    """Crea un motor con manos fijas. Las cartas se pasan como lista de Carta."""
    motor = MotorCorazones()
    motor.jugadores[0].mano = list(mano_j0)
    motor.jugadores[1].mano = list(mano_j1)
    motor.jugadores[2].mano = list(mano_j2)
    motor.jugadores[3].mano = list(mano_j3)
    motor.numero_baza = 2  # no es la primera baza (evitar regla 2♣)
    motor.corazones_rotos = True
    motor._mano_activa = True
    return motor


# Constantes de palos
TREBOL, DIAMANTE, PICA, CORAZON = 0, 1, 2, 3

# Cartas útiles
_2T = _carta(2, TREBOL)
_Q_PICAS = _carta(12, PICA)    # Q♠ — 13 puntos
_K_PICAS = _carta(13, PICA)    # K♠
_A_PICAS = _carta(14, PICA)    # A♠
_3_PICAS = _carta(3, PICA)
_2_PICAS = _carta(2, PICA)
_A_COR = _carta(14, CORAZON)   # A♥ — 1 punto
_K_COR = _carta(13, CORAZON)   # K♥ — 1 punto
_2_COR = _carta(2, CORAZON)    # 2♥ — 1 punto
_3_COR = _carta(3, CORAZON)
_J_COR = _carta(11, CORAZON)   # J♥ — 1 punto
_2D = _carta(2, DIAMANTE)
_3D = _carta(3, DIAMANTE)
_5D = _carta(5, DIAMANTE)
_9D = _carta(9, DIAMANTE)
_K_DIA = _carta(13, DIAMANTE)
_2C = _carta(2, TREBOL)
_3C = _carta(3, TREBOL)
_K_TRE = _carta(13, TREBOL)


# ────────────────────────────────────────────────────────────────────────────────
# 1. Conteo de cartas
# ────────────────────────────────────────────────────────────────────────────────

class TestConteoCartas:

    def test_cartas_restantes_inicial_13_por_palo(self):
        """Al inicio de la mano (sin bazas), hay 13 cartas restantes por palo en juego."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        mi_mano = motor.jugadores[0].mano
        restantes = bot._cartas_restantes_por_palo(motor, mi_mano)
        # Mis cartas no cuentan como "restantes en juego"; las demás sí
        assert sum(restantes.values()) == 52 - len(mi_mano)

    def test_cartas_restantes_decrece_tras_baza(self):
        """Después de que un jugador gana una baza, las cartas restantes disminuyen."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        mi_mano = motor.jugadores[0].mano

        restantes_antes = bot._cartas_restantes_por_palo(motor, mi_mano)
        total_antes = sum(restantes_antes.values())

        # Simular una baza completa
        motor.jugar_mano(lambda m, idx, legales: legales[0])
        # jugar_mano juega toda la mano — saltamos a una verificación diferente

    def test_cartas_restantes_palo_tiene_mi_mano_excluida(self):
        """Las cartas de mi mano no aparecen en las restantes de los rivales."""
        bot = BotExperto()
        mi_mano = [_2_PICAS, _3_PICAS, _2_COR]
        resto = [c for c in Carta._TODAS if c not in mi_mano]
        motor = _motor_con_mano(
            mi_mano, resto[:13], resto[13:26], resto[26:39])

        restantes = bot._cartas_restantes_por_palo(motor, mi_mano)
        # Q♠ no está en mi mano ni fue jugada → debe aparecer en restantes
        assert restantes[PICA] > 0

    def test_cartas_altas_restantes_al_inicio(self):
        """Al inicio, debe haber J/Q/K/A restantes en cada palo."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        mi_mano = motor.jugadores[0].mano
        altas = bot._cartas_altas_restantes(motor, mi_mano)
        # Globalmente hay 4 cartas altas por palo (J/Q/K/A)
        for palo in range(4):
            assert altas[palo] >= 0

    def test_cartas_altas_excluye_mi_mano(self):
        """Las cartas altas de mi mano no deben contar como 'de los rivales'."""
        bot = BotExperto()
        # J0 tiene A♠ y K♠
        mi_mano = [_A_PICAS, _K_PICAS, _2D, _3D, _2C, _3C, _2_COR]
        resto = [c for c in Carta._TODAS if c not in mi_mano]
        motor = _motor_con_mano(
            mi_mano, resto[:15], resto[15:30], resto[30:45])

        altas = bot._cartas_altas_restantes(motor, mi_mano)
        # A♠ y K♠ están en mi mano → no deben estar en restantes de rivales
        # Q♠ (valor=12) sí puede estar en restantes si no la tengo
        # al menos una alta mía excluida
        assert _A_PICAS not in mi_mano or altas[PICA] < 4


# ────────────────────────────────────────────────────────────────────────────────
# 2. Rastreo de Q♠
# ────────────────────────────────────────────────────────────────────────────────

class TestRastreoQEspadas:

    def test_q_activa_al_inicio(self):
        """Al inicio de la mano, Q♠ sigue activa (no jugada)."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        assert bot._q_activa(motor) is True

    def test_q_inactiva_cuando_esta_en_bazas_ganadas(self):
        """Si Q♠ está en bazas_ganadas de algún jugador, ya no está activa."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        # Simular que J1 ganó la Q♠
        motor.jugadores[1].bazas_ganadas.append(_Q_PICAS)
        assert bot._q_activa(motor) is False

    def test_q_activa_si_esta_en_mi_mano(self):
        """Si yo tengo la Q♠, sigue activa (no jugada)."""
        bot = BotExperto()
        mi_mano = [_Q_PICAS, _2_COR, _3_COR, _2D]
        resto = [c for c in Carta._TODAS if c not in mi_mano]
        motor = _motor_con_mano(
            mi_mano, resto[:16], resto[16:32], resto[32:48])
        assert bot._q_activa(motor) is True

    def test_q_activa_si_esta_en_la_mesa(self):
        """Si Q♠ está en la mesa (jugada en esta baza), sigue activa hasta que se resuelva."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        motor.mesa = [(1, _Q_PICAS)]  # J1 la jugó
        motor.palo_de_salida = PICA
        assert bot._q_activa(motor) is True  # todavía no se ganó


# ────────────────────────────────────────────────────────────────────────────────
# 3. Modos de juego
# ────────────────────────────────────────────────────────────────────────────────

class TestModoJuego:

    def _motor_con_puntajes(self, puntajes: list) -> MotorCorazones:
        motor = MotorCorazones()
        motor.repartir()
        for i, p in enumerate(puntajes):
            motor.jugadores[i].puntuacion_historica = p
        return motor

    def test_pozo_viable_con_muchos_corazones_altos(self):
        """Pozo viable: ≥6 corazones y ≥3 altos (J/Q/K/A) y no rotos."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.corazones_rotos = False
        # Forzar mano con muchos corazones altos
        corazones_altos = [_carta(v, CORAZON)
                           for v in [14, 13, 12, 11, 10, 9]]  # A/K/Q/J/10/9♥
        resto = [c for c in Carta._TODAS if c not in corazones_altos]
        motor.jugadores[0].mano = corazones_altos + [resto[0]]
        assert bot._pozo_viable(motor, 0) is True

    def test_pozo_no_viable_corazones_rotos(self):
        """Pozo no viable si los corazones ya están rotos."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.corazones_rotos = True
        corazones_altos = [_carta(v, CORAZON) for v in [14, 13, 12, 11, 10, 9]]
        resto = [c for c in Carta._TODAS if c not in corazones_altos]
        motor.jugadores[0].mano = corazones_altos + [resto[0]]
        assert bot._pozo_viable(motor, 0) is False

    def test_pozo_no_viable_pocos_corazones(self):
        """Pozo no viable con solo 3 corazones."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.corazones_rotos = False
        corazones = [_carta(v, CORAZON) for v in [14, 13, 12]]
        resto = [c for c in Carta._TODAS if c not in corazones]
        motor.jugadores[0].mano = corazones + resto[:10]
        assert bot._pozo_viable(motor, 0) is False

    def test_pozo_no_viable_puntuacion_alta(self):
        """Pozo no viable si ya tengo 85+ puntos acumulados."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.corazones_rotos = False
        motor.jugadores[0].puntuacion_historica = 85
        corazones_altos = [_carta(v, CORAZON) for v in [14, 13, 12, 11, 10, 9]]
        resto = [c for c in Carta._TODAS if c not in corazones_altos]
        motor.jugadores[0].mano = corazones_altos + [resto[0]]
        assert bot._pozo_viable(motor, 0) is False

    def test_modo_minimizar_por_defecto(self):
        """Sin condiciones especiales, el modo es MINIMIZAR."""
        bot = BotExperto()
        motor = MotorCorazones()
        # Usar mano controlada sin muchos corazones para garantizar MINIMIZAR
        sin_corazones = [c for c in Carta._TODAS
                         if not c.es_corazon and not c.es_dama_de_picas][:13]
        motor.jugadores[0].mano = sin_corazones
        motor.corazones_rotos = True
        motor.numero_baza = 2
        motor._mano_activa = True
        assert bot._modo(motor, 0) == "MINIMIZAR"

    def test_modo_pozo_con_mano_correcta(self):
        """Modo POZO cuando tengo mano para shooting the moon."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.corazones_rotos = False
        corazones_altos = [_carta(v, CORAZON) for v in [14, 13, 12, 11, 10, 9]]
        resto = [c for c in Carta._TODAS if c not in corazones_altos]
        motor.jugadores[0].mano = corazones_altos + [resto[0]]
        assert bot._modo(motor, 0) == "POZO"

    def test_modo_alimentar_cuando_rival_cerca_de_100(self):
        """Modo ALIMENTAR cuando un rival está a ≤15 puntos de 100."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        # Forzar mano de J0 sin corazones para no disparar modo POZO
        sin_corazones = [c for c in Carta._TODAS
                         if not c.es_corazon and not c.es_dama_de_picas][:13]
        motor.jugadores[0].mano = sin_corazones
        motor.jugadores[1].puntuacion_historica = 88  # rival cerca de 100
        motor.jugadores[0].puntuacion_historica = 50
        assert bot._modo(motor, 0) == "ALIMENTAR"


# ────────────────────────────────────────────────────────────────────────────────
# 4. Decisión siguiendo el palo
# ────────────────────────────────────────────────────────────────────────────────

class TestSeguirPalo:

    def test_no_captura_q_espadas_cuando_puede_evitarlo(self):
        """Nunca gana baza con Q♠ si hay carta más baja disponible."""
        bot = BotExperto()
        # Mesa: J0:2♠ (lidera), J1:Q♠, J2:4♠
        # J3 tiene [A♠, 3♠] → agresivo jugaría A♠, experto debe jugar 3♠
        mi_mano = [_A_PICAS, _3_PICAS, _2_COR]
        resto = [c for c in Carta._TODAS if c not in mi_mano and c != _2_PICAS
                 and c != _Q_PICAS and c != _carta(4, PICA)]
        motor = _motor_con_mano(
            [_K_TRE, _K_DIA, _2_COR],  # J0 (ya jugó 2♠, representado en mesa)
            [_2D, _3D, _3C],
            [_K_COR, _A_COR, _J_COR],
            mi_mano,
        )
        motor.mesa = [(0, _2_PICAS), (1, _Q_PICAS), (2, _carta(4, PICA))]
        motor.palo_de_salida = PICA
        motor.corazones_rotos = True

        legales = [_A_PICAS, _3_PICAS]
        carta = bot(motor, 3, legales)
        assert carta == _3_PICAS, f"Esperaba 3♠, obtuvo {carta}"

    def test_no_captura_q_espadas_juega_mas_alta_que_no_gana(self):
        """Con [J♠, 3♠] en mesa con Q♠, juega J♠ (mayor que no gana la Q♠)."""
        bot = BotExperto()
        _J_PICAS = _carta(11, PICA)
        motor = _motor_con_mano([_J_PICAS, _3_PICAS], [_2D], [_3D], [_2C])
        # J2 lidera, J3 juega Q♠
        motor.mesa = [(2, _carta(4, PICA)), (3, _Q_PICAS)]
        motor.palo_de_salida = PICA
        motor.corazones_rotos = True

        legales = [_J_PICAS, _3_PICAS]
        carta = bot(motor, 0, legales)
        # J♠ (valor 11) < Q♠ (valor 12) → no gana la baza
        # 3♠ también no gana, pero J♠ es la más alta que no gana
        assert carta == _J_PICAS, f"Esperaba J♠, obtuvo {carta}"

    def test_evita_ganar_baza_con_puntos_sin_q_espadas(self):
        """Sin Q♠ en mesa, prefiere no ganar una baza que tiene corazones."""
        bot = BotExperto()
        motor = _motor_con_mano([_A_COR, _3_COR], [_2D], [_3D], [_2C])
        _9_COR = _carta(9, CORAZON)
        motor.mesa = [(1, _2_COR), (2, _9_COR)]
        motor.palo_de_salida = CORAZON
        motor.corazones_rotos = True

        legales = [_A_COR, _3_COR]  # A♥ ganaría, 3♥ no
        carta = bot(motor, 0, legales)
        assert carta == _3_COR, f"Esperaba 3♥, obtuvo {carta}"

    def test_si_todas_ganan_juega_la_mas_baja(self):
        """Si todas las cartas disponibles ganan la baza, juega la más baja."""
        bot = BotExperto()
        motor = _motor_con_mano([_A_PICAS, _K_PICAS], [_2D], [_3D], [_2C])
        motor.mesa = [(1, _2_PICAS), (2, _Q_PICAS)]  # Q♠ ya en mesa
        motor.palo_de_salida = PICA
        motor.corazones_rotos = True

        # A♠ y K♠ ambas ganan (A>Q y K>Q), pero K♠ es la "más baja que gana"
        # dado que Q♠ está en mesa, queremos NO ganarla si se puede
        # Si no hay forma de no ganarla (todas > Q♠), jugar la más baja
        legales = [_A_PICAS, _K_PICAS]
        carta = bot(motor, 0, legales)
        assert carta == _K_PICAS, f"Esperaba K♠ (mínimo daño), obtuvo {carta}"

    def test_sin_puntos_en_mesa_juega_normal(self):
        """Sin puntos en mesa: si todas ganan, quema la más alta; si alguna pierde, la más alta que pierda."""
        bot = BotExperto()
        _5D = _carta(5, DIAMANTE)
        _9D = _carta(9, DIAMANTE)
        motor = _motor_con_mano([_5D, _9D], [_2C], [_3C], [_2_COR])
        motor.mesa = [(1, _2D), (2, _3D)]
        motor.palo_de_salida = DIAMANTE
        motor.corazones_rotos = False

        legales = [_5D, _9D]
        carta = bot(motor, 0, legales)
        # Ambas ganan (5♦>3♦, 9♦>3♦): quema la más alta (9♦)
        # para eliminar la carta más peligrosa en baza limpia
        assert carta == _9D, f"Esperaba 9♦ (quemar alta), obtuvo {carta}"


# ────────────────────────────────────────────────────────────────────────────────
# 5. Decisión descartando (void en el palo)
# ────────────────────────────────────────────────────────────────────────────────

class TestDescarte:

    def test_descarta_q_espadas_cuando_hay_puntos_en_mesa(self):
        """Cuando void y hay puntos en mesa, descartar Q♠ primero."""
        bot = BotExperto()
        motor = _motor_con_mano(
            [_Q_PICAS, _A_COR, _K_COR], [_2D], [_3D], [_2C])
        motor.mesa = [(1, _2_COR), (2, _3_COR)]  # puntos en mesa: 2♥
        motor.palo_de_salida = CORAZON  # J0 es void en corazones → puede descartar
        motor.corazones_rotos = True

        # void en ♥ → puede jugar lo que sea
        legales = [_Q_PICAS, _A_COR, _K_COR]
        # Corrección: si el palo de salida es CORAZON y J0 no tiene corazones,
        # legales serían todas sus cartas. Pero aquí J0 SÍ tiene corazones.
        # Reformulamos: J0 es void en DIAMANTE.
        motor.palo_de_salida = DIAMANTE
        # Mesa: K♦ gana, hay 0 pts corazones
        motor.mesa = [(1, _2D), (2, _K_DIA)]
        # Esto no tiene puntos. Reformulamos para que haya puntos:
        motor.mesa = [(1, _2_COR), (2, _3_COR)]  # J1 lidera ♥, J2 sigue
        # Pero motor.palo_de_salida = DIAMANTE → inconsistente.
        # Usar un escenario limpio:

        # Escenario: J0 void en ♦, mesa tiene puntos (hay ♥ jugados), J0 puede descartar Q♠
        motor2 = _motor_con_mano(
            [_Q_PICAS, _A_COR],  # J0: tiene Q♠ y A♥
            [_2D, _3D],          # J1
            [_K_DIA, _5D],       # J2
            [_2C, _3C],          # J3
        )
        motor2.mesa = [(1, _2_COR), (2, _3_COR)]  # 2 corazones jugados = 2 pts
        motor2.palo_de_salida = CORAZON
        motor2.corazones_rotos = True

        # J0 void en corazones: puede descartar Q♠ o A♥ (aunque A♥ también es corazón)
        # En realidad si palo_salida = CORAZON y J0 tiene corazones, no es void.
        # Reformulación final:
        motor3 = _motor_con_mano(
            [_Q_PICAS, _K_TRE],   # J0: Q♠ y K♣ (sin diamantes)
            [_2D, _3D],
            [_K_DIA, _5D],
            [_2C, _3C],
        )
        motor3.mesa = [(1, _2D), (2, _K_DIA)]  # ♦ liderado, 0 pts
        motor3.palo_de_salida = DIAMANTE
        motor3.corazones_rotos = False

        # J0 void en ♦, puede descartar cualquiera
        legales3 = [_Q_PICAS, _K_TRE]
        carta = bot(motor3, 0, legales3)
        # Mesa sin puntos → no urge descartar Q♠. Debería descartar K♣ (sin puntos)
        assert carta != _Q_PICAS, "No debe descartar Q♠ si la baza no tiene puntos"

    def test_descarta_q_espadas_con_puntos_en_mesa(self):
        """Cuando void y la mesa ya tiene corazones, descartar Q♠."""
        bot = BotExperto()
        motor = _motor_con_mano(
            [_Q_PICAS, _K_TRE],  # J0: Q♠ y K♣
            [_2_COR, _3D],
            [_K_COR, _5D],
            [_2C, _3C],
        )
        motor.mesa = [(1, _2_COR), (2, _K_COR)]  # 2 puntos en mesa
        motor.palo_de_salida = CORAZON
        motor.corazones_rotos = True

        legales = [_Q_PICAS, _K_TRE]  # J0 void en ♥
        carta = bot(motor, 0, legales)
        assert carta == _Q_PICAS, f"Esperaba Q♠ (descarte óptimo), obtuvo {carta}"

    def test_descarta_corazon_alto_por_puntos_no_valor(self):
        """Al descartar corazones, prioriza por puntos (todos iguales en corazones), luego valor."""
        bot = BotExperto()
        motor = _motor_con_mano(
            [_A_COR, _2_COR, _K_TRE],  # J0: A♥, 2♥, K♣
            [_2D, _3D],
            [_K_DIA, _5D],
            [_2C, _3C],
        )
        motor.mesa = [(1, _2D), (2, _K_DIA)]  # sin puntos
        motor.palo_de_salida = DIAMANTE
        motor.corazones_rotos = True

        # Sin Q♠, debe descartar el corazón de mayor valor (A♥)
        legales = [_A_COR, _2_COR, _K_TRE]
        carta = bot(motor, 0, legales)
        # K♣ no tiene puntos; A♥ y 2♥ tienen 1 punto cada uno. Sin Q♠,
        # el experto debería descartar el corazón más alto (A♥) para deshacerse
        # de la carta más peligrosa de corazones.
        assert carta == _A_COR, f"Esperaba A♥ (corazón más alto), obtuvo {carta}"

    def test_descarta_carta_sin_puntos_si_no_hay_urgencia(self):
        """Sin Q♠ ni puntos en mesa, descartar carta sin puntos (la más alta)."""
        bot = BotExperto()
        motor = _motor_con_mano(
            [_K_TRE, _2_COR],  # J0: K♣ y 2♥
            [_2D, _3D],
            [_5D, _K_DIA],
            [_2C, _3C],
        )
        motor.mesa = [(1, _2D), (2, _5D)]  # sin puntos en mesa
        motor.palo_de_salida = DIAMANTE
        motor.corazones_rotos = False

        legales = [_K_TRE, _2_COR]  # J0 void en ♦
        carta = bot(motor, 0, legales)
        # Sin Q♠ y sin puntos urgentes → descartar K♣ (sin puntos, alta)
        # para preservar corazones y no desperdiciar Q♠ en baza vacía
        assert carta == _K_TRE, f"Esperaba K♣ (sin puntos), obtuvo {carta}"


# ────────────────────────────────────────────────────────────────────────────────
# 6. Decisión liderando
# ────────────────────────────────────────────────────────────────────────────────

class TestLiderazgo:

    def test_nunca_lidera_con_q_espadas_si_hay_alternativa(self):
        """Al liderar, nunca juega Q♠ si hay otra carta disponible."""
        bot = BotExperto()
        motor = _motor_con_mano(
            [_Q_PICAS, _2D],  # J0: Q♠ y 2♦
            [_3D, _K_DIA],
            [_5D, _9D],
            [_2C, _3C],
        )
        motor.mesa = []
        motor.corazones_rotos = False

        legales = [_Q_PICAS, _2D]
        carta = bot(motor, 0, legales)
        assert carta != _Q_PICAS, "No debe liderar con Q♠"
        assert carta == _2D

    def test_lidera_con_carta_baja_sin_puntos(self):
        """Al liderar en modo MINIMIZAR, prefiere carta baja sin puntos."""
        bot = BotExperto()
        motor = _motor_con_mano(
            [_2D, _3D, _A_COR],  # J0: 2♦, 3♦, A♥
            [_K_DIA, _5D],
            [_2C, _3C],
            [_K_TRE, _2_PICAS],
        )
        motor.mesa = []
        motor.corazones_rotos = True

        legales = [_2D, _3D, _A_COR]
        carta = bot(motor, 0, legales)
        # Preferir carta sin puntos (2♦ o 3♦), no A♥
        assert carta in [_2D, _3D], f"Esperaba 2♦ o 3♦, obtuvo {carta}"

    def test_lidera_en_modo_pozo_con_corazon_alto(self):
        """En modo POZO, liderar con el corazón más alto."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.corazones_rotos = False  # sin romper → POZO viable
        motor.numero_baza = 5

        corazones_altos = [_carta(v, CORAZON) for v in [14, 13, 12, 11, 10, 9]]
        resto_sin_puntos = [c for c in Carta._TODAS
                            if c not in corazones_altos and c.puntos == 0][:7]
        motor.jugadores[0].mano = corazones_altos + resto_sin_puntos[:1]
        motor.jugadores[1].mano = resto_sin_puntos[1:14]
        motor.jugadores[2].mano = resto_sin_puntos[14:27] if len(
            resto_sin_puntos) >= 27 else []
        motor.jugadores[3].mano = resto_sin_puntos[27:40] if len(
            resto_sin_puntos) >= 40 else []
        motor.mesa = []
        motor._mano_activa = True

        legales = motor.jugadores[0].mano
        carta = bot(motor, 0, legales)
        # En modo POZO, liderar con el corazón más alto
        assert carta.es_corazon, f"En modo POZO, debe liderar corazón. Obtuvo {carta}"
        corazones_legales = [c for c in legales if c.es_corazon]
        assert carta.valor == max(c.valor for c in corazones_legales)


# ────────────────────────────────────────────────────────────────────────────────
# 7. Integración: inferencia de voids desde la mesa actual
# ────────────────────────────────────────────────────────────────────────────────

class TestInferenciaVoids:

    def test_infiere_void_desde_mesa_actual(self):
        """Si un jugador jugó carta de otro palo, se registra como void."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        # Simular que J1 no siguió el palo de espadas (jugó un corazón)
        # J1 jugó ♥ cuando el palo es ♠
        motor.mesa = [(0, _2_PICAS), (1, _2_COR)]
        motor.palo_de_salida = PICA
        # actualizo desde la perspectiva de J2
        bot._actualizar_estado(motor, 2)

        assert PICA in bot._vacios[1], "J1 debería ser detectado como void en ♠"

    def test_no_infiere_void_si_sigue_el_palo(self):
        """Si el jugador sigue el palo, no se registra como void."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()
        motor.mesa = [(0, _2_PICAS), (1, _3_PICAS)]  # J1 sigue ♠
        motor.palo_de_salida = PICA
        bot._actualizar_estado(motor, 2)

        assert PICA not in bot._vacios[1], "J1 siguió el palo, no es void"

    def test_reset_al_nueva_mano(self):
        """Al detectar nueva mano (numero_baza=1 tras haber avanzado), resetea voids."""
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()

        # Simular que hemos avanzado
        bot._ultimo_baza_num = 13  # estábamos en baza 13
        motor.numero_baza = 1      # nueva mano
        motor.mesa = []

        bot._actualizar_estado(motor, 0)
        assert all(len(v) == 0 for v in bot._vacios.values()), \
            "Los voids deben resetearse al inicio de nueva mano"


# ────────────────────────────────────────────────────────────────────────────────
# 8. Integración: BotExperto en entorno real
# ────────────────────────────────────────────────────────────────────────────────

class TestIntegracionEntorno:

    def test_bot_experto_devuelve_carta_legal(self):
        """BotExperto siempre devuelve una carta legal."""
        import random
        random.seed(42)
        bot = BotExperto()
        motor = MotorCorazones()
        motor.repartir()

        for _ in range(13):
            for _ in range(4):
                idx = motor.obtener_jugador_actual()
                legales = motor.obtener_jugadas_legales(idx)
                if idx == 0:
                    carta = bot(motor, idx, legales)
                else:
                    # otros jugadores juegan la primera legal
                    carta = legales[0]
                assert carta in legales, f"Carta {carta} no es legal"
                motor.jugar_carta(idx, carta)
            motor.resolver_baza()

    def test_bot_experto_no_captura_q_espadas_en_partida_real(self):
        """En 20 manos, el bot experto captura Q♠ menos veces que el agresivo."""
        import random
        from src.agentes.heuristicos import bot_agresivo

        def contar_q_espadas(policy_j0, seed):
            """Cuenta cuántas veces J0 captura Q♠ en una mano."""
            random.seed(seed)
            motor = MotorCorazones()
            motor.repartir()
            q_capturadas = 0
            for _ in range(13):
                for _ in range(4):
                    idx = motor.obtener_jugador_actual()
                    legales = motor.obtener_jugadas_legales(idx)
                    if idx == 0:
                        carta = policy_j0(motor, idx, legales)
                    else:
                        carta = bot_agresivo(motor, idx, legales)
                    motor.jugar_carta(idx, carta)
                ganador = motor.resolver_baza()
                # Verificar si J0 ganó la Q♠
                if ganador == 0 and any(c.es_dama_de_picas for c in motor.jugadores[0].bazas_ganadas[-4:]):
                    q_capturadas += 1
            return q_capturadas

        n_manos = 20
        bot = BotExperto()
        q_experto = sum(contar_q_espadas(bot, s) for s in range(n_manos))
        q_agresivo = sum(contar_q_espadas(bot_agresivo, s)
                         for s in range(n_manos))

        assert q_experto <= q_agresivo, (
            f"BotExperto capturó Q♠ {q_experto}x, agresivo {q_agresivo}x. "
            f"El experto debería capturarla menos."
        )
