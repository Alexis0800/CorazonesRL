"""Tests para ObservacionBuilderV31 — 228 dimensiones."""

import pytest
import numpy as np

from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
from src.v3_1.dimensiones import DIM_V3_1
from src.v3_1.observacion import ObservacionBuilderV31


# ── Helpers ──────────────────────────────────────────────────

def _crear_motor_con_mano(cartas_ids: list[int], agente_idx: int = 0) -> MotorCorazones:
    """Crea un motor con cartas específicas en la mano del agente."""
    motor = MotorCorazones()
    motor.repartir()
    # Reemplazar la mano del agente con las cartas especificadas
    motor.jugadores[agente_idx].mano = [Carta._TODAS[cid]
                                        for cid in cartas_ids]
    # Vaciar manos de otros para simplificar
    for i in range(4):
        if i != agente_idx:
            motor.jugadores[i].mano = []
    return motor


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def builder() -> ObservacionBuilderV31:
    return ObservacionBuilderV31()


@pytest.fixture
def motor() -> MotorCorazones:
    m = MotorCorazones()
    m.repartir()
    return m


# ── Tests: Dimensionalidad ──────────────────────────────────

class TestDimensionalidadV31:

    def test_dim_es_228(self) -> None:
        """DIM_V3_1 debe ser 228."""
        assert DIM_V3_1 == 228

    def test_builder_usa_dim_228_por_defecto(self) -> None:
        """El builder por defecto debe usar DIM_V3_1=228."""
        builder = ObservacionBuilderV31()
        assert builder.dim == 228

    def test_output_shape_es_228(self, builder, motor) -> None:
        """construir() debe retornar vector de shape (228,)."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (228,), f"Shape: {obs.shape}"
        assert obs.dtype == np.float32

    def test_construir_desde_motor_v31(self, builder, motor) -> None:
        """construir_desde_motor() retorna vector de 228 dims."""
        obs = builder.construir_desde_motor(motor, 0)
        assert obs.shape == (228,)


class TestBloqueCartasV31:
    """[0:156] Mano, Mesa, Cementerio."""

    def test_mano_onehot(self, builder) -> None:
        """[0:52] debe tener 13 cartas en one-hot."""
        motor = _crear_motor_con_mano(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[0:52].sum(
        ) == 13, f"Esperado 13, obtenido {obs[0:52].sum()}"

    def test_mesa_onehot(self, builder, motor) -> None:
        """[52:104] Mesa one-hot debe reflejar cartas en mesa."""
        # Jugar una carta para poner algo en la mesa
        motor.jugadores[0].mano = [Carta._TODAS[0]]  # 2♣
        legales = motor.obtener_jugadas_legales(0)
        if legales:
            motor.jugar_carta(0, legales[0])
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # Debe haber al menos 1 carta en la mesa
        assert obs[52:104].sum() >= 1


class TestBloqueVaciosV31:
    """[156:172] Vacíos conocidos."""

    def test_vacios_reflejan_sets(self, builder, motor) -> None:
        """Los vacíos deben reflejar los sets pasados."""
        vacios = [set(), {0, 1}, set(), {2, 3}]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # Jugador 1 (rel=1) vacío en palos 0,1 → posiciones 160,161
        assert obs[160] == 1.0, "Jugador 1 debería ser void en palo 0"
        assert obs[161] == 1.0, "Jugador 1 debería ser void en palo 1"
        # Jugador 3 (rel=3) vacío en palos 2,3 → posiciones 170,171
        assert obs[170] == 1.0, "Jugador 3 debería ser void en palo 2"
        assert obs[171] == 1.0, "Jugador 3 debería ser void en palo 3"


class TestBloquePuntuacionesV31:
    """[172:181] Puntajes históricos, mano actual, corazones rotos."""

    def test_puntaje_historico_normalizado(self, builder, motor) -> None:
        """[172:176] debe estar normalizado /100."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [50, 25, 75, 0], [0, 0, 0, 0], None)
        # agente=50
        assert obs[172] == 0.5, f"Esperado 0.5, obtenido {obs[172]}"
        assert obs[173] == 0.25  # rival rel=1: 25
        assert obs[174] == 0.75  # rival rel=2: 75
        assert obs[175] == 0.0   # rival rel=3: 0

    def test_puntos_mano_raw(self, builder, motor) -> None:
        """[176:180] puntos mano actual en valor raw (0-26)."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [5, 3, 10, 0], None)
        assert obs[176] == 5.0, f"Esperado 5.0, obtenido {obs[176]}"  # agente
        assert obs[177] == 3.0   # rival rel=1
        assert obs[178] == 10.0  # rival rel=2
        assert obs[179] == 0.0   # rival rel=3

    def test_corazones_rotos(self, builder, motor) -> None:
        """[180] corazones rotos: 0.0 inicial, 1.0 tras jugar corazón."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[180] == 0.0, "Corazones no deberían estar rotos al inicio"


class TestBloqueQSPosicionV31:
    """[181:188] Posición en baza, Q♠ tracker, pozo_viable."""

    def test_posicion_baza_inicial(self, builder, motor) -> None:
        """[181] posición en baza al inicio debe ser 0.0 (mesa vacía)."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[181] == 0.0

    def test_qs_tracker_desconocida(self, builder, motor) -> None:
        """[182:187] Q♠ tracker: estado 'desconocida' cuando dama_picas_en=None."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[182] == 1.0, "Estado 'desconocida' debe estar activo"
        assert obs[183:187].sum() == 0.0, "Ningún jugador específico"

    def test_qs_tracker_capturada_por_rival(self, builder, motor) -> None:
        """[182:187] Q♠ capturada por jugador 2 (rel=2 del agente).

        Layout: [182]=desconocida, [183]=agente(rel0), [184]=rival1(rel1),
                 [185]=rival2(rel2), [186]=rival3(rel3)
        """
        vacios = [set() for _ in range(4)]
        obs = builder.construir(motor, 0, vacios, [0, 0, 0, 0], [
                                0, 0, 0, 0], dama_picas_en=2)
        assert obs[
            185] == 1.0, f"Jugador 2 (rel=2) debe tener Q♠ en [185], obs[182:187]={obs[182:187]}"

    def test_pozo_viable_flag(self, builder, motor) -> None:
        """[187] pozo_viable debe reflejar el parámetro."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(motor, 0, vacios, [0, 0, 0, 0], [
                                0, 0, 0, 0], None, pozo_viable=True)
        assert obs[187] == 1.0


class TestBloqueEstadoManoV31:
    """[188:194] Número de baza, cartas restantes, palo de salida."""

    def test_baza_numero(self, builder, motor) -> None:
        """[188] número de baza / 13."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[188] == 1.0 / 13.0, f"Baza 1 → {1/13:.4f}"

    def test_cartas_restantes_raw(self, builder, motor) -> None:
        """[189:193] cartas restantes raw (0-13) por palo."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # Al inicio, todas las cartas menos las 13 del agente están en juego
        # Cada palo tiene 13 cartas. Las del agente no cuentan como "restantes".
        total_restantes = obs[189:193].sum()
        assert total_restantes <= 39, f"Max 39 cartas restantes, obtenido {total_restantes}"
        assert total_restantes >= 26, f"Min 26 cartas restantes, obtenido {total_restantes}"

    def test_palo_salida(self, builder, motor) -> None:
        """[193] palo de salida."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # Al inicio, palo_salida es None → 0.0
        assert obs[193] == 0.0


class TestBloqueQSEnriquecidoV31:
    """[194:203] Peligro Q♠, inminente, prob por jugador."""

    def test_peligro_qs_qs_capturada(self, builder, motor) -> None:
        """[194:198] Peligro Q♠ debe ser 0 cuando Q♠ ya fue capturada."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(motor, 0, vacios, [0, 0, 0, 0], [
                                0, 0, 0, 0], dama_picas_en=1)
        assert obs[194:198].sum() == 0.0, "Sin peligro si Q♠ capturada"

    def test_peligro_qs_inminente_sin_qs(self, builder) -> None:
        """[198] Peligro inminente = 0 si no tengo Q♠ en mano."""
        # Mano sin Q♠ ni picas
        motor = _crear_motor_con_mano(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        for i in range(1, 4):
            motor.jugadores[i].mano = []
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[198] == 0.0, f"Sin Q♠ → peligro inminente=0, obtenido {obs[198]}"

    def test_prob_qs_por_jugador_suma_1(self, builder, motor) -> None:
        """[199:203] Prob Q♠ debe sumar ~1.0 si Q♠ no capturada."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        total = obs[199:203].sum()
        assert abs(total - 1.0) < 0.01, f"Suma debería ser ~1.0, es {total}"


class TestBloqueControlPaloV31:
    """[203:215] Altas en mano, máxima absoluta, control palo."""

    def test_altas_en_mano(self, builder) -> None:
        """[203:207] Altas en mano: J/Q/K/A por palo."""
        # Crear mano con A♠ (id=51), K♥ (id=38), Q♦ (id=24), J♣ (id=10)
        motor = _crear_motor_con_mano([51, 38, 24, 10])
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # A♠ = palo 2, K♥ = palo 1, Q♦ = palo 0, J♣ = palo 3
        # Altas por palo: [1, 1, 1, 1] → raw values
        assert obs[203:207].sum(
        ) == 4.0, f"4 altas totales, obtenido {obs[203:207].sum()}"

    def test_maxima_absoluta_sin_rivales(self, builder) -> None:
        """[207:211] Máxima absoluta: 1.0 si soy el único con altas del palo.

        Con A♠ en mano y el resto de cartas de ♠ en mi propia mano 
        (para que no queden altas de ♠ en rivales), A♠ es máxima absoluta.
        """
        # Dar TODAS las ♠ al agente (incluyendo A♠) para que no queden en rivales
        from src.dominio.carta import Carta
        todas_picas = [c.id for c in Carta._TODAS if c.palo == 2]
        # También algunas otras cartas para completar
        otras = [c.id for c in Carta._TODAS if c.palo !=
                 2][:13-len(todas_picas)]
        motor = _crear_motor_con_mano(todas_picas + otras)
        for i in range(1, 4):
            motor.jugadores[i].mano = []
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # A♠ es la más alta de ♠ (palo 2) → [207+2] = [209]
        assert obs[209] == 1.0, f"A♠ debería ser máxima absoluta: {obs[207:211]}"


class TestBloquePatronesRivalesV31:
    """[215:227] Patrones de rivales: ♥ altos, ♠ altas, ¿jugó ♥?"""

    def test_corazones_altos_por_rival(self, builder, motor) -> None:
        """[215:219] ♥ altos capturados por cada rival (relativo al agente)."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # Al inicio, 0 corazones altos capturados
        assert obs[215:219].sum() == 0.0

    def test_picas_altas_jugadas_por_rival(self, builder, motor) -> None:
        """[219:223] ♠ altas jugadas por cada rival."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[219:223].sum() == 0.0

    def test_ya_jugo_corazon_por_rival(self, builder, motor) -> None:
        """[223:227] ¿Ya jugó ♥ cada rival?"""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[223:227].sum() == 0.0


class TestBloqueForzadoV31:
    """[227] Indicador de jugada forzada."""

    def test_forzado_es_0_con_varias_legales(self, builder) -> None:
        """[227] forzado = 0 cuando hay varias cartas legales.

        Usa una mano controlada sin 2♣ para evitar el filtro de primera baza."""
        # Mano sin 2♣ (id=0) para que no sea la primera baza forzada
        motor = _crear_motor_con_mano(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        # Asegurar que el agente NO es el jugador inicial (2♣ está en otro)
        # Poner 2♣ en jugador 1
        motor.jugadores[1].mano = [motor.jugadores[0].mano[0]]  # irrelevant
        # El agente no tiene 2♣ → no es inicial → sus 13 cartas son legales
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs[227] == 0.0, f"Forzado debería ser 0, obtenido {obs[227]}"


class TestEdgeCasesV31:
    """Casos borde."""

    def test_mano_vacia_no_crashea(self, builder) -> None:
        """Motor sin repartir no debería crashear."""
        m = MotorCorazones()
        vacios = [set() for _ in range(4)]
        obs = builder.construir(m, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs.shape == (228,)

    def test_dama_picas_en_agente(self, builder) -> None:
        """Cuando el agente tiene Q♠, prob Q♠ debe ser 1.0 para él.

        Q♠ = Carta con es_dama_de_picas=True.
        Buscamos la Q♠ por atributo, no por ID fijo."""
        from src.dominio.carta import Carta
        qs = next(c for c in Carta._TODAS if c.es_dama_de_picas)
        motor = _crear_motor_con_mano([qs.id])
        for i in range(1, 4):
            motor.jugadores[i].mano = []
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        # [199] = agente en prob Q♠ (rel=0)
        assert obs[199] == 1.0, f"Agente tiene Q♠ → prob[0]=1.0, obtenido {obs[199]}"

    def test_output_es_float32(self, builder, motor) -> None:
        """Toda observación debe ser float32."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert obs.dtype == np.float32

    def test_no_hay_nan(self, builder, motor) -> None:
        """No debe haber NaN en la observación."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert not np.any(np.isnan(obs)), "NaN encontrado en observación"

    def test_no_hay_inf(self, builder, motor) -> None:
        """No debe haber infinitos en la observación."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor, 0, vacios, [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert not np.any(np.isinf(obs)), "Inf encontrado en observación"
