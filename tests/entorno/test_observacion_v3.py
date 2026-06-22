"""
Tests unitarios para ObservacionBuilderV3 (250 dimensiones).

Verifica que cada bloque de features enriquecidas se construye correctamente
y que el vector total tiene la dimensionalidad y tipo esperados.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.v3.observacion import ObservacionBuilderV3, DIM_V3


class TestObservacionBuilderV3:
    """Suite de tests para el constructor de observación v3 (250-d)."""

    @pytest.fixture
    def motor(self) -> MotorCorazones:
        """Motor fresco con una mano recién repartida."""
        m = MotorCorazones()
        m.repartir()
        return m

    @pytest.fixture
    def builder(self) -> ObservacionBuilderV3:
        """Builder v3 con 260 dimensiones."""
        return ObservacionBuilderV3()

    # ------------------------------------------------------------------
    # Dimensionalidad
    # ------------------------------------------------------------------

    def test_dim_v3_es_260(self) -> None:
        """DIM_V3 debe ser exactamente 260."""
        assert DIM_V3 == 260, f"Esperado 260, obtenido {DIM_V3}"

    def test_builder_usa_dim_v3_por_defecto(self) -> None:
        """El builder por defecto debe usar DIM_V3=260."""
        builder = ObservacionBuilderV3()
        assert builder.dim == 260

    def test_output_shape_es_260(self, builder, motor) -> None:
        """construir() debe retornar vector de shape (260,)."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor,
            agente_idx=0,
            vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (
            260,), f"Shape esperado (260,), obtenido {obs.shape}"
        assert obs.dtype == np.float32

    # ------------------------------------------------------------------
    # Bloque base (0:220) — heredado
    # ------------------------------------------------------------------

    def test_bloque_base_mano(self, builder, motor) -> None:
        """Las primeras 52 dims codifican la mano del agente."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        # El agente 0 tiene 13 cartas → 13 unos en [0:52]
        mano_count = int(obs[0:52].sum())
        assert mano_count == 13, f"Esperadas 13 cartas en mano, hay {mano_count}"

    # ------------------------------------------------------------------
    # Bloque enriquecido [220:250]
    # ------------------------------------------------------------------

    def test_cartas_restantes_220_224(self, builder, motor) -> None:
        """[220:224] debe contener cartas restantes por palo (no en mi mano)."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        # Al inicio, todas las cartas están en manos. Las "restantes" son
        # las que no están en mi mano (en manos rivales).
        cartas_restantes = obs[220:224]
        assert np.all(cartas_restantes >= 0.0)
        assert np.all(cartas_restantes <= 13.0)
        for palo in range(4):
            en_mano = sum(1 for c in motor.jugadores[0].mano if c.palo == palo)
            # cartas_restantes = 13 - en_mi_mano (las que están en manos rivales)
            assert cartas_restantes[palo] == pytest.approx(
                float(13 - en_mano), abs=0.01
            ), f"Palo {palo}: restantes={cartas_restantes[palo]}, en_mano={en_mano}"

    def test_cartas_altas_mano_228_232(self, builder, motor) -> None:
        """[228:232] debe contar J/Q/K/A en mi mano por palo."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        altas_mano = obs[228:232]
        for palo in range(4):
            real = sum(
                1 for c in motor.jugadores[0].mano
                if c.palo == palo and c.valor >= 11
            )
            assert altas_mano[palo] == pytest.approx(float(real), abs=0.01), \
                f"Palo {palo}: esperado {real}, obtenido {altas_mano[palo]}"

    def test_bazas_restantes_244(self, builder, motor) -> None:
        """[244] debe ser 13 - baza_actual."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        # Al inicio: baza 1 → 12 bazas restantes
        assert obs[244] == pytest.approx(12.0, abs=0.01)

    def test_puntos_rivales_245_249(self, builder, motor) -> None:
        """[245:249] debe contener puntos de rivales (relativo al agente)."""
        vacios = [set() for _ in range(4)]
        # agente=0: 5pts, rival1=0, rival2=3, rival3=0
        puntos_mano = [5, 0, 3, 0]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=puntos_mano,
            dama_picas_en=None,
        )
        # Orden relativo: rival1(rel=1)=0, rival2(rel=2)=3, rival3(rel=3)=0, agente(rel=0)=5
        expected = [5.0, 0.0, 3.0, 0.0]
        for r in range(4):
            assert obs[245 + r] == pytest.approx(expected[r], abs=0.01), \
                f"Pos {r}: esperado {expected[r]}, obtenido {obs[245+r]}"

    def test_output_values_in_range(self, builder, motor) -> None:
        """Todos los valores deben estar en [0, 26] aproximadamente."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[50, 30, 70, 10],
            puntos_mano_actual=[0, 4, 0, 2],
            dama_picas_en=None,
        )
        assert np.all(obs >= -0.01), "No debe haber valores negativos"
        # Los valores máximos razonables: Q♠=13, puntos=26, cartas=13
        assert np.all(obs <= 26.01), f"Valor máximo excesivo: {obs.max()}"

    # ------------------------------------------------------------------
    # Integración con MotorCorazones
    # ------------------------------------------------------------------

    def test_construir_no_modifica_motor(self, builder, motor) -> None:
        """construir() no debe modificar el estado del motor."""
        mano_antes = [c.id for c in motor.jugadores[0].mano]
        mesa_antes = len(motor.mesa)
        vacios = [set() for _ in range(4)]
        _ = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        mano_despues = [c.id for c in motor.jugadores[0].mano]
        assert mano_antes == mano_despues
        assert len(motor.mesa) == mesa_antes

    def test_construir_desde_motor_v3(self, builder, motor) -> None:
        """construir_desde_motor() retorna vector de 260 dims."""
        obs = builder.construir_desde_motor(motor, 0)
        assert obs.shape == (260,)
        assert obs.dtype == np.float32
        # Debe tener mano del jugador
        mano_count = int(obs[0:52].sum())
        assert mano_count == 13


class TestObservacionBuilderV3EdgeCases:
    """Casos extremos y de borde."""

    @pytest.fixture
    def builder(self) -> ObservacionBuilderV3:
        return ObservacionBuilderV3()

    def test_mano_vacia(self, builder) -> None:
        """Con un motor sin iniciar, no debería crashear."""
        m = MotorCorazones()
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=m, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (260,)
        # Sin mano repartida: 0 cartas en mano
        assert obs[0:52].sum() == 0.0

    def test_todos_los_jugadores_con_puntos(self, builder) -> None:
        """Cuando todos tienen puntos, puntos_rivales debe reflejarlo."""
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        puntos = [10, 5, 8, 3]  # idx0=10, idx1=5, idx2=8, idx3=3
        obs = builder.construir(
            motor=m, agente_idx=1, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=puntos,
            dama_picas_en=None,
        )
        # Agente=1 → rel: r=0=idx1(5), r=1=idx2(8), r=2=idx3(3), r=3=idx0(10)
        assert obs[245] == pytest.approx(5.0)   # agente (rel=0, idx1)
        assert obs[246] == pytest.approx(8.0)   # rel=1 (idx2)
        assert obs[247] == pytest.approx(3.0)   # rel=2 (idx3)
        assert obs[248] == pytest.approx(10.0)  # rel=3 (idx0)
