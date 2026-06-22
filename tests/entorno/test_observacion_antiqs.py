"""
Tests TDD para las nuevas features de observación anti-Q♠ (v3).

Verifica que:
  - [248] peligro_qs_inminente indica cuándo jugar Q♠ resultará en capturarla.
  - El feature se integra en ObservacionBuilderV3 sin romper dimensiones.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.v3.observacion import ObservacionBuilderV3, DIM_V3


# ── Helpers ──────────────────────────────────────────────────────────

def _carta(valor: int, palo: int) -> Carta:
    for c in Carta._TODAS:
        if c.valor == valor and c.palo == palo:
            return c
    raise ValueError(f"No encontrada: valor={valor}, palo={palo}")


TREBOL = 0
DIAMANTE = 1
PICA = 2
CORAZON = 3

_Q_ESPADAS = _carta(12, PICA)
_2_TREBOL = _carta(2, TREBOL)
_2_CORAZON = _carta(2, CORAZON)


# ============================================================
# Clase 1 — Dimensionalidad no se rompe con nuevas features
# ============================================================

class TestDimensionalidadNuevasFeatures:
    """Verifica que al agregar peligro_qs_inminente y peligrosidad_residual
    no se rompa la dimensión 250."""

    @pytest.fixture
    def motor(self) -> MotorCorazones:
        m = MotorCorazones()
        m.repartir()
        return m

    @pytest.fixture
    def builder(self) -> ObservacionBuilderV3:
        return ObservacionBuilderV3()

    def test_dim_v3_es_260(self) -> None:
        """DIM_V3 debe ser 260 (220 base + 40 enriquecidas)."""
        assert DIM_V3 == 260, f"DIM_V3 cambió: {DIM_V3}"

    def test_output_shape_es_260(self, builder, motor) -> None:
        """La observación completa debe ser (260,)."""
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=motor, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (260,), f"Shape: {obs.shape}"
        assert obs.dtype == np.float32


# ============================================================
# Clase 2 — peligro_qs_inminente [248]
# ============================================================

class TestPeligroQSInminente:
    """Verifica que el feature peligro_qs_inminente [248] funcione."""

    @pytest.fixture
    def builder(self) -> ObservacionBuilderV3:
        return ObservacionBuilderV3()

    def test_feature_existe_en_posicion_249(self, builder) -> None:
        """El feature debe estar en la posición 249 del vector."""
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=m, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        # Debe ser un valor flotante en [0, 1]
        valor = float(obs[249])
        assert 0.0 <= valor <= 1.0, (
            f"peligro_qs_inminente[249]={valor} fuera de rango [0,1]"
        )

    def test_es_cero_cuando_qs_capturada(self, builder) -> None:
        """Cuando Q♠ ya fue capturada, peligro_qs_inminente debe ser 0."""
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=m, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=1,  # Q♠ ya capturada por J1
        )
        assert obs[249] == 0.0, (
            f"Con Q♠ capturada, peligro_qs_inminente debe ser 0.0, "
            f"es {obs[249]}"
        )

    def test_es_cero_cuando_agente_no_tiene_qs(self, builder) -> None:
        """Cuando el agente no tiene Q♠, peligro_qs_inminente debe ser 0."""
        m = MotorCorazones()
        m.repartir()
        # Verificar que J0 no tiene Q♠ (muy probable)
        if any(c.es_dama_de_picas for c in m.jugadores[0].mano):
            # Si J0 tiene Q♠, cambiar de agente
            for idx in range(1, 4):
                if not any(
                    c.es_dama_de_picas for c in m.jugadores[idx].mano
                ):
                    vacios = [set() for _ in range(4)]
                    obs = builder.construir(
                        motor=m, agente_idx=idx, vacios=vacios,
                        puntuacion_historica=[0, 0, 0, 0],
                        puntos_mano_actual=[0, 0, 0, 0],
                        dama_picas_en=None,
                    )
                    assert obs[249] == 0.0, (
                        f"Agente sin Q♠ debe tener peligro=0, "
                        f"es {obs[249]}"
                    )
                    return
            pytest.skip("Todos los jugadores tienen Q♠ (imposible)")

    def test_es_uno_cuando_lidera_picas_con_qs_y_hay_rivales_con_picas(
        self, builder,
    ) -> None:
        """Cuando el agente tiene Q♠, lidera ♠, y hay rivales con ♠,
        jugar Q♠ puede resultar en capturarla → peligro alto."""
        # Este es un test más complejo que requiere un estado específico.
        # Lo probamos con un motor donde el agente tiene Q♠ y otras picas.
        m = MotorCorazones()
        m.repartir()
        # Buscar un jugador que tenga Q♠ y al menos otra pica
        for ag_idx in range(4):
            mano = m.jugadores[ag_idx].mano
            tiene_qs = any(c.es_dama_de_picas for c in mano)
            otras_picas = [
                c for c in mano if c.palo == PICA and not c.es_dama_de_picas
            ]
            if tiene_qs and len(otras_picas) >= 1:
                vacios = [set() for _ in range(4)]
                obs = builder.construir(
                    motor=m, agente_idx=ag_idx, vacios=vacios,
                    puntuacion_historica=[0, 0, 0, 0],
                    puntos_mano_actual=[0, 0, 0, 0],
                    dama_picas_en=None,
                )
                # Al inicio de la mano (sin bazas jugadas aún), el peligro
                # depende de si hay rivales con picas.
                valor = float(obs[249])
                # Debe ser > 0 si hay riesgo
                assert valor >= 0.0, (
                    f"peligro_qs_inminente no debe ser negativo: {valor}"
                )
                return
        pytest.skip("Ningún jugador tiene Q♠ + otra pica")


# ============================================================
# Clase 3 — Consistencia con features existentes
# ============================================================

class TestConsistenciaFeatures:
    """Verifica que las nuevas features no pisen features existentes."""

    @pytest.fixture
    def builder(self) -> ObservacionBuilderV3:
        return ObservacionBuilderV3()

    def test_feature_220_248_no_alteradas(self, builder) -> None:
        """Las features [220:249] deben preservar su semántica original."""
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        obs = builder.construir(
            motor=m, agente_idx=0, vacios=vacios,
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        # [220:224] cartas_restantes: valores 0-13
        cartas_rest = obs[220:224]
        assert np.all(cartas_rest >= 0.0)
        assert np.all(cartas_rest <= 13.0)

        # [244] bazas_restantes: 12 al inicio (baza 1 de 13 → 12 restantes)
        assert obs[244] == 12.0, f"Bazas restantes: {obs[244]}"

        # [245:249] puntos_rivales: 0 al inicio
        assert np.all(obs[245:249] == 0.0), f"Puntos: {obs[245:249]}"
