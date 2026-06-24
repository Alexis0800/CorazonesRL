"""Tests para el módulo de dimensiones de v3.1 (228 dims)."""

import pytest
from src.v3_1.dimensiones import DIM_V3_1, DIMS_VALIDAS


class TestDimensionesV31:
    """Validación de constantes dimensionales para v3.1."""

    def test_dim_v3_1_es_228(self) -> None:
        """DIM_V3_1 debe ser exactamente 228."""
        assert DIM_V3_1 == 228, f"Esperado 228, obtenido {DIM_V3_1}"

    def test_dim_v3_1_esta_en_dims_validas(self) -> None:
        """DIM_V3_1 debe estar en DIMS_VALIDAS."""
        assert DIM_V3_1 in DIMS_VALIDAS, (
            f"DIM_V3_1={DIM_V3_1} no está en DIMS_VALIDAS={DIMS_VALIDAS}"
        )

    def test_dim_v3_1_es_menor_que_v3(self) -> None:
        """DIM_V3_1 debe ser menor que DIM_V3=265 (reducción de dims)."""
        from src.v3.observacion import DIM_V3
        assert DIM_V3_1 < DIM_V3, (
            f"DIM_V3_1={DIM_V3_1} debería ser < DIM_V3={DIM_V3}"
        )

    def test_dim_v3_1_es_mayor_que_v10(self) -> None:
        """DIM_V3_1 debe ser mayor que DIM_V10=220 (tiene features adicionales)."""
        from src.entorno.dimensiones import DIM_V10
        assert DIM_V3_1 > DIM_V10, (
            f"DIM_V3_1={DIM_V3_1} debería ser > DIM_V10={DIM_V10}"
        )
