"""
Tests TDD para la observación 220-dim (Fase C).

Verifica que ObservacionBuilder soporte completamente el vector de 220
dimensiones, absorbiendo la lógica que estaba duplicada en
CorazonesEnv._construir_bloque_v9().
"""
import pytest
import numpy as np
from src.entorno.observacion import ObservacionBuilder
from src.dominio.motor import MotorCorazones


class TestObservacion220Dim:
    """Verifica que ObservacionBuilder soporta 220 dimensiones."""

    def test_dim_194_funciona(self):
        """Builder con dim=194 no debe tirar error."""
        builder = ObservacionBuilder(dim=194)
        motor = MotorCorazones()
        motor.repartir()
        obs = builder.construir(
            motor, 0,
            vacios=[set(), set(), set(), set()],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (194,)
        assert obs.dtype == np.float32

    def test_dim_220_funciona(self):
        """Builder con dim=220 debe producir vector de 220."""
        builder = ObservacionBuilder(dim=220)
        motor = MotorCorazones()
        motor.repartir()
        obs = builder.construir(
            motor, 0,
            vacios=[set(), set(), set(), set()],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (220,)
        assert obs.dtype == np.float32

    def test_dim_190_retrocompatibilidad(self):
        """Builder con dim=190 debe funcionar (v5)."""
        builder = ObservacionBuilder(dim=190)
        motor = MotorCorazones()
        motor.repartir()
        obs = builder.construir(
            motor, 0,
            vacios=[set(), set(), set(), set()],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (190,)

    def test_bloque_v9_lleno_con_220(self):
        """Con dim=220, los features [194:220] deben tener valores calculados."""
        builder = ObservacionBuilder(dim=220)
        motor = MotorCorazones()
        motor.repartir()

        # Simular estado para poblar bien el bloque v9
        obs = builder.construir(
            motor, 0,
            vacios=[set(), set(), set(), set()],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        # [194] baza_numero: debe ser > 0 (al menos la baza 1)
        assert obs[194] > 0.0, f"baza_numero debe > 0, got {obs[194]}"
        # [219] palo_salida: 0.0 si no hay, o algo entre 0 y 1
        assert 0.0 <= obs[219] <= 1.0, f"palo_salida={obs[219]} fuera de rango"

    def test_bloque_v9_no_afecta_194(self):
        """Con dim=194, los features extra no deben existir."""
        builder = ObservacionBuilder(dim=194)
        motor = MotorCorazones()
        motor.repartir()
        obs = builder.construir(
            motor, 0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        assert obs.shape == (194,)

    def test_prob_q_picas_yo_la_tengo(self):
        """Si tengo Q♠ en mano, mi posición relativa debe ser 1.0."""
        builder = ObservacionBuilder(dim=220)
        motor = MotorCorazones()
        motor.repartir()
        # Buscar Q♠ en mano del agente
        agente = 0
        obs = builder.construir(
            motor, agente,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )
        # Si tengo Q♠, prob[0] debe ser 1.0
        tengo_q = any(c.es_dama_de_picas for c in motor.jugadores[agente].mano)
        if tengo_q:
            assert obs[207] == pytest.approx(
                1.0), f"prob[0] debe ser 1.0, got {obs[207]}"
            assert obs[208] == 0.0
            assert obs[209] == 0.0
            assert obs[210] == 0.0

    def test_construir_desde_motor_ignora_bloque_v9(self):
        """construir_desde_motor es mínimo (solo cartas visibles)."""
        builder = ObservacionBuilder(dim=220)
        motor = MotorCorazones()
        motor.repartir()
        obs = builder.construir_desde_motor(motor, 0)
        # Debe tener shape correcto pero features avanzadas en 0
        assert obs.shape == (220,)
        # features estratégicas deben ser 0
        assert obs[187] == 0.0, "pozo_viable debe ser 0 en construir_desde_motor"
        assert obs[188] == 0.0
        assert obs[189] == 0.0
        # bloque v9 también en 0
        assert obs[194] == 0.0, "baza_numero debe ser 0 en construir_desde_motor"
