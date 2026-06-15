"""
Pruebas unitarias para Fase 6: Features all_void (190 → 194 dimensiones).

Cubre:
  - Vector de observación ampliado (190 → 194 dimensiones)
  - Features all_void_tréboles, all_void_diamantes, all_void_corazones,
    all_void_espadas (índices 190-193)
  - Verificación de que all_void detecta correctamente cuando los 3
    rivales están vacíos en un palo específico
"""
import pytest
import numpy as np
from src.entorno.single_agent import CorazonesEnv
from src.dominio.carta import Carta


# ============================================================
# Helpers
# ============================================================

def _crear_entorno_con_vacios(agente_idx=0, vacios=None):
    """Crea un entorno e inyecta vacíos específicos para los rivales.

    Args:
        agente_idx: Índice del agente (0 por defecto).
        vacios: Lista de 4 sets, uno por jugador, con palos donde son void.

    Returns:
        Entorno CorazonesEnv con vacíos inyectados.
    """
    env = CorazonesEnv(agente_idx=agente_idx)
    env.reset(seed=42)

    if vacios is not None:
        env._vacios = [set(s) for s in vacios]

    return env


# ============================================================
# Tests de features all_void en el vector de observación
# ============================================================

class TestObservacionV6:
    """Verifica que el vector de observación tenga 194 dimensiones
    y que las features all_void se calculen correctamente."""

    def test_observacion_tiene_194_dimensiones(self):
        """El vector de observación debe tener exactamente 194 floats."""
        env = CorazonesEnv(agente_idx=0)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (194,), \
            f"Se esperaba (194,), se obtuvo {obs.shape}"
        assert obs.dtype == np.float32

    def test_all_void_treboles_todos_vacios(self):
        """Cuando los 3 rivales son void en tréboles → all_void_tréboles=1.0.

        Setup:
          - Agente (idx=0): sin voids relevantes
          - Rival 1 (idx=1): void en tréboles (palo 0)
          - Rival 2 (idx=2): void en tréboles (palo 0)
          - Rival 3 (idx=3): void en tréboles (palo 0)
        """
        vacios = [
            set(),        # agente: sin voids
            {0},          # rival 1: void en tréboles
            {0},          # rival 2: void en tréboles
            {0},          # rival 3: void en tréboles
        ]
        env = _crear_entorno_con_vacios(agente_idx=0, vacios=vacios)

        obs = env._construir_observacion()
        assert obs[190] == 1.0, \
            f"all_void_tréboles debería ser 1.0; obs[190]={obs[190]}"

    def test_all_void_treboles_no_todos_vacios(self):
        """Si al menos 1 rival NO es void → all_void_tréboles=0.0.

        Setup:
          - Agente (idx=0): sin voids relevantes
          - Rival 1 (idx=1): void en tréboles
          - Rival 2 (idx=2): NO void en tréboles → quiebra la condición
          - Rival 3 (idx=3): void en tréboles
        """
        vacios = [
            set(),
            {0},          # rival 1: void
            set(),        # rival 2: NO void
            {0},          # rival 3: void
        ]
        env = _crear_entorno_con_vacios(agente_idx=0, vacios=vacios)

        obs = env._construir_observacion()
        assert obs[190] == 0.0, \
            f"all_void_tréboles debería ser 0.0 (rival 2 no es void); obs[190]={obs[190]}"

    def test_all_void_espadas_todos_vacios(self):
        """Cuando los 3 rivales son void en espadas → all_void_espadas=1.0.

        Este es el caso más importante: el modelo debe saber que liderar
        espadas cuando todos son void es un error (se come la baza).

        Palo 2 = Picas (Espadas) → obs[192].
        """
        vacios = [
            set(),
            {2},          # rival 1: void en espadas (palo 2)
            {2},          # rival 2: void en espadas
            {2},          # rival 3: void en espadas
        ]
        env = _crear_entorno_con_vacios(agente_idx=0, vacios=vacios)

        obs = env._construir_observacion()
        # Palo 2 = Picas/Espadas → obs[192]
        assert obs[192] == 1.0, \
            f"all_void_espadas debería ser 1.0; obs[192]={obs[192]}"

    def test_all_void_corazones_todos_vacios(self):
        """Cuando los 3 rivales son void en corazones → all_void_corazones=1.0.

        Palo 3 = Corazones → obs[193].
        """
        vacios = [
            set(),
            {3},          # rival 1: void en corazones (palo 3)
            {3},          # rival 2: void en corazones
            {3},          # rival 3: void en corazones
        ]
        env = _crear_entorno_con_vacios(agente_idx=0, vacios=vacios)

        obs = env._construir_observacion()
        # Palo 3 = Corazones → obs[193]
        assert obs[193] == 1.0, \
            f"all_void_corazones debería ser 1.0; obs[193]={obs[193]}"

    def test_all_void_multiples_palos(self):
        """Cuando todos son void en tréboles Y diamantes,
        ambas features deben ser 1.0. Las otras 0.0."""
        vacios = [
            set(),
            {0, 1},       # rival 1: void en tréboles y diamantes
            {0, 1},       # rival 2: void en tréboles y diamantes
            {0, 1},       # rival 3: void en tréboles y diamantes
        ]
        env = _crear_entorno_con_vacios(agente_idx=0, vacios=vacios)

        obs = env._construir_observacion()
        assert obs[190] == 1.0, f"all_void_tréboles debería ser 1.0; obs[190]={obs[190]}"
        assert obs[191] == 1.0, f"all_void_diamantes debería ser 1.0; obs[191]={obs[191]}"
        assert obs[192] == 0.0, f"all_void_espadas debería ser 0.0; obs[192]={obs[192]}"
        assert obs[193] == 0.0, f"all_void_corazones debería ser 0.0; obs[193]={obs[193]}"

    def test_ningun_all_void_al_inicio(self):
        """Al inicio del juego, sin bazas jugadas, ningún all_void debe ser 1.0."""
        env = CorazonesEnv(agente_idx=0)
        obs, _ = env.reset(seed=42)

        assert obs[190] == 0.0, \
            f"all_void_tréboles debería ser 0.0 al inicio; obs[190]={obs[190]}"
        assert obs[191] == 0.0, \
            f"all_void_diamantes debería ser 0.0 al inicio; obs[191]={obs[191]}"
        assert obs[192] == 0.0, \
            f"all_void_corazones debería ser 0.0 al inicio; obs[192]={obs[192]}"
        assert obs[193] == 0.0, \
            f"all_void_espadas debería ser 0.0 al inicio; obs[193]={obs[193]}"
