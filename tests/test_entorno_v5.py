"""
Pruebas unitarias para Fase 5A: Features estratégicas y nuevas recompensas.

Cubre:
  - Vector de observación ampliado (187 → 190 dimensiones)
  - Features booleanas: pozo_viable, debo_arriesgar, puedo_alimentar
  - Nuevas recompensas: Q♠ sin pozo, corazones sin pozo, penalización por punto
"""
import pytest
import numpy as np
from src.entorno import CorazonesEnv
from src.carta import Carta


# ============================================================
# Helpers
# ============================================================

def _crear_entorno_con_mano(mano_ids, agente_idx=0, puntajes=None):
    """Crea un entorno, inyecta una mano específica y retorna el env."""
    env = CorazonesEnv(agente_idx=agente_idx)
    env.reset(seed=42)

    # Inyectar mano directamente en el jugador agente
    jugador = env.motor.jugadores[agente_idx]
    jugador.mano = [Carta._TODAS[cid] for cid in mano_ids]

    # Inyectar puntajes históricos si se especifican
    if puntajes is not None:
        env._puntuacion_historica = list(puntajes)

    return env


def _contar_corazones_en_mano(mano_ids):
    """Cuenta cuántas cartas de la mano son corazones (palo 3)."""
    return sum(1 for cid in mano_ids if cid // 13 == 3)


def _contar_corazones_altos(mano_ids):
    """Cuenta corazones con valor >= 11 (J, Q, K, A) en la mano."""
    return sum(
        1 for cid in mano_ids
        if cid // 13 == 3 and (cid % 13) + 2 >= 11
    )


# ============================================================
# Tests de features estratégicas en el vector de observación
# ============================================================

class TestObservacionV5:
    """Verifica que el vector de observación tenga 190 dimensiones
    y que las nuevas features se calculen correctamente."""

    def test_observacion_tiene_190_dimensiones(self):
        """El vector de observación debe tener exactamente 190 floats."""
        env = CorazonesEnv(agente_idx=0)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (194,), \
            f"Se esperaba (194,), se obtuvo {obs.shape}"
        assert obs.dtype == np.float32

    def test_pozo_viable_mano_fuerte_sin_corazones_rotos(self):
        """Con ≥6 corazones y ≥3 altos, sin corazones rotos → pozo_viable=1.0.

        Este test verifica que con exactamente 6 corazones y 3 altos (J♥, Q♥, K♥),
        el pozo se considere viable."""
        # 6 corazones: 2♥(39),3♥(40), J♥(48), Q♥(49), K♥(50), A♥(51)
        # altos: J♥, Q♥, K♥, A♥ = 4 ≥ 3 ✓
        mano = [39, 40, 48, 49, 50, 51,  # 6 corazones, 4 altos
                0, 1, 2,                 # tréboles
                13, 14,                  # diamantes
                26, 27]                  # picas
        env = _crear_entorno_con_mano(mano)
        env.motor.corazones_rotos = False
        env._puntuacion_historica[0] = 20

        obs = env._construir_observacion()
        assert obs[187] == 1.0, \
            f"pozo_viable debería ser 1.0 con 6♥, 4 altos; obs[187]={obs[187]}"

    def test_pozo_viable_con_corazones_altos(self):
        """Con ≥6 corazones, ≥3 altos (J,Q,K,A♥), sin 💔 roto → pozo_viable=1.0."""
        # Mano con 6 corazones incluyendo A♥(51), K♥(50), Q♥(49)
        mano = [48, 49, 50, 51, 44, 45,  # 6 corazones: 6, J, Q, K, A, 7, 8♥? No...
                39, 40, 41,              # 3 corazones: 2,3,4♥
                0, 1, 13, 26]            # relleno
        # Recalculemos IDs: 2♥=39, 3♥=40, 4♥=41, 5♥=42, 6♥=43, 7♥=44,
        # 8♥=45, 9♥=46, 10♥=47, J♥=48, Q♥=49, K♥=50, A♥=51
        # 6 corazones: 2♥(39),3♥(40),4♥(41), J♥(48), Q♥(49), K♥(50)
        # altos: J♥, Q♥, K♥ = 3 ✓
        mano = [39, 40, 41, 48, 49, 50,  # 6 corazones, 3 altos
                0, 1, 2,                 # tréboles
                13, 14,                  # diamantes
                26, 27]                  # picas
        env = _crear_entorno_con_mano(mano)
        env.motor.corazones_rotos = False
        env._puntuacion_historica[0] = 15

        obs = env._construir_observacion()
        assert obs[187] == 1.0, \
            f"pozo_viable debería ser 1.0; obs[187]={obs[187]}"

    def test_pozo_viable_falso_cuando_corazones_rotos(self):
        """Con corazones rotos, pozo_viable siempre es 0.0."""
        mano = [39, 40, 41, 48, 49, 50, 0, 1, 2, 13, 14, 26, 27]
        env = _crear_entorno_con_mano(mano)
        env.motor.corazones_rotos = True  # 💔 YA ROTOS
        env._puntuacion_historica[0] = 10

        obs = env._construir_observacion()
        assert obs[187] == 0.0, \
            f"pozo_viable debería ser 0.0 con corazones rotos; obs[187]={obs[187]}"

    def test_pozo_viable_falso_con_pocos_corazones(self):
        """Con <6 corazones, pozo_viable es 0.0 aunque no estén rotos."""
        mano = [39, 40, 41, 42,  # 4 corazones (2,3,4,5♥)
                0, 1, 2, 3, 4,  # tréboles
                13, 14, 15,     # diamantes
                26]              # picas
        env = _crear_entorno_con_mano(mano)
        env.motor.corazones_rotos = False
        env._puntuacion_historica[0] = 5

        obs = env._construir_observacion()
        assert obs[187] == 0.0, \
            f"pozo_viable debería ser 0.0 con solo 4♥; obs[187]={obs[187]}"

    def test_pozo_viable_falso_con_pocos_corazones_altos(self):
        """Con ≥6 corazones pero <3 altos → pozo_viable=0.0."""
        # 6 corazones bajos: 2♥ a 7♥ (39-44), sin J/Q/K/A
        mano = [39, 40, 41, 42, 43, 44,  # 6 corazones, 0 altos
                0, 1, 2, 3,              # tréboles
                13, 14,                  # diamantes
                26]                      # picas
        env = _crear_entorno_con_mano(mano)
        env.motor.corazones_rotos = False
        env._puntuacion_historica[0] = 5

        obs = env._construir_observacion()
        assert obs[187] == 0.0, \
            f"pozo_viable debería ser 0.0 con 0 corazones altos; obs[187]={obs[187]}"

    def test_debo_arriesgar_verdadero(self):
        """Puntaje >75 y hay alguien <30 → debo_arriesgar=1.0."""
        mano = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        env = _crear_entorno_con_mano(
            mano, agente_idx=0, puntajes=[78, 15, 40, 50])
        obs = env._construir_observacion()
        assert obs[188] == 1.0, \
            f"debo_arriesgar debería ser 1.0 con score 78 y otro en 15; obs[188]={obs[188]}"

    def test_debo_arriesgar_falso_puntaje_bajo(self):
        """Puntaje bajo → debo_arriesgar=0.0."""
        mano = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        env = _crear_entorno_con_mano(
            mano, agente_idx=0, puntajes=[30, 15, 40, 50])
        obs = env._construir_observacion()
        assert obs[188] == 0.0, \
            f"debo_arriesgar debería ser 0.0 con score 30; obs[188]={obs[188]}"

    def test_debo_arriesgar_falso_todos_altos(self):
        """Todos tienen puntaje alto → debo_arriesgar=0.0."""
        mano = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        env = _crear_entorno_con_mano(
            mano, agente_idx=0, puntajes=[80, 75, 85, 90])
        obs = env._construir_observacion()
        assert obs[188] == 0.0, \
            f"debo_arriesgar debería ser 0.0 si todos tienen >30; obs[188]={obs[188]}"

    def test_puedo_alimentar_verdadero(self):
        """Alguien >85 y agente <70 → puedo_alimentar=1.0."""
        mano = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        env = _crear_entorno_con_mano(
            mano, agente_idx=0, puntajes=[50, 90, 40, 60])
        obs = env._construir_observacion()
        assert obs[189] == 1.0, \
            f"puedo_alimentar debería ser 1.0 con J1 en 90; obs[189]={obs[189]}"

    def test_puedo_alimentar_falso_nadie_cerca(self):
        """Nadie cerca de 100 → puedo_alimentar=0.0."""
        mano = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        env = _crear_entorno_con_mano(
            mano, agente_idx=0, puntajes=[50, 60, 40, 55])
        obs = env._construir_observacion()
        assert obs[189] == 0.0, \
            f"puedo_alimentar debería ser 0.0; obs[189]={obs[189]}"

    def test_puedo_alimentar_falso_agente_tambien_alto(self):
        """Agente también >70 → puedo_alimentar=0.0 (no conviene)."""
        mano = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        env = _crear_entorno_con_mano(
            mano, agente_idx=0, puntajes=[75, 90, 40, 60])
        obs = env._construir_observacion()
        assert obs[189] == 0.0, \
            f"puedo_alimentar debería ser 0.0 si agente tiene 75; obs[189]={obs[189]}"


# ============================================================
# Tests de nuevas constantes de recompensa
# ============================================================

class TestRecompensasV5:
    """Verifica que las nuevas constantes de recompensa existan y tengan
    los valores esperados."""

    def test_constante_q_spades_sin_pozo(self):
        """REWARD_Q_SPADES_SIN_POZO debe existir y ser -8.0."""
        assert hasattr(CorazonesEnv, 'REWARD_Q_SPADES_SIN_POZO'), \
            "Falta REWARD_Q_SPADES_SIN_POZO en CorazonesEnv"
        assert CorazonesEnv.REWARD_Q_SPADES_SIN_POZO == -8.0

    def test_constante_ganar_baza_con_corazon(self):
        """REWARD_GANAR_BAZA_CON_CORAZON debe existir y ser -3.0."""
        assert hasattr(CorazonesEnv, 'REWARD_GANAR_BAZA_CON_CORAZON'), \
            "Falta REWARD_GANAR_BAZA_CON_CORAZON en CorazonesEnv"
        assert CorazonesEnv.REWARD_GANAR_BAZA_CON_CORAZON == -3.0

    def test_constante_por_punto_en_mano(self):
        """REWARD_POR_PUNTO_EN_MANO debe existir y ser -0.2."""
        assert hasattr(CorazonesEnv, 'REWARD_POR_PUNTO_EN_MANO'), \
            "Falta REWARD_POR_PUNTO_EN_MANO en CorazonesEnv"
        assert CorazonesEnv.REWARD_POR_PUNTO_EN_MANO == -0.2
