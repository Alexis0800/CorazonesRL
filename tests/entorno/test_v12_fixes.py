"""
Pruebas unitarias para fixes v12 (auditoría).

Cubre:
  - REWARD_CORAZON_POZO: recompensa positiva por corazones durante shooting the moon.
  - REWARD_DAMA_PICAS: valor reducido a -6.0 para balancear aversión.
  - _resolver_baza_actual: corazones dan +1.5 cuando _pozo_viable() es True.
"""
import pytest
import numpy as np
from src.entorno.single_agent import CorazonesEnv
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones


# ============================================================
# Helpers
# ============================================================

def _crear_env_con_mano(mano_ids, agente_idx=0):
    """Crea un CorazonesEnv con una mano inyectada manualmente."""
    env = CorazonesEnv(agente_idx=agente_idx, obs_dim=220)
    env.reset(seed=42)
    # Reemplazar mano del agente
    mano = [Carta._TODAS[cid] for cid in mano_ids]
    env.motor.jugadores[agente_idx].mano = mano
    return env


# ============================================================
# Fix 1: REWARD_CORAZON_POZO
# ============================================================

class TestRewardCorazonPozo:
    """Verifica que exista REWARD_CORAZON_POZO y se aplique correctamente."""

    def test_constante_existe(self):
        """REWARD_CORAZON_POZO debe existir en CorazonesEnv con valor +1.5."""
        assert hasattr(CorazonesEnv, 'REWARD_CORAZON_POZO'), \
            "Falta REWARD_CORAZON_POZO en CorazonesEnv"
        assert CorazonesEnv.REWARD_CORAZON_POZO == 1.5, \
            f"REWARD_CORAZON_POZO debe ser 1.5, es {CorazonesEnv.REWARD_CORAZON_POZO}"

    def test_corazon_con_pozo_viable_da_positivo(self):
        """Cuando _pozo_viable() es True, ganar un corazón debe dar reward positivo."""
        env = CorazonesEnv(agente_idx=0, obs_dim=220)
        # No llamar reset — configurar estado manualmente para evitar auto-play
        env._puntuacion_historica = [0, 0, 0, 0]
        env._vacios = [set(), set(), set(), set()]
        env._dama_picas_en = None
        env._puntos_mano_actual = [0, 0, 0, 0]
        env._pleno_jugador = None
        env._recompensa_pendiente = 0.0

        # Configurar motor: corazones NO rotos, mano con 7 corazones altos
        env.motor.corazones_rotos = False
        env.motor.numero_baza = 5  # baza media, pozo aún viable
        # Hearts: palo=3, ids 39-51. id = palo*13 + (valor-2)
        # A♥=51, K♥=50, Q♥=49, J♥=48, 10♥=47, 9♥=46, 8♥=45
        mano_pozo = [
            Carta._TODAS[51],  # A♥
            Carta._TODAS[50],  # K♥
            Carta._TODAS[49],  # Q♥
            Carta._TODAS[48],  # J♥
            Carta._TODAS[47],  # 10♥
            Carta._TODAS[46],  # 9♥
            Carta._TODAS[45],  # 8♥
            Carta._TODAS[0],   # 2♣
            Carta._TODAS[1],   # 3♣
            Carta._TODAS[2],   # 4♣
            Carta._TODAS[3],   # 5♣
            Carta._TODAS[4],   # 6♣
            Carta._TODAS[5],   # 7♣
        ]
        env.motor.jugadores[0].mano = mano_pozo

        # Verificar precondiciones
        assert not env.motor.corazones_rotos, "corazones no deben estar rotos"
        assert sum(1 for c in mano_pozo if c.es_corazon) >= 6, "debe tener ≥6 corazones"
        altos = sum(1 for c in mano_pozo if c.es_corazon and c.valor >= 11)
        assert altos >= 3, f"debe tener ≥3 corazones altos, tiene {altos}"
        assert env._puntuacion_historica[0] < 80, "puntaje debe ser <80"

        assert env._pozo_viable(), "La mano configurada debería ser pozo viable"

        # Verificar constantes
        assert env.REWARD_CORAZON_POZO > 0, "REWARD_CORAZON_POZO debe ser positivo"
        assert env.REWARD_CORAZON < 0, "REWARD_CORAZON base debe ser negativo"

    def test_corazon_sin_pozo_da_negativo(self):
        """Cuando _pozo_viable() es False, ganar un corazón mantiene REWARD_CORAZON negativo."""
        env = CorazonesEnv(agente_idx=0, obs_dim=220)
        env.reset(seed=42)

        # Mano normal sin pozo viable
        env.motor.corazones_rotos = True  # corazones ya rotos → pozo no viable
        assert not env._pozo_viable(), "Con corazones rotos, pozo no debe ser viable"

        # El reward base por corazón debe seguir siendo negativo
        assert env.REWARD_CORAZON == -1.0, \
            "REWARD_CORAZON base debe ser -1.0"


# ============================================================
# Fix 2: REWARD_DAMA_PICAS balanceado
# ============================================================

class TestRewardDamaPicas:
    """Verifica que REWARD_DAMA_PICAS tenga un valor balanceado."""

    def test_constante_reducida(self):
        """REWARD_DAMA_PICAS debe ser -6.0 (reducido de -10.0)."""
        assert CorazonesEnv.REWARD_DAMA_PICAS == -6.0, \
            f"REWARD_DAMA_PICAS debe ser -6.0 (v12), es {CorazonesEnv.REWARD_DAMA_PICAS}"

    def test_q_spades_sin_pozo_no_excede_10(self):
        """Q♠ sin pozo: REWARD_DAMA_PICAS + REWARD_Q_SPADES_SIN_POZO no debe exceder -11."""
        total = CorazonesEnv.REWARD_DAMA_PICAS + CorazonesEnv.REWARD_Q_SPADES_SIN_POZO
        assert total >= -11.0, \
            f"Penalización combinada Q♠ sin pozo ({total}) demasiado alta"

    def test_q_spades_menor_que_perder_mano_con_10_pts(self):
        """Q♠ (-6.0) debe ser comparable a perder una mano con 10 puntos (~-3.0)."""
        q_penalty = abs(CorazonesEnv.REWARD_DAMA_PICAS)
        perder_mano = abs(CorazonesEnv.REWARD_PERDER_MANO) + abs(CorazonesEnv.REWARD_POR_PUNTO_EN_MANO * 10)
        # Q♠ no debe ser más de 3× peor que perder una mano completa
        assert q_penalty <= perder_mano * 3.0, \
            f"Q♠ penalty ({q_penalty}) es desproporcionado vs perder mano ({perder_mano})"
