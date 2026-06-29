"""Tests de la memoria del pase (v13, 332 dims): planos di[228:280] y recibí[280:332]."""
from __future__ import annotations

import numpy as np

from src.dominio.motor import MotorCorazones
from src.entorno.corazones_rllib import CorazonesEnvRLlib
from src.entorno.dimensiones import DIM_V13
from src.entorno.observacion import ObservacionBuilder


def _construir(b, motor, a, **kw):
    return b.construir(
        motor=motor, agente_idx=a, vacios=[set() for _ in range(4)],
        puntuacion_historica=[0, 0, 0, 0], puntos_mano_actual=[0, 0, 0, 0],
        dama_picas_en=None, **kw,
    )


class TestBloqueV13:
    def test_marca_dadas_y_recibidas(self):
        motor = MotorCorazones()
        motor.repartir()
        b = ObservacionBuilder(dim=DIM_V13)
        hand = [c.id for c in motor.jugadores[0].mano]
        recibidas = hand[:3]                                   # en mi mano
        pasadas = [c.id for c in motor.jugadores[1].mano][:3]  # no jugadas

        obs = _construir(b, motor, 0, cartas_pasadas=pasadas, cartas_recibidas=recibidas)
        assert obs.shape == (DIM_V13,)
        for cid in pasadas:
            assert obs[228 + cid] == 1.0
        for cid in recibidas:
            assert obs[280 + cid] == 1.0
        # nada más marcado en los planos
        assert obs[228:280].sum() == 3
        assert obs[280:332].sum() == 3

    def test_recibida_fuera_de_mano_no_se_marca(self):
        motor = MotorCorazones()
        motor.repartir()
        b = ObservacionBuilder(dim=DIM_V13)
        hand = {c.id for c in motor.jugadores[0].mano}
        fuera = next(c.id for c in motor.jugadores[2].mano if c.id not in hand)
        obs = _construir(b, motor, 0, cartas_recibidas=[fuera])
        assert obs[280 + fuera] == 0.0

    def test_dada_ya_jugada_no_se_marca(self):
        motor = MotorCorazones()
        motor.repartir()
        # juega una carta: pasa a estar "jugada" (en la mesa)
        idx = motor.obtener_jugador_actual()
        carta = motor.obtener_jugadas_legales(idx)[0]
        motor.jugar_carta(idx, carta)
        b = ObservacionBuilder(dim=DIM_V13)
        obs = _construir(b, motor, 0, cartas_pasadas=[carta.id])
        assert obs[228 + carta.id] == 0.0


class TestEnvV13:
    def test_pase_popula_memoria(self):
        env = CorazonesEnvRLlib(
            {"obs_dim": DIM_V13, "agente_idx": 0, "con_pase": True})
        obs, _ = env.reset()
        assert obs["obs"].shape == (DIM_V13,)
        # mano 1 = pase izquierda: hacer las 3 sub-decisiones eligiendo cartas legales
        for _ in range(3):
            legal = int(np.argmax(obs["action_mask"]))
            obs, _, _, _, _ = env.step(legal)
        di = obs["obs"][228:280]
        rec = obs["obs"][280:332]
        assert rec.sum() == 3          # 3 recibidas siguen en mano antes de jugar
        assert di.sum() >= 1           # cartas dadas aún en mano del receptor
        env.close()
