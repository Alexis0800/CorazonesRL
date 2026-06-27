"""
Tests de la recompensa v10 (partida completa): R_terminal + shaping PBRS.

Lo más importante: verificar la propiedad anti-farming del shaping basado en
potencial — con γ=1 y Φ(terminal)=0, la suma del shaping sobre cualquier
trayectoria telescopia a −Φ(s₀), que es 0 al inicio de partida. Es decir:
el agente NO puede acumular reward farmeando señales intermedias.
"""
from __future__ import annotations

from src.entorno.recompensas_partida import (
    CalculadoraRecompensasPartida,
    RewardConfigPartida,
)


def _calc():
    return CalculadoraRecompensasPartida(RewardConfigPartida())


class TestPotencial:
    def test_potencial_inicial_es_cero(self):
        c = _calc()
        assert c.potencial([0, 0, 0, 0], 0) == 0.0

    def test_potencial_positivo_si_voy_por_debajo(self):
        c = _calc()
        # Agente con menos puntos que la media → Φ > 0
        assert c.potencial([0, 30, 30, 30], 0) > 0.0

    def test_potencial_negativo_si_voy_por_encima(self):
        c = _calc()
        assert c.potencial([60, 0, 0, 0], 0) < 0.0

    def test_potencial_acotado(self):
        c = _calc()
        # Diferencia enorme → recortado a λ * 1.0
        val = c.potencial([0, 300, 300, 300], 0)
        assert abs(val) <= RewardConfigPartida().PHI_LAMBDA + 1e-9


class TestRecompensaTerminal:
    def test_primero_recibe_max(self):
        c = _calc()
        # Agente 0 con menos puntos → 1º
        assert c.recompensa_terminal([5, 40, 60, 100], 0) == 1.0

    def test_cuarto_recibe_min(self):
        c = _calc()
        assert c.recompensa_terminal([100, 40, 60, 5], 0) == -1.0

    def test_orden_de_puestos(self):
        c = _calc()
        scores = [10, 20, 30, 40]
        assert c.recompensa_terminal(scores, 0) == 1.0     # 1º
        assert c.recompensa_terminal(scores, 1) == 0.3     # 2º
        assert c.recompensa_terminal(scores, 2) == -0.3    # 3º
        assert c.recompensa_terminal(scores, 3) == -1.0    # 4º

    def test_empate_comparte_promedio(self):
        c = _calc()
        # Dos jugadores empatados en el 1º/2º puesto → promedio de +1.0 y +0.3
        scores = [10, 10, 30, 40]
        esperado = (1.0 + 0.3) / 2
        assert abs(c.recompensa_terminal(scores, 0) - esperado) < 1e-9
        assert abs(c.recompensa_terminal(scores, 1) - esperado) < 1e-9

    def test_puesto(self):
        c = _calc()
        assert c.puesto([10, 20, 30, 40], 0) == 1
        assert c.puesto([10, 20, 30, 40], 3) == 4


class TestPBRSAntiFarming:
    def test_telescopaje_suma_cero_gamma_1(self):
        """Con γ=1 y Φ(terminal)=0, la suma del shaping sobre una trayectoria
        telescopia a −Φ(s₀) = 0. Imposible farmear reward intermedio."""
        c = _calc()
        gamma = 1.0
        agente = 0
        # Trayectoria arbitraria de marcadores acumulados a lo largo de la partida
        trayectoria = [
            [0, 0, 0, 0],
            [5, 10, 8, 3],
            [5, 36, 8, 29],
            [31, 36, 8, 29],
            [31, 62, 8, 55],   # estado final (alguien ≥100? no importa para el test)
        ]
        total = 0.0
        phi_prev = c.potencial(trayectoria[0], agente)
        for i in range(1, len(trayectoria)):
            terminal = (i == len(trayectoria) - 1)
            phi_now = 0.0 if terminal else c.potencial(trayectoria[i], agente)
            total += gamma * phi_now - phi_prev
            phi_prev = phi_now
        # Telescopaje exacto: total = −Φ(s₀) = 0
        assert abs(total) < 1e-9

    def test_shaping_premia_mejorar_posicion(self):
        c = _calc()
        # Mano en la que los rivales reciben puntos y el agente no → mejora relativa
        f = c.shaping([10, 10, 10, 10], [10, 36, 10, 10], 0, gamma=0.999, terminal=False)
        assert f > 0.0
