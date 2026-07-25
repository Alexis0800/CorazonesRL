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


def _calc_rank():
    return CalculadoraRecompensasPartida(RewardConfigPartida(PHI_RANK=True))


class TestPhiRank:
    CASOS = [
        ([0, 0, 0, 0], 0),
        ([10, 50, 60, 90], 0),
        ([10, 50, 60, 90], 3),
        ([99, 1, 1, 1], 0),
        ([30, 30, 30, 30], 2),
    ]

    def test_default_off_potencial_identico(self):
        """Sin PHI_RANK, potencial() = fórmula vieja byte a byte."""
        cfg = RewardConfigPartida()
        assert cfg.PHI_RANK is False
        c = _calc()
        for scores, idx in self.CASOS:
            mi = float(scores[idx])
            otros = [float(s) for i, s in enumerate(scores) if i != idx]
            ventaja = max(-1.0, min(1.0, (sum(otros) / 3.0 - mi) / cfg.PHI_ESCALA))
            assert c.potencial(scores, idx) == cfg.PHI_LAMBDA * ventaja

    def test_phi_rank_cero_en_origen(self):
        assert _calc_rank().potencial([0, 0, 0, 0], 0) == 0.0

    def test_monotonico_bajar_mi_score_no_baja_phi(self):
        c = _calc_rank()
        rivales = [37, 52, 88]
        prev = None
        # Recorro mi score de peor (alto) a mejor (bajo): Φ nunca debe bajar
        for mi in range(120, -1, -1):
            phi = c.potencial([mi] + rivales, 0)
            if prev is not None:
                assert phi >= prev - 1e-12
            prev = phi

    def test_continuidad_un_punto(self):
        """Cambiar 1 punto de cualquier score mueve Φ < λ·0.15 (sin saltos ±0.6)."""
        c = _calc_rank()
        lam = RewardConfigPartida().PHI_LAMBDA
        bases = [[10, 50, 60, 90], [30, 30, 30, 30], [95, 96, 97, 98],
                 [0, 5, 10, 15], [49, 51, 50, 52]]
        for base in bases:
            for j in range(4):
                for delta in (-1, 1):
                    pert = list(base)
                    pert[j] = max(0, pert[j] + delta)
                    for idx in range(4):
                        d = abs(c.potencial(pert, idx) - c.potencial(base, idx))
                        assert d < lam * 0.15

    def test_ordena_por_puesto(self):
        c = _calc_rank()
        scores = [10, 50, 60, 90]
        phis = [c.potencial(scores, i) for i in range(4)]
        # 1º (idx0) > 2º > 3º > 4º (idx3)
        assert phis[0] > phis[1] > phis[2] > phis[3]
        assert phis[0] > phis[3]

    def test_extremos_interpolacion(self):
        c = _calc_rank()
        cfg = RewardConfigPartida()
        lam = cfg.PHI_LAMBDA
        # 1º claro (todos los rivales a más de GAP): r=1 → Φ = λ·R_PRIMERO
        assert abs(c.potencial([0, 50, 60, 90], 0) - lam * cfg.R_PRIMERO) < 1e-9
        # 4º claro: r=4 → Φ = λ·R_CUARTO
        assert abs(c.potencial([90, 0, 10, 20], 0) - lam * cfg.R_CUARTO) < 1e-9

    def test_telescopaje_pbrs_descontado(self):
        """Σ γ^t·F_t = γ^T·Φ(s_T) − Φ(s₀) = −Φ(s₀) porque shaping() usa
        Φ(terminal)=0. Con PHI_RANK activo la propiedad PBRS se mantiene."""
        c = _calc_rank()
        gamma = 0.999
        agente = 0
        trayectoria = [
            [0, 0, 0, 0],
            [5, 10, 8, 3],
            [5, 36, 8, 29],
            [31, 36, 8, 29],
            [31, 62, 8, 55],
            [57, 62, 34, 101],  # terminal
        ]
        total = 0.0
        for t in range(len(trayectoria) - 1):
            terminal = (t == len(trayectoria) - 2)
            f = c.shaping(trayectoria[t], trayectoria[t + 1], agente,
                          gamma=gamma, terminal=terminal)
            total += (gamma ** t) * f
        esperado = -c.potencial(trayectoria[0], agente)  # γ^T·0 − Φ(s₀)
        assert abs(total - esperado) < 1e-9
