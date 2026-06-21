"""
Tests TDD para el sistema de recompensas score-based (v2_ronda).

4 señales:
  1. Puntos capturados por baza: -1 ♡, -13 Q♠
  2. Fin de mano: 26 - mis_puntos  (rango [0, 26])
  3. Shooting moon: +78 (neto +52 tras per-baza)
  4. Posicion: +5 mejor mano, -5 peor mano

Sin distancia. Sin ruido de rivales. Sin señales tácticas.
"""
import pytest
from src.v2_ronda.recompensas import (
    RewardConfigScore,
    CalculadoraRecompensasScore,
)


class TestRewardConfigScore:
    """Verifica que RewardConfigScore tenga solo las 5 constantes."""

    def test_valores_core(self):
        cfg = RewardConfigScore()
        assert cfg.REWARD_CORAZON == -1.0, "Cada ♡ capturado = -1"
        assert cfg.REWARD_DAMA_PICAS == -13.0, "Q♠ = -13"
        assert cfg.REWARD_SHOOTING_MOON == 78.0, "Pozo = +78"

    def test_valores_posicion(self):
        cfg = RewardConfigScore()
        assert cfg.REWARD_MEJOR_MANO == 5.0, "Mejor mano = +5"
        assert cfg.REWARD_PEOR_MANO == -5.0, "Peor mano = -5"

    def test_no_tiene_distancia(self):
        """El config score NO debe tener ESCALA_DISTANCIA."""
        cfg = RewardConfigScore()
        assert not hasattr(cfg, "ESCALA_DISTANCIA"), \
            "Score-based no usa distancia"

    def test_es_inmutable(self):
        cfg = RewardConfigScore()
        with pytest.raises(Exception):
            cfg.REWARD_CORAZON = 999.0


class TestRecompensaBaza:
    """Verifica recompensa por ganar una baza."""

    def setup_method(self):
        from src.dominio.carta import Carta
        self.calc = CalculadoraRecompensasScore()
        self.corazon_2 = next(
            c for c in Carta._TODAS if c.es_corazon and c.valor == 2)
        self.corazon_10 = next(
            c for c in Carta._TODAS if c.es_corazon and c.valor == 10)
        self.dama_picas = next(c for c in Carta._TODAS if c.es_dama_de_picas)
        self.dos_trebol = next(
            c for c in Carta._TODAS if c.palo == 0 and c.valor == 2)

    def test_agente_gana_baza_con_corazon(self):
        """Agente gana baza con 1 corazon → -1."""
        cartas = [self.corazon_2, self.dos_trebol,
                  self.dos_trebol, self.dos_trebol]
        r = self.calc.recompensa_baza(cartas, agente_idx=0, ganador=0)
        assert r == -1.0

    def test_agente_gana_baza_con_dama_picas(self):
        """Agente gana baza con Q♠ → -13."""
        cartas = [self.dama_picas, self.dos_trebol,
                  self.dos_trebol, self.dos_trebol]
        r = self.calc.recompensa_baza(cartas, agente_idx=0, ganador=0)
        assert r == -13.0

    def test_agente_gana_baza_con_corazon_y_q(self):
        """Agente gana baza con ♡ + Q♠ → -14."""
        cartas = [self.corazon_10, self.dama_picas,
                  self.dos_trebol, self.dos_trebol]
        r = self.calc.recompensa_baza(cartas, agente_idx=0, ganador=0)
        assert r == -14.0

    def test_agente_no_gana_baza(self):
        """Agente no gana la baza → 0 (no recibe castigo por cartas ajenas)."""
        cartas = [self.corazon_2, self.dama_picas,
                  self.dos_trebol, self.dos_trebol]
        r = self.calc.recompensa_baza(cartas, agente_idx=0, ganador=1)
        assert r == 0.0

    def test_baza_sin_puntos(self):
        """Baza sin ♡ ni Q♠ → 0."""
        cartas = [self.dos_trebol, self.dos_trebol,
                  self.dos_trebol, self.dos_trebol]
        r = self.calc.recompensa_baza(cartas, agente_idx=0, ganador=0)
        assert r == 0.0

    def test_rival_gana_baza_con_puntos(self):
        """Rival gana baza con puntos → agente recibe 0 (sin ruido)."""
        cartas = [self.corazon_2, self.dama_picas,
                  self.dos_trebol, self.dos_trebol]
        r = self.calc.recompensa_baza(cartas, agente_idx=0, ganador=1)
        assert r == 0.0, "El agente no debe ser penalizado por puntos que toma un rival"


class TestRecompensaFinMano:
    """Verifica recompensa de fin de mano."""

    def setup_method(self):
        self.calc = CalculadoraRecompensasScore()

    def test_mano_perfecta(self):
        """0 puntos → +26."""
        r = self.calc.recompensa_fin_mano(0)
        assert r == 26.0

    def test_tres_corazones(self):
        """3 puntos → +23."""
        r = self.calc.recompensa_fin_mano(3)
        assert r == 23.0

    def test_q_y_dos_corazones(self):
        """15 puntos → +11."""
        r = self.calc.recompensa_fin_mano(15)
        assert r == 11.0

    def test_todos_los_puntos_sin_pozo(self):
        """26 puntos sin pozo → 0."""
        r = self.calc.recompensa_fin_mano(26)
        assert r == 0.0

    def test_rango_siempre_positivo_o_cero(self):
        """El reward de fin de mano nunca es negativo."""
        for pts in range(0, 27):
            r = self.calc.recompensa_fin_mano(pts)
            assert r >= 0.0, f"pts={pts} → reward={r} debería ser ≥ 0"


class TestRecompensaShootingMoon:
    """Verifica recompensa por shooting the moon."""

    def setup_method(self):
        self.calc = CalculadoraRecompensasScore()

    def test_agente_hace_pozo(self):
        r = self.calc.recompensa_shooting_moon(agente_idx=0, pleno_jugador=0)
        assert r == 78.0

    def test_rival_hace_pozo(self):
        r = self.calc.recompensa_shooting_moon(agente_idx=0, pleno_jugador=2)
        assert r == 0.0

    def test_nadie_hace_pozo(self):
        r = self.calc.recompensa_shooting_moon(
            agente_idx=0, pleno_jugador=None)
        assert r == 0.0


class TestRecompensaPosicion:
    """Verifica bonus por posicion en la mano."""

    def setup_method(self):
        self.calc = CalculadoraRecompensasScore()

    def test_mejor_mano(self):
        """Agente con menos puntos de todos → +5."""
        # puntuaciones: [3, 10, 0, 26] → idx 2 es el mejor
        r = self.calc.recompensa_posicion(
            agente_idx=2, puntuaciones_mano=[3, 10, 0, 26])
        assert r == 5.0

    def test_peor_mano(self):
        """Agente con mas puntos de todos → -5."""
        # puntuaciones: [3, 10, 0, 13] → idx 3 es el peor
        r = self.calc.recompensa_posicion(
            agente_idx=3, puntuaciones_mano=[3, 10, 0, 13])
        assert r == -5.0

    def test_mano_intermedia(self):
        """Agente ni mejor ni peor → 0."""
        r = self.calc.recompensa_posicion(
            agente_idx=0, puntuaciones_mano=[3, 10, 0, 26])
        assert r == 0.0

    def test_empate_mejor(self):
        """Empate en mejor puntuacion → ambos reciben bonus."""
        r = self.calc.recompensa_posicion(
            agente_idx=0, puntuaciones_mano=[0, 10, 0, 26])
        assert r == 5.0

    def test_empate_peor(self):
        """Empate en peor puntuacion → ambos reciben penalty."""
        r = self.calc.recompensa_posicion(
            agente_idx=0, puntuaciones_mano=[26, 10, 26, 3])
        assert r == -5.0


class TestNetoPozo:
    """Verifica que el neto del pozo sea +52 (supera mano perfecta de +31)."""

    def setup_method(self):
        from src.dominio.carta import Carta
        self.calc = CalculadoraRecompensasScore()
        self.corazones = [c for c in Carta._TODAS if c.es_corazon]
        self.dama_picas = next(c for c in Carta._TODAS if c.es_dama_de_picas)

    def test_neto_pozo_mayor_que_mano_perfecta(self):
        """Pozo (+52 neto) > mano perfecta (+31 neto)."""
        # Simulamos acumulacion: 13 corazones + Q♠ = 14 cartas con puntos
        # Pero en realidad solo hay 13 corazones, y Q♠ es de picas
        # El pozo: 13 corazones × -1 + Q♠ × -13 = -26 per-baza
        per_baza = 13 * (-1.0) + (-13.0)  # -26
        moon = 78.0
        fin_mano = self.calc.recompensa_fin_mano(
            0)  # todos reciben 26, agente 0 neto
        # Despues de shooting moon, el agente suma 0 a su score historico
        # Fin de mano: mis_puntos efectivos = 0 (porque el pozo los cancela)
        neto_pozo = per_baza + moon + fin_mano  # -26 + 78 + 26 = 78
        # Pero esperamos neto +52... revisemos
        # El pozo hace que los rivales sumen 26, el agente 0
        # mis_puntos en esta mano = 26 (bruto), pero el pozo los convierte en 0 neto
        # reward_fin_mano(0) = 26
        # neto = -26 + 78 + 26 = 78
        # Hmm, esto es +78 neto?
        # En realidad, despues del pozo, el agente tiene 0 puntos historicos adicionales
        # Pero tomo 26 puntos en la mano. El reward fin de mano evalua los puntos TOMADOS
        # no los netos. Entonces mis_puntos = 26, fin_mano = 0.
        fin_mano_real = self.calc.recompensa_fin_mano(26)  # 26 - 26 = 0
        neto_real = per_baza + moon + fin_mano_real  # -26 + 78 + 0 = 52
        assert neto_real == 52.0, f"Neto pozo deberia ser 52, fue {neto_real}"

        # Mano perfecta (0 pts): per_baza=0, fin_mano=26, posicion=+5
        neto_perfecta = 0.0 + 26.0 + 5.0  # 31
        assert neto_pozo > neto_perfecta, \
            f"Pozo neto {neto_real} deberia superar mano perfecta {neto_perfecta}"

    def test_pozo_neto_es_52(self):
        """Verificacion directa: pozo neto = -26 + 78 + 0 = 52."""
        per_baza = -26.0
        moon = 78.0
        fin_mano = self.calc.recompensa_fin_mano(26)  # 26 pts tomados
        assert per_baza + moon + fin_mano == 52.0
