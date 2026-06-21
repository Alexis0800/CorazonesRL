"""
Tests TDD para el sistema de recompensas minimal (v13_minimal).

Solo 3 señales:
  - Puntos capturados (-1 por corazon, -13 por Q♠)
  - Reward de distancia (basado en cambio de ventaja ponderado)
  - Shooting moon (+26)

Sin señales tácticas densas.
"""
import pytest
from src.entorno.recompensas_minimal import (
    RewardConfigMinimal,
    CalculadoraRecompensasMinimal,
    calcular_reward_distancia,
)


class TestRewardConfigMinimal:
    """Verifica que RewardConfigMinimal tenga solo las señales esenciales."""

    def test_valores_core(self):
        cfg = RewardConfigMinimal()
        assert cfg.REWARD_CORAZON == -1.0, "Cada corazon = -1 punto"
        assert cfg.REWARD_DAMA_PICAS == -13.0, "Q♠ = -13 puntos"
        assert cfg.REWARD_SHOOTING_MOON == 26.0, "Moon base = +26"
        assert cfg.ESCALA_DISTANCIA == 5.0, "ESCALA default = 5"

    def test_no_tiene_recompensas_tacticas(self):
        """El config minimal NO debe tener señales tácticas."""
        cfg = RewardConfigMinimal()
        attrs_no_permitidos = [
            "PENALTY_LIDERAR_PICA",
            "REWARD_QUEMAR_MAXIMA",
            "REWARD_DUMP_Q",
            "REWARD_DESCARTAR_CORAZON_SEGURO",
            "PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE",
            "REWARD_NO_GANAR_BAZA_CON_PUNTOS",
            "REWARD_LIDERAR_Q_DUMP",
        ]
        for attr in attrs_no_permitidos:
            assert not hasattr(
                cfg, attr), f"El config minimal no debe tener {attr}"

    def test_es_inmutable(self):
        cfg = RewardConfigMinimal()
        with pytest.raises(Exception):
            cfg.REWARD_CORAZON = 999.0


class TestRewardDistancia:
    """Verifica la fórmula de distancia ponderada."""

    def test_formula_basica(self):
        """Caso del ejemplo: A recibe 5 corazones, C le pasa a B."""
        # Jugador A: antes 0, despues 5
        reward = calcular_reward_distancia(
            puntuacion_antes=[0, 5, 6, 15],
            puntuacion_despues=[5, 11, 8, 28],
            agente_idx=0,
            escala=5.0,
        )
        # Verificado manualmente: ~1.2
        assert reward == pytest.approx(1.2, abs=0.1)

    def test_pasar_rival_cercano_recompensa_alto(self):
        """C: estaba a 1 punto de B, lo pasa → reward alto."""
        reward = calcular_reward_distancia(
            puntuacion_antes=[0, 5, 6, 15],
            puntuacion_despues=[5, 11, 8, 28],
            agente_idx=2,  # C
            escala=5.0,
        )
        # C pasó a B (estaba a 1 punto) → ~17.6
        assert reward > 10.0, f"C deberia recibir recompensa alta, obtuvo {reward}"

    def test_recibir_q_espadas_castigo_fuerte(self):
        """D: recibe Q♠ (13 pts), reward distancia negativo."""
        reward = calcular_reward_distancia(
            puntuacion_antes=[0, 5, 6, 15],
            puntuacion_despues=[5, 11, 8, 28],
            agente_idx=3,  # D
            escala=5.0,
        )
        assert reward < 0, f"D deberia ser castigado, obtuvo {reward}"

    def test_sin_cambios_reward_cero(self):
        """Si nadie cambia posicion, reward = 0."""
        reward = calcular_reward_distancia(
            puntuacion_antes=[10, 20, 30, 40],
            puntuacion_despues=[10, 20, 30, 40],
            agente_idx=0,
            escala=5.0,
        )
        assert reward == pytest.approx(0.0, abs=0.01)

    def test_escala_afecta_proporcionalmente(self):
        """Duplicar escala duplica el reward."""
        r1 = calcular_reward_distancia(
            [0, 5, 6, 15], [5, 11, 8, 28], 2, escala=5.0)
        r2 = calcular_reward_distancia(
            [0, 5, 6, 15], [5, 11, 8, 28], 2, escala=10.0)
        assert r2 == pytest.approx(r1 * 2, abs=0.1)


class TestCalculadoraRecompensasMinimal:
    """Verifica la calculadora minimal completa."""

    def test_baza_ganada_solo_puntos(self):
        """Ganar baza con corazones = -1 por cada uno, sin señales extra."""
        from src.dominio.carta import Carta
        calc = CalculadoraRecompensasMinimal()
        corazon = next(
            c for c in Carta._TODAS if c.es_corazon and c.valor == 5)
        no_puntos = next(
            c for c in Carta._TODAS if c.puntos == 0 and c.palo == 0)

        # Baza con 2 corazones
        cartas = [corazon, no_puntos, no_puntos, corazon]
        r = calc.recompensa_baza(cartas, agente_idx=0, ganador=0)
        assert r == -2.0, f"2 corazones = -2, obtuvo {r}"

    def test_baza_evitada_sin_recompensa(self):
        """En modo minimal, evitar baza no da reward (solo importa el resultado final)."""
        from src.dominio.carta import Carta
        calc = CalculadoraRecompensasMinimal()
        corazon = next(c for c in Carta._TODAS if c.es_corazon)
        no_puntos = next(
            c for c in Carta._TODAS if c.puntos == 0 and c.palo == 0)

        cartas = [corazon, no_puntos, no_puntos, no_puntos]
        r = calc.recompensa_baza(cartas, agente_idx=0, ganador=1)
        assert r == 0.0, "Evitar baza no da reward en minimal"

    def test_shooting_moon(self):
        """Moon da +26 base, la distancia se suma despues."""
        calc = CalculadoraRecompensasMinimal()
        r = calc.recompensa_shooting_moon(agente_idx=0, pleno_jugador=0)
        assert r == 26.0

    def test_shooting_moon_rival_no_recompensa(self):
        """Si otro hace moon, el agente no recibe +26."""
        calc = CalculadoraRecompensasMinimal()
        r = calc.recompensa_shooting_moon(agente_idx=0, pleno_jugador=1)
        assert r == 0.0
