"""
Tests TDD para la consolidación de recompensas v12 (Fase A + B).

Verifica que:
  - RewardConfig es el SSOT con valores v12 (antes duplicados en CorazonesEnv).
  - CalculadoraRecompensas cubre TODAS las señales que usa el entorno.
  - Los valores coinciden exactamente con los que estaban en CorazonesEnv.
  - El phase-gating (BAZA_TARDIA) es consistente.
  - Los nuevos métodos agregados producen las recompensas correctas.
"""
import pytest
from src.entorno.recompensas import RewardConfig, CalculadoraRecompensas
from src.dominio.carta import Carta


# ============================================================
# Helpers
# ============================================================

def _carta(valor: int, palo: int) -> Carta:
    for c in Carta._TODAS:
        if c.valor == valor and c.palo == palo:
            return c
    raise ValueError(f"No encontrada: valor={valor}, palo={palo}")


TREBOL = 0
DIAMANTE = 1
PICA = 2
CORAZON = 3

_Q_ESPADAS = _carta(12, 2)
_A_CORAZON = _carta(14, 3)
_2_CORAZON = _carta(2, 3)
_2_TREBOL = _carta(2, 0)
_K_ESPADAS = _carta(13, 2)
_A_ESPADAS = _carta(14, 2)
_3_DIAMANTE = _carta(3, 1)
_5_PICAS = _carta(5, 2)
_10_PICAS = _carta(10, 2)


# ============================================================
# Clase 1 — RewardConfig SSOT (valores v12)
# ============================================================

class TestRewardConfigV12SSOT:
    """Verifica que RewardConfig tenga los valores v12 correctos.

    Estos son los valores que ESTABAN en CorazonesEnv (v12) y que
    ahora son el SSOT en RewardConfig.
    """

    def test_valores_core_sin_cambio(self):
        """Recompensas core que no cambiaron entre versiones."""
        cfg = RewardConfig()
        assert cfg.REWARD_CORAZON == -1.0
        assert cfg.REWARD_SHOOTING_MOON == 50.0
        assert cfg.REWARD_PRIMERO == 500.0
        assert cfg.REWARD_SEGUNDO == 200.0
        assert cfg.REWARD_TERCERO == -200.0
        assert cfg.REWARD_CUARTO == -500.0

    def test_valores_v12_magnitudes_reducidas(self):
        """v12 redujo magnitudes para evitar reward hacking."""
        cfg = RewardConfig()
        assert cfg.REWARD_DAMA_PICAS == -6.0, "v12: -10.0 → -6.0"
        assert cfg.REWARD_Q_SPADES_SIN_POZO == -5.0, "v12: -10.0 → -5.0"
        assert cfg.REWARD_GANAR_BAZA_CON_CORAZON == -2.0, "v12: -3.0 → -2.0"
        assert cfg.REWARD_DESCARTAR_DAMA_SEGURO == 2.0, "v12: 5.0 → 2.0"
        assert cfg.REWARD_NO_GANAR_BAZA_CON_PUNTOS == 0.3, "v12: 1.5 → 0.3"

    def test_valores_v12_tacticos(self):
        """Recompensas tácticas v12."""
        cfg = RewardConfig()
        assert cfg.PENALTY_LIDERAR_PICA_CON_Q_ACTIVA == -2.0
        assert cfg.REWARD_QUEMAR_MAXIMA_PALO_SEGURO == 1.0
        assert cfg.PENALTY_LIDERAR_Q_EQUIVOCADO == -6.0
        assert cfg.REWARD_DUMP_Q_SIGUIENDO_PICAS == 1.0
        assert cfg.PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE == -2.0
        assert cfg.REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA == 1.0
        assert cfg.REWARD_QUEMAR_MAXIMA_FORZADA == 0.5
        assert cfg.REWARD_QUEMAR_ALTA_SIGUIENDO_PALO == 1.0

    def test_valores_v12_fase_b(self):
        """Recompensas agregadas en v10 Fase B."""
        cfg = RewardConfig()
        assert cfg.PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD == -1.0
        assert cfg.REWARD_LIDERAR_Q_DUMP_SEGURO == 2.0
        assert cfg.REWARD_DESCARTAR_CORAZON_BAJO_ROTO == 0.5

    def test_umbrales_v12(self):
        """Umbrales de phase-gating v12."""
        cfg = RewardConfig()
        assert cfg.BAZA_TARDIA == 7, "v12: 9 → 7"
        assert cfg.PUNTUACION_MAXIMA == 100.0
        assert cfg.SCORE_RIVAL_CERCA == 85

    def test_reward_config_es_inmutable(self):
        """RewardConfig es frozen dataclass → no se puede modificar."""
        cfg = RewardConfig()
        with pytest.raises(Exception):
            cfg.REWARD_CORAZON = 999.0


# ============================================================
# Clase 2 — CalculadoraRecompensas: baza ganada
# ============================================================

class TestCalculadoraBazaGanada:
    """Verifica recompensa_baza_ganada con valores v12."""

    def test_ganar_baza_con_corazon_sin_pozo(self):
        calc = CalculadoraRecompensas()
        cartas = [_2_CORAZON, _3_DIAMANTE, _2_TREBOL, _5_PICAS]
        r = calc.recompensa_baza_ganada(cartas, 0, 0, pozo_viable=False)
        assert r == -1.0 + -2.0, f"corazón + penalización = -3.0, got {r}"

    def test_ganar_baza_con_q_spades_sin_pozo(self):
        calc = CalculadoraRecompensas()
        cartas = [_Q_ESPADAS, _3_DIAMANTE, _2_TREBOL, _5_PICAS]
        r = calc.recompensa_baza_ganada(cartas, 0, 0, pozo_viable=False)
        assert r == -6.0 + -5.0, f"Q♠ + penalización = -11.0, got {r}"

    def test_ganar_baza_con_corazon_en_pozo(self):
        calc = CalculadoraRecompensas()
        cartas = [_2_CORAZON, _3_DIAMANTE, _2_TREBOL, _5_PICAS]
        # Con pozo_viable=True y en_modo_pozo=False
        r = calc.recompensa_baza_ganada(cartas, 0, 0, pozo_viable=True)
        assert r == 1.5, f"corazón pozo = +1.5, got {r}"

    def test_no_ganador_no_recibe_recompensa(self):
        calc = CalculadoraRecompensas()
        cartas = [_2_CORAZON, _3_DIAMANTE, _2_TREBOL, _5_PICAS]
        r = calc.recompensa_baza_ganada(cartas, 0, 1, pozo_viable=False)
        assert r == 0.0, "Agente no ganó la baza"

    def test_baza_sin_puntos_baza_tardia(self):
        calc = CalculadoraRecompensas()
        cartas = [_2_TREBOL, _3_DIAMANTE, _5_PICAS, _carta(4, TREBOL)]
        # baza ≥7 con 0 puntos
        r = calc.recompensa_baza_ganada(
            cartas, 0, 0, pozo_viable=False, numero_baza=9)
        assert r == -0.1, f"ganar baza sin puntos en tardía = -0.1, got {r}"

    def test_baza_sin_puntos_baza_temprana(self):
        calc = CalculadoraRecompensas()
        cartas = [_2_TREBOL, _3_DIAMANTE, _5_PICAS, _carta(4, TREBOL)]
        r = calc.recompensa_baza_ganada(
            cartas, 0, 0, pozo_viable=False, numero_baza=2)
        assert r == 0.0, "Sin penalización en baza temprana"


# ============================================================
# Clase 3 — CalculadoraRecompensas: baza evitada
# ============================================================

class TestCalculadoraBazaEvitada:
    """Verifica recompensa_baza_evitada con valores v12."""

    def test_evitar_baza_con_puntos_tardia(self):
        calc = CalculadoraRecompensas()
        cartas = [_2_CORAZON, _3_DIAMANTE, _2_TREBOL, _5_PICAS]
        r = calc.recompensa_baza_evitada(cartas, 0, 1, numero_baza=9)
        # puntos_baza=1, reward = 0.3 * min(1, 3) = 0.3
        assert r == pytest.approx(0.3), f"Esperado 0.3, got {r}"

    def test_descartar_dama_seguro(self):
        calc = CalculadoraRecompensas()
        cartas = [_Q_ESPADAS, _3_DIAMANTE, _2_TREBOL, _5_PICAS]
        # agente jugó Q♠ (idx 0), no ganó (ganador idx 1)
        r = calc.recompensa_baza_evitada(cartas, 0, 0, numero_baza=5)
        assert r == 2.0, f"descartar Q♠ = +2.0, got {r}"

    def test_descartar_corazon_seguro_tardia(self):
        calc = CalculadoraRecompensas()
        cartas = [_2_CORAZON, _3_DIAMANTE, _2_TREBOL, _5_PICAS]
        r = calc.recompensa_baza_evitada(cartas, 0, 0, numero_baza=9)
        # puntos_baza=1 (0.3) + descartar_corazon (0.2) = 0.5
        assert r == pytest.approx(0.5), f"Esperado 0.5, got {r}"


# ============================================================
# Clase 4 — Métodos tácticos nuevos en CalculadoraRecompensas
# ============================================================

class TestMetodosTacticos:
    """Verifica los métodos tácticos agregados en v12."""

    def test_liderar_pica_no_maxima_con_q_activa(self):
        calc = CalculadoraRecompensas()
        mano = [_5_PICAS, _10_PICAS]
        r = calc.recompensa_liderar_pica(
            agente_idx=0,
            carta_jugada=_5_PICAS,
            mano_agente=mano,
            posicion_en_baza=0,
            dama_picas_activa=True,
        )
        assert r == -2.0, f"Penalización = -2.0, got {r}"

    def test_liderar_pica_maxima_no_penaliza(self):
        calc = CalculadoraRecompensas()
        mano = [_5_PICAS, _10_PICAS]
        r = calc.recompensa_liderar_pica(
            agente_idx=0,
            carta_jugada=_10_PICAS,
            mano_agente=mano,
            posicion_en_baza=0,
            dama_picas_activa=True,
        )
        assert r == 0.0, "Liderar máxima pica no debe penalizar"

    def test_liderar_q_equivocado_temprano(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_liderar_q_equivocado(
            numero_baza=3,
            carta_jugada=_Q_ESPADAS,
            es_maxima_picas=True,
        )
        assert r == -6.0, f"Penalización = -6.0, got {r}"

    def test_liderar_q_dump_seguro(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_liderar_q_dump(
            numero_baza=9,
            carta_jugada=_Q_ESPADAS,
            es_maxima_en_picas=False,
            hay_altas_en_circulacion=True,
        )
        assert r == 2.0, f"Reward = +2.0, got {r}"

    def test_liderar_q_no_seguro_si_maxima_picas(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_liderar_q_dump(
            numero_baza=9,
            carta_jugada=_Q_ESPADAS,
            es_maxima_en_picas=True,
            hay_altas_en_circulacion=True,
        )
        assert r == 0.0, "No debe recompensar si es máxima en picas"

    def test_quemar_maxima_palo_seguro(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_quemar_maxima_palo_seguro(
            posicion_en_baza=0,
            carta_jugada=_carta(14, TREBOL),  # A♣
            es_maxima=True,
        )
        assert r == 1.0, f"Reward = +1.0, got {r}"

    def test_quemar_maxima_no_seguro_si_no_es_maxima(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_quemar_maxima_palo_seguro(
            posicion_en_baza=0,
            carta_jugada=_carta(5, TREBOL),
            es_maxima=False,
        )
        assert r == 0.0

    def test_descartar_k_picas_con_q_activa(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_descartar_k_a_picas(
            carta_jugada=_K_ESPADAS,
            puntos_baza=0,
            dama_picas_activa=True,
            es_descarte=True,
        )
        assert r == 1.0, f"Reward = +1.0, got {r}"

    def test_descartar_k_picas_no_aplica_si_baza_con_puntos(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_descartar_k_a_picas(
            carta_jugada=_K_ESPADAS,
            puntos_baza=5,
            dama_picas_activa=True,
            es_descarte=True,
        )
        assert r == 0.0

    def test_ganar_baza_con_puntos_evitable(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_ganar_baza_con_puntos_evitable(
            numero_baza=9,
            puntos_baza=4,
            pozo_viable=False,
        )
        assert r == -2.0, f"Penalización = -2.0, got {r}"

    def test_descartar_corazon_bajo_roto(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_descartar_corazon_bajo(
            corazones_rotos=True,
            carta_descartada=_2_CORAZON,
            es_descarte=True,
            puntos_baza=0,
        )
        assert r == 0.5, f"Reward = +0.5, got {r}"

    def test_quemar_maxima_forzada(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_quemar_maxima_forzada(
            posicion_en_baza=2,
            forzado_a_ganar=True,
            carta_jugada=_A_ESPADAS,
        )
        assert r == 0.5, f"Reward = +0.5, got {r}"

    def test_quemar_alta_siguiendo_palo(self):
        calc = CalculadoraRecompensas()
        r = calc.recompensa_quemar_alta_siguiendo_palo(
            puntos_baza=0,
            carta_jugada=_K_ESPADAS,
            es_mismo_palo=True,
        )
        assert r == 1.0, f"Reward = +1.0, got {r}"

    def test_fin_partida_posiciones(self):
        calc = CalculadoraRecompensas()
        assert calc.recompensa_fin_partida(
            0, [10, 50, 80, 120]) == 500.0   # 1º
        assert calc.recompensa_fin_partida(
            1, [10, 50, 80, 120]) == 200.0   # 2º
        assert calc.recompensa_fin_partida(
            2, [10, 50, 80, 120]) == -200.0  # 3º
        assert calc.recompensa_fin_partida(
            3, [10, 50, 80, 120]) == -500.0  # 4º


# ============================================================
# Clase 5 — Integridad de la interfaz
# ============================================================

class TestInterfazCompleta:
    """Verifica que CalculadoraRecompensas tenga todos los métodos necesarios."""

    def test_metodos_publicos_completos(self):
        """Todos los métodos que CorazonesEnv necesita deben existir."""
        calc = CalculadoraRecompensas()
        metodos_requeridos = [
            "recompensa_baza_ganada",
            "recompensa_baza_evitada",
            "recompensa_fin_mano",
            "recompensa_fin_partida",
            "recompensa_bloquear_pozo",
            "recompensa_alimentar",
            "recompensa_liderar_pica",
            "recompensa_liderar_q_equivocado",
            "recompensa_liderar_q_dump",
            "recompensa_quemar_maxima_palo_seguro",
            "recompensa_quemar_maxima_forzada",
            "recompensa_quemar_alta_siguiendo_palo",
            "recompensa_descartar_k_a_picas",
            "recompensa_dump_q_siguiendo_picas",
            "recompensa_ganar_baza_con_puntos_evitable",
            "recompensa_ganar_baza_tardia",
            "recompensa_descartar_corazon_bajo",
        ]
        for metodo in metodos_requeridos:
            assert hasattr(calc, metodo), f"Falta método: {metodo}"
            assert callable(getattr(calc, metodo)), f"No es callable: {metodo}"
