"""
Tests TDD para el rediseño de recompensas v9 (Fase 6).

Cambios respecto a v5:
  - Recompensas densas solo activas en bazas ≥9 (phase-gating)
  - REWARD_Q_SPADES_SIN_POZO: -8.0 → -10.0
  - REWARD_DESCARTAR_DAMA_SEGURO: 3.0 → 5.0
  - Nueva: REWARD_BLOQUEAR_POZO = 15.0
  - Nueva: REWARD_ALIMENTAR_EXITOSO = 10.0
  - Nueva: REWARD_CORAZON_POZO = 1.5 (positivo en modo pozo)
  - REWARD_POR_PUNTO_EN_MANO: -0.2 → -0.1
  - REWARD_NO_GANAR_BAZA_CON_PUNTOS ahora solo en baza ≥9
  - REWARD_GANAR_BAZA_SIN_PUNTOS ahora solo en baza ≥9
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


_Q_ESPADAS = _carta(12, 2)   # Q♠
_A_CORAZON = _carta(14, 3)   # A♥
_2_CORAZON = _carta(2, 3)    # 2♥
_2_TREBOL = _carta(2, 0)     # 2♣
_K_ESPADAS = _carta(13, 2)   # K♠
_3_DIAMANTE = _carta(3, 1)   # 3♦


def _cfg_v9() -> RewardConfig:
    """RewardConfig con valores v9."""
    return RewardConfig(
        REWARD_Q_SPADES_SIN_POZO=-10.0,
        REWARD_DESCARTAR_DAMA_SEGURO=5.0,
        REWARD_POR_PUNTO_EN_MANO=-0.1,
        REWARD_BLOQUEAR_POZO=15.0,
        REWARD_ALIMENTAR_EXITOSO=10.0,
        REWARD_CORAZON_POZO=1.5,
    )


def _calc_v9() -> CalculadoraRecompensas:
    return CalculadoraRecompensas(config=_cfg_v9())


# ============================================================
# Clase 1 — RewardConfig v9
# ============================================================

class TestRewardConfigV9:
    """Verifica que los nuevos valores estén en RewardConfig."""

    def test_config_por_defecto_tiene_bloquear_pozo(self):
        cfg = RewardConfig()
        assert hasattr(cfg, "REWARD_BLOQUEAR_POZO"), \
            "RewardConfig debe tener REWARD_BLOQUEAR_POZO"

    def test_config_por_defecto_tiene_alimentar_exitoso(self):
        cfg = RewardConfig()
        assert hasattr(cfg, "REWARD_ALIMENTAR_EXITOSO")

    def test_config_por_defecto_tiene_corazon_pozo(self):
        cfg = RewardConfig()
        assert hasattr(cfg, "REWARD_CORAZON_POZO")

    def test_config_v9_valores_correctos(self):
        cfg = _cfg_v9()
        assert cfg.REWARD_Q_SPADES_SIN_POZO == -10.0
        assert cfg.REWARD_DESCARTAR_DAMA_SEGURO == 5.0
        assert cfg.REWARD_POR_PUNTO_EN_MANO == -0.1
        assert cfg.REWARD_BLOQUEAR_POZO == 15.0
        assert cfg.REWARD_ALIMENTAR_EXITOSO == 10.0
        assert cfg.REWARD_CORAZON_POZO == 1.5

    def test_recompensas_terminales_sin_cambio(self):
        """Las recompensas terminales no se modificaron."""
        cfg = RewardConfig()
        assert cfg.REWARD_PRIMERO == 500.0
        assert cfg.REWARD_SEGUNDO == 200.0
        assert cfg.REWARD_TERCERO == -200.0
        assert cfg.REWARD_CUARTO == -500.0
        assert cfg.REWARD_SHOOTING_MOON == 50.0


# ============================================================
# Clase 2 — Phase-gating: bazas tempranas (1-8)
# ============================================================

class TestPhaseGatingBazasTempranas:
    """Las recompensas densas NO se activan en bazas 1-8."""

    def test_no_recompensa_por_evitar_puntos_baza_temprana(self):
        """En baza 5, evitar baza con puntos no da recompensa."""
        calc = _calc_v9()
        cartas = [_A_CORAZON, _2_TREBOL, _2_CORAZON, _3_DIAMANTE]
        # Agente NO ganó baza con puntos
        reward = calc.recompensa_baza_evitada(
            cartas_baza=cartas,
            agente_idx=0,
            idx_agente_en_mesa=1,  # agente jugó 2♣ (sin puntos)
            numero_baza=5,  # baza temprana
        )
        assert reward == pytest.approx(0.0, abs=0.01), \
            f"Baza temprana: evitar puntos no debe dar reward; obtuvo {reward}"

    def test_no_penalizacion_ganar_baza_limpia_temprana(self):
        """En baza 3, ganar baza sin puntos no penaliza."""
        calc = _calc_v9()
        cartas = [_2_TREBOL, _K_ESPADAS, _3_DIAMANTE, _carta(4, 1)]
        reward = calc.recompensa_baza_ganada(
            cartas_baza=cartas,
            agente_idx=0,
            ganador=0,
            pozo_viable=False,
            numero_baza=3,  # baza temprana
        )
        # Solo penaliza K♠ si es Q♠ — aquí hay K♠ pero no Q♠
        assert reward == pytest.approx(0.0, abs=0.01), \
            f"Baza temprana sin puntos: no debe penalizar; obtuvo {reward}"

    def test_penalizacion_corazon_siempre_aplica(self):
        """La penalización por corazón aplica en CUALQUIER baza."""
        calc = _calc_v9()
        cartas = [_A_CORAZON, _2_TREBOL, _3_DIAMANTE, _carta(4, 1)]
        reward = calc.recompensa_baza_ganada(
            cartas_baza=cartas,
            agente_idx=0,
            ganador=0,
            pozo_viable=False,
            numero_baza=2,  # baza temprana
        )
        # -1.0 (corazón) + -3.0 (ganar corazón sin pozo)
        assert reward < 0, f"Ganar corazón siempre penaliza; obtuvo {reward}"

    def test_penalizacion_q_espadas_siempre_aplica(self):
        """La penalización por Q♠ (sin pozo) aplica en CUALQUIER baza."""
        calc = _calc_v9()
        cartas = [_Q_ESPADAS, _2_TREBOL, _3_DIAMANTE, _carta(4, 1)]
        reward = calc.recompensa_baza_ganada(
            cartas_baza=cartas,
            agente_idx=0,
            ganador=0,
            pozo_viable=False,
            numero_baza=4,  # baza temprana
        )
        # -10.0 (dama) + -10.0 (sin pozo) = -20.0
        assert reward <= -18.0, \
            f"Q♠ sin pozo debe penalizar fuerte incluso en baza temprana; obtuvo {reward}"


# ============================================================
# Clase 3 — Phase-gating: bazas tardías (9-13)
# ============================================================

class TestPhaseGatingBazasTardias:
    """Las recompensas densas SÍ se activan en bazas ≥9."""

    def test_recompensa_por_evitar_puntos_baza_tardia(self):
        """En baza 10, evitar baza con puntos da recompensa."""
        calc = _calc_v9()
        cartas = [_A_CORAZON, _2_TREBOL, _2_CORAZON, _3_DIAMANTE]
        reward = calc.recompensa_baza_evitada(
            cartas_baza=cartas,
            agente_idx=0,
            idx_agente_en_mesa=1,  # agente jugó 2♣
            numero_baza=10,  # baza tardía
        )
        assert reward > 0, \
            f"Baza tardía: evitar puntos debe dar reward positivo; obtuvo {reward}"

    def test_penalizacion_ganar_baza_limpia_tardia(self):
        """En baza 11, ganar baza sin puntos penaliza."""
        calc = _calc_v9()
        cartas = [_2_TREBOL, _K_ESPADAS, _3_DIAMANTE, _carta(4, 1)]
        reward = calc.recompensa_baza_ganada(
            cartas_baza=cartas,
            agente_idx=0,
            ganador=0,
            pozo_viable=False,
            numero_baza=11,  # baza tardía
        )
        assert reward < 0, \
            f"Baza tardía sin puntos: debe penalizar por perder control; obtuvo {reward}"

    def test_descartar_dama_seguro_siempre_recompensa(self):
        """Descartar Q♠ (sin ganar baza) siempre da recompensa."""
        calc = _calc_v9()
        cartas = [_A_CORAZON, _Q_ESPADAS, _2_TREBOL, _3_DIAMANTE]
        # Agente jugó Q♠ pero el ganador fue otro (A♥ gana si es corazón... no)
        # En este escenario: baza de corazones, agente fue void y descartó Q♠
        reward = calc.recompensa_baza_evitada(
            cartas_baza=cartas,
            agente_idx=0,
            idx_agente_en_mesa=1,  # posición 1 en mesa = agente jugó Q♠
            numero_baza=5,
        )
        # Debe dar recompensa por descartar Q♠
        assert reward >= calc.cfg.REWARD_DESCARTAR_DAMA_SEGURO, \
            f"Descartar Q♠ debe dar {calc.cfg.REWARD_DESCARTAR_DAMA_SEGURO}; obtuvo {reward}"


# ============================================================
# Clase 4 — Recompensa por bloquear pozo
# ============================================================

class TestBloquearPozo:
    """La recompensa BLOQUEAR_POZO se activa cuando el agente interrumpe un moon."""

    def test_bloquear_pozo_agente_captura_q_cuando_rival_tiene_muchos_corazones(self):
        """Si un rival acumuló ≥10 corazones y el agente capturó Q♠ → reward."""
        calc = _calc_v9()
        # rival 1 tiene todos los corazones
        corazones_por_jugador = [0, 13, 0, 0]
        reward = calc.recompensa_bloquear_pozo(
            agente_idx=0,
            ganador_q_espadas=0,  # agente capturó Q♠
            corazones_por_jugador=corazones_por_jugador,
            pleno_jugador=None,   # pozo no exitoso
        )
        assert reward == pytest.approx(15.0), \
            f"Bloquear pozo debe dar 15.0; obtuvo {reward}"

    def test_bloquear_pozo_no_activa_si_pozo_exitoso(self):
        """Si el pozo fue exitoso, no hay bloqueo que recompensar."""
        calc = _calc_v9()
        reward = calc.recompensa_bloquear_pozo(
            agente_idx=0,
            ganador_q_espadas=1,
            corazones_por_jugador=[0, 13, 0, 0],
            pleno_jugador=1,  # rival 1 hizo pozo
        )
        assert reward == pytest.approx(0.0)

    def test_bloquear_pozo_no_activa_sin_rival_con_muchos_corazones(self):
        """Si ningún rival tiene ≥10 corazones, no hubo intento de pozo."""
        calc = _calc_v9()
        reward = calc.recompensa_bloquear_pozo(
            agente_idx=0,
            ganador_q_espadas=0,
            corazones_por_jugador=[0, 5, 3, 2],  # nadie cerca del moon
            pleno_jugador=None,
        )
        assert reward == pytest.approx(0.0)

    def test_bloquear_pozo_no_activa_si_agente_no_capturo_q(self):
        """Si el agente no capturó Q♠, no bloqueó directamente."""
        calc = _calc_v9()
        reward = calc.recompensa_bloquear_pozo(
            agente_idx=0,
            ganador_q_espadas=2,  # rival 2 tiene Q♠
            corazones_por_jugador=[0, 12, 0, 0],
            pleno_jugador=None,
        )
        assert reward == pytest.approx(0.0)


# ============================================================
# Clase 5 — Recompensa en modo pozo propio
# ============================================================

class TestModoPozo:
    """En modo pozo, capturar corazones debe ser recompensado, no penalizado."""

    def test_corazon_en_modo_pozo_da_recompensa_positiva(self):
        """Si pozo_viable=True, ganar corazón da REWARD_CORAZON_POZO (positivo)."""
        calc = _calc_v9()
        cartas = [_A_CORAZON, _2_TREBOL, _3_DIAMANTE, _carta(4, 1)]
        reward = calc.recompensa_baza_ganada(
            cartas_baza=cartas,
            agente_idx=0,
            ganador=0,
            pozo_viable=True,  # en modo pozo
            numero_baza=6,
        )
        # No debe penalizar el corazón en modo pozo
        assert reward >= 0, \
            f"En modo pozo, ganar corazón no debe penalizar; obtuvo {reward}"

    def test_corazon_sin_pozo_penaliza(self):
        """Si pozo_viable=False, ganar corazón penaliza."""
        calc = _calc_v9()
        cartas = [_A_CORAZON, _2_TREBOL, _3_DIAMANTE, _carta(4, 1)]
        reward = calc.recompensa_baza_ganada(
            cartas_baza=cartas,
            agente_idx=0,
            ganador=0,
            pozo_viable=False,
            numero_baza=6,
        )
        assert reward < 0, f"Sin pozo, corazón debe penalizar; obtuvo {reward}"


# ============================================================
# Clase 6 — Recompensa alimentar rival
# ============================================================

class TestAlimentar:
    """La recompensa ALIMENTAR_EXITOSO se dispara al fin de mano en condiciones específicas."""

    def test_alimentar_exitoso_rival_cerca_100_recibe_puntos(self):
        """Si el rival objetivo (el que estaba cerca de 100) recibió puntos → reward."""
        calc = _calc_v9()
        puntuaciones_mano = [2, 0, 0, 18]  # rival 3 recibió 18 pts
        puntuaciones_historicas = [30, 20, 25, 85]  # rival 3 estaba cerca
        reward = calc.recompensa_alimentar(
            agente_idx=0,
            puntuaciones_mano=puntuaciones_mano,
            puntuaciones_historicas=puntuaciones_historicas,
        )
        assert reward == pytest.approx(10.0), \
            f"Alimentar exitoso debe dar 10.0; obtuvo {reward}"

    def test_alimentar_no_activa_sin_rival_cerca(self):
        """Sin rival cerca de 100, no aplica alimentar."""
        calc = _calc_v9()
        puntuaciones_mano = [2, 0, 0, 18]
        puntuaciones_historicas = [30, 20, 25, 60]  # nadie cerca de 100
        reward = calc.recompensa_alimentar(
            agente_idx=0,
            puntuaciones_mano=puntuaciones_mano,
            puntuaciones_historicas=puntuaciones_historicas,
        )
        assert reward == pytest.approx(0.0)

    def test_alimentar_no_activa_si_rival_no_recibio_puntos(self):
        """Si el rival cerca de 100 no recibió puntos esta mano → no aplica."""
        calc = _calc_v9()
        puntuaciones_mano = [2, 20, 2, 0]  # rival 3 (cerca de 100) no recibió
        puntuaciones_historicas = [30, 20, 25, 85]
        reward = calc.recompensa_alimentar(
            agente_idx=0,
            puntuaciones_mano=puntuaciones_mano,
            puntuaciones_historicas=puntuaciones_historicas,
        )
        assert reward == pytest.approx(0.0)


# ============================================================
# Clase 7 — Punto por mano (reducido)
# ============================================================

class TestPuntoPorMano:
    """REWARD_POR_PUNTO_EN_MANO debe ser -0.1 (era -0.2)."""

    def test_penalizacion_reducida_por_punto(self):
        calc = _calc_v9()
        reward = calc.recompensa_fin_mano(
            agente_idx=0,
            puntuaciones_mano=[10, 5, 5, 6],  # agente tomó 10 pts
            pleno_jugador=None,
        )
        # -2.0 (perder mano) + 10 * -0.1 = -3.0
        assert reward == pytest.approx(-3.0, abs=0.1), \
            f"Penalización reducida: esperaba -3.0 aprox., obtuvo {reward}"


# ============================================================
# Clase 8 — Nuevas recompensas tácticas (Fase B, v10)
# ============================================================

class TestRecompensasTacticasV10:
    """Tests para las 3 nuevas recompensas tácticas basadas en errores del BotExperto."""

    def test_config_tiene_penalty_ganar_baza_tardia(self):
        """RewardConfig debe tener PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD."""
        cfg = RewardConfig()
        assert hasattr(cfg, "PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD")
        assert cfg.PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD == -2.0

    def test_config_tiene_reward_liderar_q_dump_seguro(self):
        """RewardConfig debe tener REWARD_LIDERAR_Q_DUMP_SEGURO."""
        cfg = RewardConfig()
        assert hasattr(cfg, "REWARD_LIDERAR_Q_DUMP_SEGURO")
        assert cfg.REWARD_LIDERAR_Q_DUMP_SEGURO == 3.0

    def test_config_tiene_reward_descartar_corazon_bajo_roto(self):
        """RewardConfig debe tener REWARD_DESCARTAR_CORAZON_BAJO_ROTO."""
        cfg = RewardConfig()
        assert hasattr(cfg, "REWARD_DESCARTAR_CORAZON_BAJO_ROTO")
        assert cfg.REWARD_DESCARTAR_CORAZON_BAJO_ROTO == 0.5

    def test_penalty_ganar_baza_tardia_aplica(self):
        """Penalización -2.0 cuando baza ≥9 y se gana baza sin puntos con carta alta de palo seguro."""
        calc = _calc_v9()
        reward = calc.recompensa_ganar_baza_tardia(
            numero_baza=10,
            puntos_baza=0,
            palo_salida=TREBOL,
            carta_jugada=_carta(14, TREBOL),  # A♣ = máxima de palo seguro
            es_maxima_en_mano=True,
        )
        assert reward == pytest.approx(-2.0, abs=0.01)

    def test_penalty_ganar_baza_tardia_no_aplica_baza_temprana(self):
        """No aplica en baza < 9."""
        calc = _calc_v9()
        reward = calc.recompensa_ganar_baza_tardia(
            numero_baza=5,
            puntos_baza=0,
            palo_salida=TREBOL,
            carta_jugada=_carta(14, TREBOL),
            es_maxima_en_mano=True,
        )
        assert reward == 0.0

    def test_penalty_ganar_baza_tardia_no_aplica_con_puntos(self):
        """No aplica si la baza tiene puntos (ya hay otras penalizaciones)."""
        calc = _calc_v9()
        reward = calc.recompensa_ganar_baza_tardia(
            numero_baza=10,
            puntos_baza=5,
            palo_salida=TREBOL,
            carta_jugada=_carta(14, TREBOL),
            es_maxima_en_mano=True,
        )
        assert reward == 0.0

    def test_penalty_ganar_baza_tardia_no_aplica_no_maxima(self):
        """No aplica si la carta no es la máxima del palo."""
        calc = _calc_v9()
        reward = calc.recompensa_ganar_baza_tardia(
            numero_baza=10,
            puntos_baza=0,
            palo_salida=TREBOL,
            carta_jugada=_carta(5, TREBOL),
            es_maxima_en_mano=False,
        )
        assert reward == 0.0

    def test_reward_liderar_q_dump_seguro_aplica(self):
        """+3.0 cuando baza ≥7, K♠/A♠ en circulación, se lidera Q♠."""
        calc = _calc_v9()
        reward = calc.recompensa_liderar_q_dump(
            numero_baza=8,
            carta_jugada=_carta(12, PICA),  # Q♠
            es_maxima_en_picas=False,         # no soy máxima en picas
            hay_altas_en_circulacion=True,    # K♠/A♠ aún en juego
        )
        assert reward == pytest.approx(3.0, abs=0.01)

    def test_reward_liderar_q_dump_no_aplica_baza_temprana(self):
        """No aplica en baza < 7."""
        calc = _calc_v9()
        reward = calc.recompensa_liderar_q_dump(
            numero_baza=5,
            carta_jugada=_carta(12, PICA),
            es_maxima_en_picas=False,
            hay_altas_en_circulacion=True,
        )
        assert reward == 0.0

    def test_reward_liderar_q_dump_no_aplica_siendo_maxima(self):
        """No aplica si soy máxima en picas (sería auto-13pts)."""
        calc = _calc_v9()
        reward = calc.recompensa_liderar_q_dump(
            numero_baza=9,
            carta_jugada=_carta(12, PICA),
            es_maxima_en_picas=True,
            hay_altas_en_circulacion=True,
        )
        assert reward == 0.0

    def test_reward_liderar_q_dump_no_aplica_sin_cobertura(self):
        """No aplica si no hay K♠/A♠ en circulación."""
        calc = _calc_v9()
        reward = calc.recompensa_liderar_q_dump(
            numero_baza=9,
            carta_jugada=_carta(12, PICA),
            es_maxima_en_picas=False,
            hay_altas_en_circulacion=False,
        )
        assert reward == 0.0

    def test_reward_descartar_corazon_bajo_roto_aplica(self):
        """+0.5 cuando corazones rotos y se descarta corazón en baza limpia."""
        calc = _calc_v9()
        reward = calc.recompensa_descartar_corazon_bajo(
            corazones_rotos=True,
            carta_descartada=_carta(5, CORAZON),
            es_descarte=True,   # void en palo de salida
            puntos_baza=0,
        )
        assert reward == pytest.approx(0.5, abs=0.01)

    def test_reward_descartar_corazon_bajo_no_aplica_sin_rotos(self):
        """No aplica si corazones no están rotos."""
        calc = _calc_v9()
        reward = calc.recompensa_descartar_corazon_bajo(
            corazones_rotos=False,
            carta_descartada=_carta(5, CORAZON),
            es_descarte=True,
            puntos_baza=0,
        )
        assert reward == 0.0

    def test_reward_descartar_corazon_bajo_no_aplica_siguiendo_palo(self):
        """No aplica si el agente está siguiendo palo (no descartando)."""
        calc = _calc_v9()
        reward = calc.recompensa_descartar_corazon_bajo(
            corazones_rotos=True,
            carta_descartada=_carta(5, CORAZON),
            es_descarte=False,
            puntos_baza=0,
        )
        assert reward == 0.0

    def test_reward_descartar_corazon_bajo_no_aplica_con_puntos(self):
        """No aplica si la baza tiene puntos."""
        calc = _calc_v9()
        reward = calc.recompensa_descartar_corazon_bajo(
            corazones_rotos=True,
            carta_descartada=_carta(5, CORAZON),
            es_descarte=True,
            puntos_baza=3,
        )
        assert reward == 0.0
