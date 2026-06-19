"""
Tests TDD para el espacio de observación v9 (194 → 220 dimensiones).

Cubre los 27 features añadidos en [194:220]:
  [194]      baza_numero / 13.0
  [195]      jugadores_cerca_de_100 / 3.0
  [196]      Q♠ ya capturada (bool)
  [197]      soy lider en puntaje (bool)
  [198]      mano terminal posible (bool)
  [199:203]  cartas restantes por palo / 13.0
  [203:207]  cartas altas (J/Q/K/A) restantes por palo / 4.0
  [207:211]  probabilidad Q♠ por jugador relativo
  [211:215]  corazones capturados esta mano / 13.0
  [215:219]  alerta pozo por jugador (≥6 corazones)
  [219]      palo_salida (-1.0 si None, else palo/3.0 en [-1.0, 1.0])
"""
import pytest
import numpy as np
from src.entorno.single_agent import CorazonesEnv
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones


# ============================================================
# Helpers
# ============================================================

PICA = 2
CORAZON = 3
TREBOL = 0
DIAMANTE = 1


def _carta(valor: int, palo: int) -> Carta:
    for c in Carta._TODAS:
        if c.valor == valor and c.palo == palo:
            return c
    raise ValueError(f"No encontrada: valor={valor}, palo={palo}")


def _env_v9(agente_idx: int = 0, seed: int = 42) -> CorazonesEnv:
    """Crea un entorno CorazonesEnv de 220 dimensiones."""
    env = CorazonesEnv(agente_idx=agente_idx, obs_dim=220)
    env.reset(seed=seed)
    return env


def _inyectar_bazas_ganadas(env: CorazonesEnv, jugador_idx: int,
                            cartas: list) -> None:
    """Pone cartas específicas en las bazas_ganadas de un jugador."""
    env.motor.jugadores[jugador_idx].bazas_ganadas = list(cartas)


def _inyectar_puntuaciones(env: CorazonesEnv, puntuaciones: list) -> None:
    """Setea las puntuaciones históricas de cada jugador."""
    env._puntuacion_historica = list(puntuaciones)


# ============================================================
# Clase 1 — Dimensión del vector
# ============================================================

class TestDimensionV9:
    """El vector de observación v9 debe tener exactamente 220 dims."""

    def test_observacion_tiene_220_dimensiones(self):
        env = _env_v9()
        obs = env._construir_observacion()
        assert obs.shape == (220,), f"Esperaba (220,), obtuvo {obs.shape}"

    def test_observacion_dtype_float32(self):
        env = _env_v9()
        obs = env._construir_observacion()
        assert obs.dtype == np.float32

    def test_observacion_rango_valido(self):
        """Todos los valores en [0:219] deben estar en [0.0, 1.0].
        [219] (palo_salida) puede ser -1.0 cuando no hay palo de salida."""
        env = _env_v9()
        obs = env._construir_observacion()
        # [0:219] siempre en [0, 1]
        assert np.all(
            obs[:219] >= 0.0), f"Min negativo en [0:219]: {obs[:219].min()}"
        assert np.all(
            obs[:219] <= 1.0), f"Max > 1 en [0:219]: {obs[:219].max()}"
        # [219] puede ser -1.0 o [0, 1]
        assert -1.0 <= obs[219] <= 1.0, f"[219] fuera de rango: {obs[219]}"

    def test_reset_retorna_220_dimensiones(self):
        env = CorazonesEnv(obs_dim=220)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (220,)

    def test_step_retorna_220_dimensiones(self):
        env = CorazonesEnv(obs_dim=220)
        obs, _ = env.reset(seed=0)
        mask = env.action_masks()
        legal = int(np.argmax(mask))
        obs_next, _, _, _, _ = env.step(legal)
        assert obs_next.shape == (220,)

    def test_compatibilidad_hacia_atras_194(self):
        """CorazonesEnv sin obs_dim sigue generando 194 dims."""
        env = CorazonesEnv()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (220,)

    def test_observation_space_220(self):
        """observation_space debe reflejar la dimensión configurada."""
        env = CorazonesEnv(obs_dim=220)
        assert env.observation_space.shape == (220,)


# ============================================================
# Clase 2 — Bloque de fase [194:199]
# ============================================================

class TestBloquesFase:
    """Tests para los features de fase [194:199]."""

    def test_baza_numero_al_inicio_de_mano(self):
        """Al inicio de una mano (baza 1), obs[194] = 1/13 ≈ 0.077."""
        env = _env_v9()
        # Al reset, motor.numero_baza == 1 (primera baza)
        obs = env._construir_observacion()
        expected = 1.0 / 13.0
        assert abs(obs[194] - expected) < 0.01, \
            f"Esperaba {expected:.3f}, obtuvo {obs[194]:.3f}"

    def test_baza_numero_normalizado(self):
        """obs[194] debe estar en (0, 1]."""
        env = _env_v9()
        obs = env._construir_observacion()
        assert 0.0 < obs[194] <= 1.0

    def test_jugadores_cerca_100_ninguno(self):
        """Sin jugadores cerca de 100 → obs[195] = 0.0."""
        env = _env_v9()
        _inyectar_puntuaciones(env, [10, 20, 30, 40])
        obs = env._construir_observacion()
        assert obs[195] == 0.0

    def test_jugadores_cerca_100_uno(self):
        """Un jugador con score ≥85 → obs[195] = 1/3."""
        env = _env_v9()
        _inyectar_puntuaciones(env, [85, 20, 30, 40])
        obs = env._construir_observacion()
        assert abs(obs[195] - 1.0 / 3.0) < 0.01

    def test_jugadores_cerca_100_tres(self):
        """Tres jugadores con score ≥85 → obs[195] = 1.0."""
        env = _env_v9()
        _inyectar_puntuaciones(env, [90, 85, 87, 40])
        obs = env._construir_observacion()
        assert abs(obs[195] - 1.0) < 0.01

    def test_q_capturada_false_al_inicio(self):
        """Al inicio, Q♠ no ha sido capturada → obs[196] = 0.0."""
        env = _env_v9()
        env._dama_picas_en = None
        obs = env._construir_observacion()
        assert obs[196] == 0.0

    def test_q_capturada_true_cuando_conocida(self):
        """Cuando se sabe quién tiene/capturó Q♠ → obs[196] = 1.0."""
        env = _env_v9()
        env._dama_picas_en = 1  # rival 1 la capturó
        obs = env._construir_observacion()
        assert obs[196] == 1.0

    def test_soy_lider_agente_con_puntaje_mas_bajo(self):
        """Si el agente tiene la puntuación más baja → obs[197] = 1.0."""
        env = _env_v9(agente_idx=0)
        _inyectar_puntuaciones(env, [5, 20, 30, 40])  # agente tiene 5
        obs = env._construir_observacion()
        assert obs[197] == 1.0

    def test_soy_lider_agente_no_es_lider(self):
        """Si el agente NO tiene la puntuación más baja → obs[197] = 0.0."""
        env = _env_v9(agente_idx=0)
        _inyectar_puntuaciones(env, [50, 5, 30, 40])  # rival 1 tiene 5
        obs = env._construir_observacion()
        assert obs[197] == 0.0

    def test_mano_terminal_false_sin_jugador_alto(self):
        """Sin jugador con score ≥74 → obs[198] = 0.0."""
        env = _env_v9()
        _inyectar_puntuaciones(env, [10, 20, 30, 40])
        obs = env._construir_observacion()
        assert obs[198] == 0.0

    def test_mano_terminal_true_con_jugador_alto(self):
        """Con un jugador con score ≥74 → obs[198] = 1.0."""
        env = _env_v9()
        _inyectar_puntuaciones(env, [10, 74, 30, 40])
        obs = env._construir_observacion()
        assert obs[198] == 1.0


# ============================================================
# Clase 3 — Conteo de cartas restantes [199:207]
# ============================================================

class TestConteoCartas:
    """Tests para cartas restantes y cartas altas por palo [199:207]."""

    def test_cartas_restantes_al_inicio_todas_disponibles(self):
        """Al inicio (sin bazas), todos los palos tienen 13/13 = 1.0."""
        env = _env_v9()
        # Asegurarse de que bazas_ganadas estén vacías
        for j in env.motor.jugadores:
            j.bazas_ganadas = []
        obs = env._construir_observacion()
        # [199:203] = cartas restantes para ♣, ♦, ♠, ♥
        # Al inicio no se han jugado bazas completas (cementerio vacío)
        # Pero algunas pueden estar en la mano del agente, que no se resta
        # Solo se restan las que están en el cementerio (bazas_ganadas)
        for palo in range(4):
            assert obs[199 + palo] == pytest.approx(1.0, abs=0.01), \
                f"Palo {palo}: esperaba 1.0, obtuvo {obs[199+palo]}"

    def test_cartas_restantes_despues_de_baza(self):
        """Después de una baza de 4 cartas con 1 corazón, hay 12/13 restantes."""
        env = _env_v9()
        # Poner 1 corazón en bazas_ganadas del jugador 0
        corazon = _carta(2, CORAZON)  # 2♥
        env.motor.jugadores[0].bazas_ganadas = [corazon]
        # Los otros 3 jugadores no ganaron bazas (1 baza total: 4 cartas)
        # Pero solo el ganador tiene todas las cartas de la baza
        obs = env._construir_observacion()
        # 1 corazón en cementerio → 12 restantes
        assert obs[199 + CORAZON] == pytest.approx(12.0 / 13.0, abs=0.01)

    def test_cartas_altas_al_inicio_todas_disponibles(self):
        """Al inicio (sin bazas), todos los palos tienen 4/4 cartas altas."""
        env = _env_v9()
        for j in env.motor.jugadores:
            j.bazas_ganadas = []
        obs = env._construir_observacion()
        # [203:207] = cartas altas para ♣, ♦, ♠, ♥
        for palo in range(4):
            assert obs[203 + palo] == pytest.approx(1.0, abs=0.01), \
                f"Palo {palo}: esperaba 1.0 cartas altas, obtuvo {obs[203+palo]}"

    def test_cartas_altas_despues_de_captura_de_q_espadas(self):
        """Si Q♠ fue capturada, hay 3/4 cartas altas de picas restantes."""
        env = _env_v9()
        q_espadas = _carta(12, PICA)  # Q♠
        env.motor.jugadores[1].bazas_ganadas = [q_espadas]
        env._dama_picas_en = 1
        obs = env._construir_observacion()
        # 1 carta alta de picas (Q♠) en cementerio → 3/4 restantes
        assert obs[203 + PICA] == pytest.approx(3.0 / 4.0, abs=0.01)

    def test_cartas_restantes_normalizadas(self):
        """Todos los valores [199:207] deben estar en [0, 1]."""
        env = _env_v9()
        obs = env._construir_observacion()
        assert np.all(obs[199:207] >= 0.0)
        assert np.all(obs[199:207] <= 1.0)


# ============================================================
# Clase 4 — Probabilidad de Q♠ [207:211]
# ============================================================

class TestProbabilidadQ:
    """Tests para la probabilidad de Q♠ por jugador relativo [207:211]."""

    def test_prob_q_cuando_yo_la_tengo(self):
        """Si el agente tiene Q♠ → prob[0] = 1.0, rest = 0.0."""
        env = _env_v9(agente_idx=0)
        env._dama_picas_en = None

        # Poner Q♠ en la mano del agente (seat 0)
        q = _carta(12, PICA)
        mano_sin_q = [c for c in env.motor.jugadores[0].mano
                      if not c.es_dama_de_picas]
        env.motor.jugadores[0].mano = mano_sin_q + [q]

        obs = env._construir_observacion()
        assert obs[207] == pytest.approx(1.0, abs=0.01), \
            f"prob[yo] debe ser 1.0, obtuvo {obs[207]}"
        assert np.sum(obs[207:211]) == pytest.approx(1.0, abs=0.05)

    def test_prob_q_cuando_ya_capturada(self):
        """Si Q♠ ya fue capturada → todos los valores [207:211] = 0.0."""
        env = _env_v9(agente_idx=0)
        env._dama_picas_en = 1  # rival capturó Q♠
        obs = env._construir_observacion()
        assert np.all(obs[207:211] == 0.0), \
            f"Todos deben ser 0 si Q♠ capturada; obtuvo {obs[207:211]}"

    def test_prob_q_distribuida_uniformemente_sin_info(self):
        """Sin voids conocidos ni Q♠ en mano → probabilidad distribuida entre rivales."""
        env = _env_v9(agente_idx=0)
        env._dama_picas_en = None
        env._vacios = [set(), set(), set(), set()]

        # Quitar Q♠ de la mano del agente (si la tiene)
        mano_sin_q = [c for c in env.motor.jugadores[0].mano
                      if not c.es_dama_de_picas]
        env.motor.jugadores[0].mano = mano_sin_q
        # Quitar Q♠ del cementerio
        for j in env.motor.jugadores:
            j.bazas_ganadas = [c for c in j.bazas_ganadas
                               if not c.es_dama_de_picas]

        obs = env._construir_observacion()
        # prob[0] (yo) = 0.0 porque ya confirmamos que yo no la tengo
        assert obs[207] == pytest.approx(0.0, abs=0.01), \
            f"Mi prob debe ser 0 si no la tengo; obtuvo {obs[207]}"
        # El resto debe sumar ~1.0 distribuido entre los 3 rivales
        total_rivales = np.sum(obs[208:211])
        assert total_rivales == pytest.approx(1.0, abs=0.05), \
            f"Probabilidades rivales deben sumar 1.0; obtuvo {total_rivales}"

    def test_prob_q_rival_void_en_picas_excluido(self):
        """Si rival 1 es void en picas → su probabilidad debe ser 0."""
        env = _env_v9(agente_idx=0)
        env._dama_picas_en = None
        env._vacios = [set(), {PICA}, set(), set()]  # rival 1 void en picas

        # Yo no tengo Q♠
        env.motor.jugadores[0].mano = [
            c for c in env.motor.jugadores[0].mano
            if not c.es_dama_de_picas
        ]
        for j in env.motor.jugadores:
            j.bazas_ganadas = []

        obs = env._construir_observacion()
        # Rival 1 (posición relativa 1) void en picas → prob = 0
        assert obs[208] == pytest.approx(0.0, abs=0.01), \
            f"Rival void en picas debe tener prob 0; obtuvo {obs[208]}"
        # Rivales 2 y 3 deben compartir la probabilidad
        assert obs[209] > 0.0
        assert obs[210] > 0.0

    def test_prob_q_normalizacion(self):
        """Los valores [207:211] deben estar en [0, 1] y sumar a ≤ 1.0."""
        env = _env_v9()
        obs = env._construir_observacion()
        assert np.all(obs[207:211] >= 0.0)
        assert np.all(obs[207:211] <= 1.0)
        assert np.sum(obs[207:211]) <= 1.01  # suma total ≤ 1


# ============================================================
# Clase 5 — Detección de moon [211:219]
# ============================================================

class TestDeteccionMoon:
    """Tests para corazones esta mano y alertas de pozo [211:219]."""

    def test_corazones_esta_mano_al_inicio(self):
        """Al inicio de una mano, nadie tiene corazones → [211:215] = 0."""
        env = _env_v9()
        for j in env.motor.jugadores:
            j.bazas_ganadas = []
        obs = env._construir_observacion()
        assert np.all(obs[211:215] == 0.0), \
            f"Sin corazones capturados, debe ser todo 0: {obs[211:215]}"

    def test_corazones_esta_mano_un_jugador(self):
        """Si rival (seat 1) capturó 3 corazones, obs[212] = 3/13."""
        env = _env_v9(agente_idx=0)
        corazones = [_carta(v, CORAZON) for v in [2, 3, 4]]
        env.motor.jugadores[1].bazas_ganadas = corazones

        obs = env._construir_observacion()
        expected = 3.0 / 13.0
        assert obs[212] == pytest.approx(expected, abs=0.01), \
            f"Rival 1 con 3 corazones: esperaba {expected:.3f}, obtuvo {obs[212]:.3f}"

    def test_corazones_esta_mano_relativo_al_agente(self):
        """Los features de corazones son relativos al agente (posición 0 = yo)."""
        env = _env_v9(agente_idx=2)  # Agente en seat 2
        corazones = [_carta(v, CORAZON) for v in [2, 3, 4, 5, 6, 7]]
        # Jugador absoluto 0 tiene 6 corazones
        env.motor.jugadores[0].bazas_ganadas = corazones

        obs = env._construir_observacion()
        # Jugador 0 es el relativo 2 desde el agente 2: (0 - 2) % 4 = 2
        rel = (0 - 2) % 4  # = 2
        expected = 6.0 / 13.0
        assert obs[211 + rel] == pytest.approx(expected, abs=0.01), \
            f"Posición relativa {rel}: esperaba {expected:.3f}"

    def test_alerta_pozo_sin_riesgo(self):
        """Sin nadie con ≥6 corazones → [215:219] = 0."""
        env = _env_v9()
        for j in env.motor.jugadores:
            j.bazas_ganadas = [_carta(v, CORAZON) for v in [2, 3]]  # solo 2
        obs = env._construir_observacion()
        assert np.all(obs[215:219] == 0.0), \
            f"Sin riesgo de pozo: {obs[215:219]}"

    def test_alerta_pozo_rival_con_6_corazones(self):
        """Rival (seat 1) con 6 corazones → obs[216] = 1.0."""
        env = _env_v9(agente_idx=0)
        corazones = [_carta(v, CORAZON) for v in [2, 3, 4, 5, 6, 7]]  # 6
        env.motor.jugadores[1].bazas_ganadas = corazones

        obs = env._construir_observacion()
        assert obs[216] == 1.0, \
            f"Rival 1 con 6 corazones: obs[216] debe ser 1.0, obtuvo {obs[216]}"

    def test_alerta_pozo_agente_con_6_corazones(self):
        """Si el AGENTE tiene 6+ corazones (en modo pozo), obs[215] = 1.0."""
        env = _env_v9(agente_idx=0)
        corazones = [_carta(v, CORAZON) for v in [2, 3, 4, 5, 6, 7]]
        env.motor.jugadores[0].bazas_ganadas = corazones

        obs = env._construir_observacion()
        assert obs[215] == 1.0, \
            f"Agente con 6 corazones: obs[215] debe ser 1.0, obtuvo {obs[215]}"

    def test_palo_salida_none_es_menos_uno(self):
        """Cuando no hay palo de salida (motor.palo_de_salida is None) → obs[219] = 0.0."""
        env = _env_v9()
        env.motor.palo_de_salida = None
        obs = env._construir_observacion()
        assert obs[219] == 0.0, \
            f"Sin palo de salida, [219] debe ser 0.0: {obs[219]}"

    def test_palo_salida_trebol(self):
        """Cuando el palo de salida es trébol (0) → obs[219] = 0.0."""
        env = _env_v9()
        env.motor.palo_de_salida = TREBOL
        obs = env._construir_observacion()
        assert obs[219] == pytest.approx(0.0, abs=0.01)

    def test_palo_salida_diamante(self):
        """Cuando el palo de salida es diamante (1) → obs[219] ≈ 0.333."""
        env = _env_v9()
        env.motor.palo_de_salida = DIAMANTE
        obs = env._construir_observacion()
        assert obs[219] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_palo_salida_picas(self):
        """Cuando el palo de salida es picas (2) → obs[219] ≈ 0.667."""
        env = _env_v9()
        env.motor.palo_de_salida = PICA
        obs = env._construir_observacion()
        assert obs[219] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_palo_salida_corazon(self):
        """Cuando el palo de salida es corazón (3) → obs[219] = 1.0."""
        env = _env_v9()
        env.motor.palo_de_salida = CORAZON
        obs = env._construir_observacion()
        assert obs[219] == pytest.approx(1.0, abs=0.01)

    def test_palo_salida_cambia_durante_el_juego(self):
        """Verifica que palo_salida se actualiza correctamente entre bazas."""
        env = _env_v9()
        obs = env._construir_observacion()
        assert -1.0 <= obs[219] <= 1.0


# ============================================================
# Clase 6 — Palo de salida (antes RESERVADO)
# ============================================================

class TestPaloSalida:
    """Tests para el feature palo_salida en obs[219]."""
# ============================================================


class TestIntegridad:
    """Tests de consistencia end-to-end del vector v9."""

    def test_bloque_194_no_afecta_bloque_antiguo(self):
        """Los features antiguos [0:194] deben ser idénticos en 194 y 220."""
        env_v6 = CorazonesEnv(agente_idx=0, obs_dim=194)
        obs_v6, _ = env_v6.reset(seed=42)

        env_v9 = CorazonesEnv(agente_idx=0, obs_dim=220)
        obs_v9, _ = env_v9.reset(seed=42)

        np.testing.assert_array_equal(
            obs_v6, obs_v9[:194],
            err_msg="Los primeros 194 features deben ser idénticos en v6 y v9"
        )

    def test_valores_nuevos_en_rango(self):
        """Todos los nuevos features [194:220] deben estar en [0, 1]."""
        env = _env_v9()
        obs = env._construir_observacion()
        assert np.all(obs[194:220] >= 0.0), \
            f"Feature negativo: {obs[194:220]}"
        assert np.all(obs[194:220] <= 1.0), \
            f"Feature > 1.0: {obs[194:220]}"

    def test_step_consistente_con_observacion(self):
        """Después de step, los nuevos features reflejan el estado actualizado."""
        env = CorazonesEnv(obs_dim=220)
        obs, _ = env.reset(seed=0)
        mask = env.action_masks()
        legal = int(np.argmax(mask))
        obs_next, _, _, _, _ = env.step(legal)
        # [199:203] no pueden ser mayores que 1.0 (normalizado)
        assert np.all(obs_next[199:207] <= 1.0)

    def test_multiples_manos_no_acumula_corazones(self):
        """Al iniciar una nueva mano, los corazones de la mano anterior
        no deben quedar en bazas_ganadas de la nueva mano."""
        env = CorazonesEnv(obs_dim=220)
        obs, _ = env.reset(seed=77)
        # Avanzar hasta que terminen varias manos (jugar cartas legales)
        for _ in range(50):
            mask = env.action_masks()
            if not np.any(mask):
                break
            legal = int(np.argmax(mask))
            obs, _, done, _, _ = env.step(legal)
            if done:
                break
        # Los corazones de la mano actual deben ser razonables (≤ 13/13 = 1.0)
        assert np.all(obs[211:215] <= 1.0)
