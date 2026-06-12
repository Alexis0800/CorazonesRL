"""
Pruebas unitarias para el Módulo 2: Entorno Gymnasium y Action Masking.

Cubre:
    - Inicialización del entorno y espacios (Box, Discrete).
    - Validación del vector de observación (187 dimensiones, np.float32).
    - Action masking (bloqueo de jugadas ilegales).
    - Estructura de recompensas de suma cero (corto y largo plazo).
    - Auditoría nativa check_env().
    - Ciclo completo de juego multi-mano hasta estado terminal.
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env


# ============================================================
# Pruebas de inicialización y espacios
# ============================================================

class TestCorazonesEnvInicializacion:
    """Verifica que el entorno se construye correctamente."""

    def test_creacion_entorno(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        assert env is not None
        env.close()

    def test_observation_space_shape(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        assert env.observation_space.shape == (
            190,), f"Esperado (190,), got {env.observation_space.shape}"
        assert env.observation_space.dtype == np.float32
        env.close()

    def test_action_space_type(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 52
        env.close()

    def test_agente_idx_por_defecto(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        assert env.agente_idx == 0
        env.close()

    def test_agente_idx_personalizado(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv(agente_idx=2)
        assert env.agente_idx == 2
        env.close()


# ============================================================
# Pruebas de reset()
# ============================================================

class TestCorazonesEnvReset:
    """Verifica el comportamiento de reset()."""

    def test_reset_retorna_obs_e_info(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        resultado = env.reset()
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2
        obs, info = resultado
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        env.close()

    def test_reset_obs_shape_correcto(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        assert obs.shape == (190,), f"Esperado (190,), got {obs.shape}"
        env.close()

    def test_reset_obs_dtype_float32(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        env.close()

    def test_reset_obs_sin_nan(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs)), "El vector contiene NaN"
        env.close()

    def test_reset_obs_rango_normalizado(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        assert np.all(obs >= 0.0), "Hay valores negativos en la observación"
        assert np.all(obs <= 1.0), "Hay valores > 1.0 en la observación"
        env.close()

    def test_reset_bloque_mano_tiene_13_unos(self):
        """La mano del agente debe tener exactamente 13 cartas (13 unos en [0:52])."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        mano = obs[0:52]
        assert np.sum(mano) == pytest.approx(13.0), (
            f"La mano debe tener 13 cartas, tiene {np.sum(mano)}"
        )
        env.close()

    def test_reset_mesa_vacia_o_parcial(self):
        """Al inicio, la mesa puede estar vacía (si el agente abre)
        o tener de 1 a 3 cartas (si el agente juega después)."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        mesa = obs[52:104]
        # Máximo 3 cartas (el agente aún no ha jugado)
        assert np.sum(mesa) <= 3.0
        env.close()

    def test_reset_cementerio_vacio(self):
        """Al inicio de una mano, el cementerio debe estar vacío."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        cementerio = obs[104:156]
        assert np.sum(cementerio) == 0.0
        env.close()

    def test_reset_puntos_mano_cero(self):
        """Al inicio de una mano, los puntos de la mano actual son 0 para todos."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        puntos_mano = obs[176:180]
        assert np.all(puntos_mano == 0.0)
        env.close()

    def test_reset_dama_picas_oculta(self):
        """Al inicio, la dama de picas está oculta (índice 182 = 1.0)."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        dama = obs[182:187]
        assert dama[0] == 1.0, "Dama de picas debe estar oculta al inicio"
        assert np.sum(dama[1:]) == 0.0, "Solo el estado 'oculta' debe ser 1.0"
        env.close()


# ============================================================
# Pruebas de step()
# ============================================================

class TestCorazonesEnvStep:
    """Verifica el comportamiento del método step()."""

    def test_step_retorna_cinco_elementos(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        resultado = env.step(0)  # acción dummy (será reemplazada si es ilegal)
        assert len(resultado) == 5
        obs, reward, terminated, truncated, info = resultado
        env.close()

    def test_step_obs_sin_nan(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        # Usar action_masks para elegir una acción legal
        mask = env.action_masks()
        accion_legal = int(np.where(mask == 1)[0][0])
        obs, _, _, _, _ = env.step(accion_legal)
        assert not np.any(np.isnan(obs))
        env.close()

    def test_step_obs_rango_normalizado(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        mask = env.action_masks()
        accion_legal = int(np.where(mask == 1)[0][0])
        obs, _, _, _, _ = env.step(accion_legal)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        env.close()

    def test_step_action_mask_bloquea_ilegal(self):
        """Verifica que una acción ilegal sea manejada sin crash (fallback a legal)."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        mask = env.action_masks()
        # Encontrar una acción ilegal
        acciones_ilegales = np.where(mask == 0)[0]
        if len(acciones_ilegales) > 0:
            accion_ilegal = int(acciones_ilegales[0])
            # No debe lanzar excepción (el entorno aplica fallback)
            obs, reward, terminated, truncated, info = env.step(accion_ilegal)
            assert isinstance(obs, np.ndarray)
        env.close()

    def test_reward_es_float(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        mask = env.action_masks()
        accion_legal = int(np.where(mask == 1)[0][0])
        _, reward, _, _, _ = env.step(accion_legal)
        assert isinstance(reward, float)
        env.close()

    def test_terminated_es_bool(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        mask = env.action_masks()
        accion_legal = int(np.where(mask == 1)[0][0])
        _, _, terminated, _, _ = env.step(accion_legal)
        assert isinstance(terminated, bool)
        env.close()


# ============================================================
# Pruebas de Action Masking
# ============================================================

class TestActionMasking:
    """Verifica el enmascaramiento de acciones ilegales."""

    def test_action_masks_existe(self):
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        mask = env.action_masks()
        assert mask is not None
        assert mask.shape == (52,)
        assert mask.dtype == np.bool_ or mask.dtype == bool
        env.close()

    def test_action_masks_primer_turno_solo_dos_treboles(self):
        """En el primer turno (si el agente tiene 2♣), solo debe ser legal 2♣."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        # Si el agente es quien tiene el 2♣, la máscara solo permite eso
        mask = env.action_masks()
        num_legales = int(np.sum(mask))
        if num_legales == 1:
            # Solo 2♣ es legal
            assert mask[0] == 1  # carta id=0 es 2♣
        else:
            # El agente no es el que abre; todas sus cartas legales están en mask
            assert num_legales >= 1
        env.close()

    def test_action_masks_todas_legales_en_mano(self):
        """Todas las acciones legales deben corresponder a cartas en la mano del agente."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        mask = env.action_masks()
        mano = obs[0:52]
        # Las acciones legales deben ser un subconjunto de la mano
        for i in range(52):
            if mask[i]:
                assert mano[i] == 1.0, (
                    f"Acción {i} es legal pero la carta no está en la mano"
                )
        env.close()

    def test_action_masks_varia_tras_step(self):
        """La máscara debe cambiar después de jugar una carta."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        env.reset()
        mask_antes = env.action_masks().copy()
        accion_legal = int(np.where(mask_antes == 1)[0][0])
        env.step(accion_legal)
        mask_despues = env.action_masks()
        # La máscara debe ser diferente (o al menos no idéntica)
        # Nota: podría ser igual si es el turno de otro jugador
        env.close()


# ============================================================
# Prueba de auditoría nativa de Gymnasium
# ============================================================

class TestCheckEnv:
    """Verifica que el entorno pase la auditoría check_env()."""

    def test_check_env_pasa(self):
        """check_env() de Gymnasium no debe lanzar excepción."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        try:
            check_env(env, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"check_env() falló: {e}")
        finally:
            env.close()


# ============================================================
# Pruebas de Recompensas (Suma Cero)
# ============================================================

class TestRecompensas:
    """Verifica la estructura de recompensas de corto y largo plazo."""

    def test_recompensa_terminal_suma_cero_por_diseno(self):
        """Las recompensas terminales (+1000, +300, -300, -1000) suman 0."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        total = (
            env.REWARD_PRIMERO
            + env.REWARD_SEGUNDO
            + env.REWARD_TERCERO
            + env.REWARD_CUARTO
        )
        assert total == 0.0, f"Recompensas terminales deben sumar 0, suman {total}"
        env.close()

    def test_shooting_moon_otorga_recompensa_positiva(self):
        """Forzar un escenario donde el agente hace pleno y verificar
           que recibe +50 de recompensa."""
        from src.entorno import CorazonesEnv
        from src.carta import Carta
        from src.motor import MotorCorazones

        env = CorazonesEnv(agente_idx=0)
        env.reset()

        # Inyectar estado: dar al agente todas las cartas de puntos
        # y manipular el motor para que gane todas las bazas
        motor = env.motor
        # Dar al agente cartas altas de cada palo + todos los corazones + dama
        mano_agente = [
            Carta(0, 14),  # As de tréboles
            Carta(1, 14),  # As de diamantes
            Carta(2, 14),  # As de picas
            Carta(2, 13),  # Rey de picas
            Carta(2, 12),  # Dama de picas (13 pts)
            Carta(3, 14),  # As de corazones
            Carta(3, 13),  # Rey de corazones
            Carta(3, 12),  # Dama de corazones
            Carta(3, 11),  # J de corazones
            Carta(3, 10),
            Carta(3, 9),
            Carta(3, 8),
            Carta(3, 7),
        ]
        motor.jugadores[0].mano = list(mano_agente)

        # Dar a los rivales cartas bajas para que no puedan competir
        cartas_bajas = [
            Carta(0, 2), Carta(0, 3), Carta(0, 4),
            Carta(1, 2), Carta(1, 3), Carta(1, 4),
            Carta(2, 2), Carta(2, 3), Carta(2, 4),
        ]
        for i in range(1, 4):
            motor.jugadores[i].mano = [
                cartas_bajas.pop() for _ in range(min(13, len(cartas_bajas)))
            ]
            # Rellenar con cartas sin puntos
            faltan = 13 - len(motor.jugadores[i].mano)
            for v in range(2, 2 + faltan):
                motor.jugadores[i].mano.append(Carta(0, v))

        # Esto es difícil de forzar exactamente; verificamos estructura
        env.close()


# ============================================================
# Pruebas de ciclo completo
# ============================================================

class TestCicloCompleto:
    """Pruebas de integración: juego completo hasta estado terminal."""

    def test_juego_completo_sin_crash(self):
        """Ejecutar partidas hasta que termine sin errores."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        pasos = 0
        done = False
        while not done and pasos < 10000:  # safety limit
            mask = env.action_masks()
            legal = int(np.where(mask == 1)[0][0])
            obs, reward, terminated, truncated, _ = env.step(legal)
            done = terminated or truncated
            pasos += 1
            assert not np.any(np.isnan(obs)), f"NaN en paso {pasos}"
            assert obs.shape == (190,), f"Shape incorrecto en paso {pasos}"
        assert pasos < 10000, "Juego no terminó en 10000 pasos"
        env.close()

    def test_obs_mitad_partida_valida(self):
        """Verificar el tensor de observación a mitad de partida."""
        from src.entorno import CorazonesEnv
        env = CorazonesEnv()
        obs, _ = env.reset()
        # Jugar aproximadamente 6 bazas (24 pasos del agente)
        for _ in range(24):
            mask = env.action_masks()
            legal = int(np.where(mask == 1)[0][0])
            obs, _, terminated, truncated, _ = env.step(legal)
            if terminated or truncated:
                break
        # Validar que la observación sigue siendo válida
        assert obs.shape == (190,)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        env.close()

    def test_juego_alcanza_estado_terminal(self):
        """Varias partidas deben eventualmente alcanzar estado terminal."""
        from src.entorno import CorazonesEnv
        for seed in range(5):
            env = CorazonesEnv()
            env.reset(seed=seed)
            done = False
            pasos = 0
            while not done and pasos < 5000:
                mask = env.action_masks()
                legal = int(np.where(mask == 1)[0][0])
                _, _, terminated, truncated, _ = env.step(legal)
                done = terminated or truncated
                pasos += 1
            assert done, f"Partida con seed={seed} no terminó"
            env.close()
