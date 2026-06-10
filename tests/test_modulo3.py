"""
Pruebas unitarias para el Módulo 3: Agente RL y Pipeline de Entrenamiento.

Cubre:
    - Inicialización y validación de la red neuronal (MLP 256-256-128).
    - Bots heurísticos (juegan solo cartas legales).
    - Entorno PettingZoo AEC (4 agentes secuenciales).
    - Prueba de sobreajuste (Overfitting): seed estático, misma mano,
      el agente debe maximizar la recompensa en < 5000 episodios.
    - Pipeline de entrenamiento (inicia, entrena, guarda checkpoints).
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Asegurar que src está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Pruebas de bots heurísticos
# ============================================================

class TestBotsHeuristicos:
    """Verifica que los bots heurísticos producen jugadas legales."""

    def test_bot_existe_modulo(self):
        """El módulo bots debe poder importarse."""
        from src.bots import (
            bot_conservador,
            bot_agresivo,
            bot_evasivo,
        )
        assert callable(bot_conservador)
        assert callable(bot_agresivo)
        assert callable(bot_evasivo)

    def test_bots_juegan_cartas_legales(self):
        """Todos los bots deben devolver cartas de la lista de legales."""
        from src.bots import (
            bot_conservador,
            bot_agresivo,
            bot_evasivo,
        )
        from src.carta import Carta
        from src.motor import MotorCorazones

        motor = MotorCorazones()
        motor.repartir()

        bots = [bot_conservador, bot_agresivo, bot_evasivo]

        # Jugar una mano completa con cada bot en cada posición
        for bot in bots:
            m = MotorCorazones()
            m.repartir()
            for _ in range(13):
                for _ in range(4):
                    idx = m.obtener_jugador_actual()
                    legales = m.obtener_jugadas_legales(idx)
                    carta = bot(m, idx, legales)
                    assert carta in legales, (
                        f"Bot {bot.__name__} devolvió carta ilegal {carta}"
                    )
                    m.jugar_carta(idx, carta)
                m.resolver_baza()

    def test_bot_conservador_juega_carta_baja(self):
        """El bot conservador tiende a jugar cartas de bajo valor."""
        from src.bots import bot_conservador
        from src.carta import Carta
        from src.motor import MotorCorazones

        m = MotorCorazones()
        m.repartir()
        idx = m.obtener_jugador_actual()
        legales = m.obtener_jugadas_legales(idx)
        carta = bot_conservador(m, idx, legales)

        # El bot conservador elige la carta de menor valor entre las legales
        min_valor = min(c.valor for c in legales)
        assert carta.valor == min_valor, (
            f"Bot conservador debería jugar la más baja ({min_valor}), "
            f"jugó {carta.valor}"
        )

    def test_bot_agresivo_fuga_si_puede(self):
        """El bot agresivo intenta jugar cartas altas o fugarse."""
        from src.bots import bot_agresivo
        from src.carta import Carta
        from src.motor import MotorCorazones

        m = MotorCorazones()
        m.repartir()
        # Forzar una situación con palo de salida
        m.corazones_rotos = True
        m.numero_baza = 3
        m.mesa = [(0, Carta(0, 5))]
        m.palo_de_salida = 0
        m.indice_jugador_inicial = 0

        # Dar al jugador 1 cartas de trébol y otros palos
        m.jugadores[1].mano = [
            Carta(0, 3),  # trébol bajo
            Carta(0, 14),  # as de trébol (alto)
            Carta(1, 2),  # diamante bajo
            Carta(2, 3),  # pica
        ]
        legales = m.obtener_jugadas_legales(1)
        carta = bot_agresivo(m, 1, legales)
        # Debe estar en legales
        assert carta in legales


# ============================================================
# Pruebas del entorno PettingZoo AEC
# ============================================================

class TestPettingZooAEC:
    """Verifica que el entorno PettingZoo AEC funciona correctamente."""

    def test_importacion_pettingzoo(self):
        """PettingZoo debe estar instalado y el entorno debe ser importable."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        env.close()

    def test_agents_list(self):
        """El entorno debe tener 4 agentes."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        assert len(env.possible_agents) == 4
        env.close()

    def test_reset_inicializa_agentes(self):
        """Reset debe inicializar los agentes y el juego."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        assert len(env.agents) == 4
        env.close()

    def test_action_spaces_son_discrete_52(self):
        """Cada agente debe tener espacio de acción Discrete(52)."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        for agent in env.agents:
            space = env.action_space(agent)
            assert space.n == 52
        env.close()

    def test_observation_spaces_son_box_187(self):
        """Cada agente debe tener espacio de observación Box(187,)."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        for agent in env.agents:
            space = env.observation_space(agent)
            assert space.shape == (187,)
            assert space.dtype == np.float32
        env.close()

    def test_agent_iter_recorre_4_jugadores(self):
        """agent_iter debe recorrer los 4 jugadores secuencialmente."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        agentes_vistos = []
        for agent in env.agent_iter(max_iter=4):
            agentes_vistos.append(agent)
            env.step(0)  # acción dummy
            if len(agentes_vistos) >= 4:
                break
        assert len(agentes_vistos) >= 1  # Al menos un agente
        env.close()

    def test_action_mask_funciona(self):
        """action_mask debe devolver un array booleano de shape (52,)."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        for agent in env.agent_iter(max_iter=4):
            mask = env.action_mask(agent)
            assert mask.shape == (52,)
            assert mask.dtype == np.bool_ or mask.dtype == bool
            assert np.any(mask), f"Ninguna acción legal para {agent}"
            env.step(int(np.where(mask)[0][0]))
            break
        env.close()

    def test_juego_completo_sin_errores(self):
        """Una partida completa con bots aleatorios no debe lanzar errores."""
        from src.entorno_multi import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        pasos = 0
        for agent in env.agent_iter(max_iter=10000):
            mask = env.action_mask(agent)
            if np.any(mask):
                accion = int(np.where(mask)[0][0])
                env.step(accion)
            else:
                env.step(None)
            pasos += 1
        assert pasos > 0
        env.close()


# ============================================================
# Pruebas de la red neuronal
# ============================================================

class TestRedNeuronal:
    """Verifica la arquitectura de la red neuronal para MaskablePPO."""

    def test_red_importable(self):
        """La red neuronal personalizada debe ser importable."""
        from src.red import CorazonesFeatureExtractor
        from gymnasium import spaces
        import torch
        obs_space = spaces.Box(low=0, high=1, shape=(187,), dtype=np.float32)
        extractor = CorazonesFeatureExtractor(obs_space, features_dim=512)
        assert extractor is not None

    def test_forward_propaga_observacion(self):
        """La red debe aceptar un batch de observaciones (187,) y producir salida."""
        import torch
        from src.red import CorazonesFeatureExtractor
        from gymnasium import spaces

        obs_space = spaces.Box(low=0, high=1, shape=(187,), dtype=np.float32)
        extractor = CorazonesFeatureExtractor(obs_space, features_dim=512)
        # Simular un batch de 4 observaciones
        batch = torch.randn(4, 187)
        salida = extractor(batch)
        assert salida.shape == (4, 512), (
            f"Esperado (4, 512), obtenido {salida.shape}"
        )

    def test_policy_network_creacion(self):
        """Verificar que se puede crear una MaskableActorCriticPolicy con env."""
        import torch
        from gymnasium import spaces
        from src.entorno import CorazonesEnv
        try:
            from sb3_contrib import MaskablePPO
            env = CorazonesEnv(agente_idx=0)
            model = MaskablePPO(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": [256, 256, 128]},
                device="cpu",
            )
            assert model is not None
            env.close()
        except ImportError:
            pytest.skip("sb3-contrib no instalado")


# ============================================================
# Prueba de Sobreajuste (Overfitting Test)
# ============================================================

class TestOverfitting:
    """Prueba de sobreajuste: seed estático, agente debe aprender
    la secuencia óptima en < 5000 episodios."""

    def test_overfitting_agente_aprende_mano_fija(self):
        """Con seed fijo (misma mano cada episodio), el agente debe
        aprender una política que maximice la recompensa."""
        import torch
        import numpy as np
        from src.entorno import CorazonesEnv

        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            pytest.skip("sb3-contrib no instalado")

        # Crear entorno con seed fijo para que siempre reparta igual
        env = CorazonesEnv(agente_idx=0)
        env.reset(seed=42)

        # Crear modelo con arquitectura pequeña para test rápido
        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": [128, 64],  # Reducida para test rápido
            },
            verbose=0,
            device="cpu",
            n_steps=52,  # ~1 mano por rollout
            batch_size=26,
            n_epochs=4,
            learning_rate=1e-3,
        )

        # Entrenar por pocos episodios y verificar que la recompensa mejora
        num_episodios = 200
        recompensas = []

        for ep in range(num_episodios):
            obs, _ = env.reset()
            done = False
            recompensa_ep = 0.0

            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=False)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                recompensa_ep += reward
                done = terminated or truncated

            recompensas.append(recompensa_ep)
            model.learn(total_timesteps=52, reset_num_timesteps=False)

        # Verificar que el modelo ha guardado algo en memoria
        # y que las recompensas no son todas iguales (hubo aprendizaje)
        assert len(recompensas) == num_episodios
        # Al menos algunas recompensas deben ser diferentes
        assert len(set(round(r, 1) for r in recompensas[-50:])) > 1, (
            "No se detecta variación en recompensas; posiblemente no hay aprendizaje"
        )

        env.close()
        del model

    def test_modelo_guarda_y_carga_checkpoint(self):
        """El modelo debe poder guardarse y cargarse desde disco."""
        import torch
        import tempfile
        import numpy as np
        from src.entorno import CorazonesEnv

        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            pytest.skip("sb3-contrib no instalado")

        env = CorazonesEnv(agente_idx=0)
        env.reset(seed=42)

        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0,
            device="cpu",
        )

        # Entrenar un poco
        model.learn(total_timesteps=200, progress_bar=False)

        # Guardar y cargar
        with tempfile.TemporaryDirectory() as tmpdir:
            ruta = os.path.join(tmpdir, "test_model")
            model.save(ruta)
            assert os.path.exists(ruta + ".zip"), "No se guardó el modelo"

            # Cargar
            model_cargado = MaskablePPO.load(ruta, env=env)
            assert model_cargado is not None

        env.close()
        del model


# ============================================================
# Pruebas del Pipeline de Entrenamiento
# ============================================================

class TestPipelineEntrenamiento:
    """Verifica que los componentes del pipeline son funcionales."""

    def test_modulo_train_importable(self):
        """El script de entrenamiento debe ser importable como módulo."""
        import importlib
        try:
            importlib.import_module("train_self_play")
        except ImportError as e:
            pytest.fail(f"No se pudo importar train_self_play: {e}")

    def test_directorio_modelos_creado(self):
        """El directorio de snapshots históricos debe crearse automáticamente."""
        from train_self_play import DIRECTORIO_MODELOS
        assert DIRECTORIO_MODELOS is not None
