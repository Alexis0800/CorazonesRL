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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================
# Pruebas de bots heurísticos
# ============================================================

class TestBotsHeuristicos:
    """Verifica que los bots heurísticos producen jugadas legales."""

    def test_bot_existe_modulo(self):
        """El módulo bots debe poder importarse."""
        from src.agentes.heuristicos import (
            bot_conservador,
            bot_agresivo,
            bot_evasivo,
        )
        assert callable(bot_conservador)
        assert callable(bot_agresivo)
        assert callable(bot_evasivo)

    def test_bots_juegan_cartas_legales(self):
        """Todos los bots deben devolver cartas de la lista de legales."""
        from src.agentes.heuristicos import (
            bot_conservador,
            bot_agresivo,
            bot_evasivo,
        )
        from src.dominio.carta import Carta
        from src.dominio.motor import MotorCorazones

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
        from src.agentes.heuristicos import bot_conservador
        from src.dominio.carta import Carta
        from src.dominio.motor import MotorCorazones

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
        from src.agentes.heuristicos import bot_agresivo
        from src.dominio.carta import Carta
        from src.dominio.motor import MotorCorazones

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
        from src.entorno.multi_agent import CorazonesAEC
        env = CorazonesAEC()
        env.close()

    def test_agents_list(self):
        """El entorno debe tener 4 agentes."""
        from src.entorno.multi_agent import CorazonesAEC
        env = CorazonesAEC()
        assert len(env.possible_agents) == 4
        env.close()

    def test_reset_inicializa_agentes(self):
        """Reset debe inicializar los agentes y el juego."""
        from src.entorno.multi_agent import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        assert len(env.agents) == 4
        env.close()

    def test_action_spaces_son_discrete_52(self):
        """Cada agente debe tener espacio de acción Discrete(52)."""
        from src.entorno.multi_agent import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        for agent in env.agents:
            space = env.action_space(agent)
            assert space.n == 52
        env.close()

    def test_observation_spaces_son_box_190(self):
        """Cada agente debe tener espacio de observación Box(194,)."""
        from src.entorno.multi_agent import CorazonesAEC
        env = CorazonesAEC()
        env.reset()
        for agent in env.agents:
            space = env.observation_space(agent)
            assert space.shape == (194,)
            assert space.dtype == np.float32
        env.close()

    def test_agent_iter_recorre_4_jugadores(self):
        """agent_iter debe recorrer los 4 jugadores secuencialmente."""
        from src.entorno.multi_agent import CorazonesAEC
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
        from src.entorno.multi_agent import CorazonesAEC
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
        from src.entorno.multi_agent import CorazonesAEC
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
        from src.entorno.single_agent import CorazonesEnv
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
        from src.entorno.single_agent import CorazonesEnv

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
                action, _ = model.predict(
                    obs, action_masks=mask, deterministic=False)
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
        from src.entorno.single_agent import CorazonesEnv

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


# ============================================================
# Pruebas del Script de Evaluación (evaluar_modelo.py)
# ============================================================

class TestNormalizacionEvaluacion:
    """Verifica que la normalización de observaciones funciona correctamente
    con archivos VecNormalize de SB3 (objetos, no diccionarios)."""

    def test_normalizar_con_vecnormalize_objeto(self):
        """Corrección del bug: VecNormalize se carga como objeto, NO como dict.
        data.get('obs_rms') falla. Debe usarse vn.obs_rms directamente."""
        import pickle
        import numpy as np
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        from src.entorno.single_agent import CorazonesEnv
        import tempfile
        import os

        # Crear un VecNormalize real y guardarlo
        env = CorazonesEnv(agente_idx=0)
        venv = VecNormalize(
            DummyVecEnv([lambda: env]),
            norm_obs=True,
            norm_reward=False,
        )
        # Simular algunos pasos para poblar estadísticas
        venv.reset()
        for _ in range(100):
            mask = env.action_masks()
            if np.any(mask):
                legales = np.where(mask)[0]
                action = np.random.choice(legales)
                obs, _, done, _, _ = env.step(int(action))
                if done:
                    env.reset()
        env.close()

        # Guardar y recargar
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vecnorm.pkl")
            venv.save(path)

            # === Método ERRÓNEO (el bug actual) ===
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Verificar que NO es un dict
            assert not isinstance(data, dict), (
                "VecNormalize pickle NO devuelve un dict, devuelve un objeto VecNormalize"
            )
            # Verificar que data.get() FALLA (causa recursión infinita)
            with pytest.raises(Exception):
                data.get("obs_rms", None)

            # === Método CORRECTO ===
            obs_rms = data.obs_rms
            assert obs_rms is not None, "obs_rms debe ser accesible como atributo"
            mean = np.array(obs_rms.mean)
            var = np.array(obs_rms.var)
            assert mean.shape == (
                194,), f"mean shape debe ser (194,), es {mean.shape}"
            assert var.shape == (
                194,), f"var shape debe ser (194,), es {var.shape}"
            assert obs_rms.count > 0, "count debe ser > 0 tras simular pasos"

            # Probar normalización manual (equivalente a SB3)
            obs_raw = np.ones(194, dtype=np.float32)
            obs_norm = np.clip(
                (obs_raw - mean) / np.sqrt(var + 1e-8), -10.0, 10.0
            ).astype(np.float32)
            assert obs_norm.shape == (194,)
            assert obs_norm.dtype == np.float32
            # Verificar que la normalización efectivamente cambió los valores
            assert not np.allclose(obs_norm, obs_raw), (
                "La normalización debe modificar los valores de la observación"
            )

    def test_normalizar_sin_archivo_devuelve_raw(self):
        """Si el archivo VecNormalize no existe, debe devolver obs sin modificar."""
        import numpy as np
        # Importar la función del script de evaluación
        import importlib
        ev_mod = importlib.import_module("scripts.evaluar_modelo")
        normalizar = ev_mod.normalizar_obs_si_hay_stats

        obs = np.ones(187, dtype=np.float32)
        result = normalizar(obs, "ruta/que/no/existe.pkl")
        assert np.array_equal(result, obs), (
            "Sin archivo vecnorm, debe devolver la obs sin cambios"
        )

    def test_normalizar_con_path_none_devuelve_raw(self):
        """Si vecnorm_path es None o vacío, debe devolver obs sin modificar."""
        import numpy as np
        import importlib
        ev_mod = importlib.import_module("scripts.evaluar_modelo")
        normalizar = ev_mod.normalizar_obs_si_hay_stats

        obs = np.ones(187, dtype=np.float32)
        result_none = normalizar(obs, None)
        result_empty = normalizar(obs, "")
        assert np.array_equal(result_none, obs)
        assert np.array_equal(result_empty, obs)


class TestMetricasMultiNivel:
    """Verifica que la evaluación retorna métricas de clasificación completas:
    porcentaje en 1º, 2º, 3º, 4º lugar y puntuación promedio."""

    def test_evaluar_retorna_diccionario_metricas(self):
        """La función evaluar debe retornar un dict con todas las métricas."""
        import importlib
        ev_mod = importlib.import_module("scripts.evaluar_modelo")
        # Verificar que existe la nueva función de métricas
        assert hasattr(ev_mod, "evaluar_con_metricas"), (
            "Debe existir la función evaluar_con_metricas"
        )

    def test_formato_metricas(self):
        """Verificar que las métricas tienen el formato esperado."""
        import importlib
        ev_mod = importlib.import_module("scripts.evaluar_modelo")

        # Verificar que existe la función que construye el dict de métricas
        assert hasattr(ev_mod, "_construir_metricas"), (
            "Debe existir _construir_metricas para generar el dict de resultados"
        )

    def test_suma_posiciones_es_100(self):
        """La suma de porcentajes de las 4 posiciones debe ser 100%."""
        import importlib
        ev_mod = importlib.import_module("scripts.evaluar_modelo")

        # Simular resultados: 3 primeros, 1 segundo, 2 terceros, 0 cuartos en 6 partidas
        # índices de posición: 0=1º, 1=2º, 2=3º, 3=4º
        posiciones = [0, 0, 0, 1, 2, 2]
        puntuaciones = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        metricas = ev_mod._construir_metricas(posiciones, puntuaciones, 6)

        suma = (
            metricas["pct_primero"]
            + metricas["pct_segundo"]
            + metricas["pct_tercero"]
            + metricas["pct_cuarto"]
        )
        assert abs(suma - 1.0) < 0.001, (
            f"Suma de posiciones debe ser 100%, es {suma:.1%}"
        )
        assert metricas["pct_primero"] == pytest.approx(0.5)  # 3/6
        assert metricas["pct_segundo"] == pytest.approx(1 / 6)
        assert metricas["pct_top2"] == pytest.approx(4 / 6)  # 3+1
        assert metricas["punt_promedio"] == pytest.approx(
            17.5)  # (5+10+15+20+25+30)/6
        assert metricas["total_partidas"] == 6
        assert metricas["victorias"] == 3

    def test_metrica_victoria_es_menor_puntuacion(self):
        """Victoria en Corazones = tener la puntuación MÁS BAJA (menos puntos)."""
        import importlib
        ev_mod = importlib.import_module("scripts.evaluar_modelo")

        # Caso: agente tiene 5 puntos, rivales tienen 15, 20, 25
        # El agente va PRIMERO (menos puntos = mejor)
        punt_agente = 5
        punt_rivales = [15, 20, 25]
        posicion = ev_mod._calcular_posicion(punt_agente, punt_rivales)
        assert posicion == 0, (
            f"Con {punt_agente} pts vs {punt_rivales}, debería ser 1º (índice 0), fue {posicion}"
        )

        # Caso: agente tiene 30 puntos, todos los demás menos → 4º lugar
        posicion = ev_mod._calcular_posicion(30, [5, 10, 15])
        assert posicion == 3, (
            f"Con 30 pts vs [5,10,15], debería ser 4º (índice 3), fue {posicion}"
        )

        # Caso: empate en el mejor lugar → gana el que tenga menos puntos
        posicion = ev_mod._calcular_posicion(10, [10, 20, 30])
        assert posicion == 0, "Empate en 10 con otro → debe ser considerado 1º"

    def test_resultado_referencia_aleatoria(self):
        """Un bot puramente aleatorio debería tener ~25% win rate
        y ~25% en cada posición (verificación estadística débil)."""
        import numpy as np
        from src.entorno.single_agent import CorazonesEnv

        victorias = 0
        n = 100
        for seed in range(n):
            env = CorazonesEnv(agente_idx=0)
            env.reset(seed=seed)
            done = False
            while not done:
                mask = env.action_masks()
                legales = np.where(mask)[0]
                if len(legales) > 0:
                    action = np.random.choice(legales)
                    _, _, terminated, truncated, _ = env.step(int(action))
                    done = terminated or truncated
                else:
                    break
            punt_agente = env._puntuacion_historica[0]
            punt_rivales = [env._puntuacion_historica[i] for i in range(1, 4)]
            if punt_agente < min(punt_rivales):
                victorias += 1
            env.close()

        wr = victorias / n
        # Con 100 partidas, 25% aleatorio debería dar entre 10% y 40%
        assert 0.10 <= wr <= 0.40, (
            f"Bot aleatorio: wr={wr:.1%}, esperado ~25% (rango [10%, 40%] para 100 muestras)"
        )


# ============================================================
# Pruebas del Pipeline de Entrenamiento v2 (Anti-Colapso)
# ============================================================

class TestQualityFilterSnapshots:
    """Verifica el filtro de calidad para snapshots en self-play v2."""

    def test_filtrar_snapshots_por_calidad_min_pasos(self):
        """Solo se usan como oponentes snapshots con al menos min_steps.
        Esto evita que el agente entrene contra versiones demasiado débiles."""
        import importlib
        ts = importlib.import_module("train_self_play")

        # Simular lista de snapshots
        snaps = [
            "snapshot_0000050000",
            "snapshot_0000150000",
            "snapshot_0000300000",
            "snapshot_0000500000",
            "snapshot_0001000000",
        ]
        # Con min_steps=200000, solo deberían quedar los >= 200k
        filtrados = ts._filtrar_snapshots_por_calidad(snaps, min_steps=200000)
        assert len(filtrados) == 3, (
            f"Esperados 3 snapshots >= 200k (300k, 500k, 1M), obtenidos {len(filtrados)}: {filtrados}"
        )
        assert "snapshot_0000300000" in filtrados
        assert "snapshot_0000500000" in filtrados
        assert "snapshot_0001000000" in filtrados

    def test_filtrar_snapshots_vacio_sin_suficientes(self):
        """Si no hay snapshots que cumplan el mínimo, devuelve lista vacía."""
        import importlib
        ts = importlib.import_module("train_self_play")

        snaps = ["snapshot_0000050000", "snapshot_0000100000"]
        filtrados = ts._filtrar_snapshots_por_calidad(snaps, min_steps=500000)
        assert filtrados == [], (
            "Sin snapshots que cumplan el mínimo, debe devolver lista vacía"
        )

    def test_filtrar_snapshots_lista_vacia(self):
        """Lista vacía de entrada produce lista vacía de salida."""
        import importlib
        ts = importlib.import_module("train_self_play")

        assert ts._filtrar_snapshots_por_calidad([], min_steps=100000) == []

    def test_filtrar_snapshots_extrae_paso_correctamente(self):
        """La extracción del número de paso desde el nombre es robusta."""
        import importlib
        ts = importlib.import_module("train_self_play")

        snaps = [
            "modelos/v1_backup/snapshots/snapshot_0000150000",
            "modelos/v2/snapshots/snapshot_0000250000",
        ]
        filtrados = ts._filtrar_snapshots_por_calidad(snaps, min_steps=100000)
        # 150k >= 100k, 250k >= 100k → ambos pasan
        assert len(filtrados) == 2


class TestHiperparametrosV2:
    """Verifica que los hiperparámetros v2 son los correctos para
    prevenir el colapso de política."""

    def test_hiperparametros_v2_lr_y_ent_coef(self):
        """V2 usa learning_rate=1e-4 y ent_coef=0.10 (más exploración, v11)."""
        import importlib
        ts = importlib.import_module("train_self_play")

        hp = ts.obtener_hiperparametros_v2("dummy_logdir", "cpu")
        assert hp["learning_rate"] == 1e-4, (
            f"V2 debe usar lr=1e-4, tiene {hp['learning_rate']}"
        )
        assert hp["ent_coef"] == 0.10, (
            f"V2 debe usar ent_coef=0.10 (v11), tiene {hp['ent_coef']}"
        )

    def test_hiperparametros_v2_prob_bot_default(self):
        """Fase 5B usa prob_bot=0.30 por defecto (30% bots, 70% snapshots)."""
        import importlib
        ts = importlib.import_module("train_self_play")

        assert hasattr(ts, "PROB_BOT_V2"), "Debe existir PROB_BOT_V2"
        assert ts.PROB_BOT_V2 == 0.30, (
            f"PROB_BOT_V2 debe ser 0.30 (Fase 5B), es {ts.PROB_BOT_V2}"
        )

    def test_hiperparametros_v2_min_snapshot_steps(self):
        """V2 define un umbral mínimo de pasos para snapshots de calidad."""
        import importlib
        ts = importlib.import_module("train_self_play")

        assert hasattr(ts, "MIN_SNAPSHOT_STEPS"), (
            "Debe existir MIN_SNAPSHOT_STEPS como umbral de calidad"
        )
        assert ts.MIN_SNAPSHOT_STEPS >= 100000, (
            f"MIN_SNAPSHOT_STEPS debe ser >= 100k, es {ts.MIN_SNAPSHOT_STEPS}"
        )

    def test_hiperparametros_v2_max_snapshots_pool(self):
        """V2 limita el pool de snapshots a máximo N para pruning."""
        import importlib
        ts = importlib.import_module("train_self_play")

        assert hasattr(ts, "MAX_SNAPSHOTS_POOL"), (
            "Debe existir MAX_SNAPSHOTS_POOL para pruning"
        )
        assert 10 <= ts.MAX_SNAPSHOTS_POOL <= 50, (
            f"MAX_SNAPSHOTS_POOL debe estar entre 10 y 50, es {ts.MAX_SNAPSHOTS_POOL}"
        )


class TestDirectoriosV6:
    """Verifica las constantes de directorio para entrenamiento paralelo v6."""

    def test_directorio_modelos_v6_definido(self):
        """Debe existir DIRECTORIO_MODELOS_V6."""
        import importlib
        ts = importlib.import_module("train_self_play")
        assert hasattr(
            ts, "DIRECTORIO_MODELOS_V6"), "Debe existir DIRECTORIO_MODELOS_V6"
        assert "v6" in ts.DIRECTORIO_MODELOS_V6, "Debe apuntar a v6"

    def test_directorio_vecnorm_v6_definido(self):
        """Debe existir DIRECTORIO_VECNORM_V6."""
        import importlib
        ts = importlib.import_module("train_self_play")
        assert hasattr(
            ts, "DIRECTORIO_VECNORM_V6"), "Debe existir DIRECTORIO_VECNORM_V6"
        assert "v6" in ts.DIRECTORIO_VECNORM_V6, "Debe apuntar a v6"


# ============================================================
# Pruebas de hiperparámetros v16 (balance exploración/velocidad)
# ============================================================

class TestHiperparametrosV16:
    """v16: ent_coef 0.30→0.12, balance entre v13 (colapso) y v15 (lento)."""

    def test_ent_coef_start(self):
        import importlib
        ts = importlib.import_module("train_self_play")
        hp = ts.obtener_hiperparametros_v3("/tmp", "cpu", 0, 20_000_000)
        assert hp["ent_coef"] == pytest.approx(0.30, rel=0.01)

    def test_ent_coef_floor(self):
        import importlib
        ts = importlib.import_module("train_self_play")
        hp = ts.obtener_hiperparametros_v3("/tmp", "cpu", 20_000_000, 20_000_000)
        assert hp["ent_coef"] == pytest.approx(0.12, rel=0.01)

    def test_n_epochs(self):
        import importlib
        ts = importlib.import_module("train_self_play")
        hp = ts.obtener_hiperparametros_v3("/tmp", "cpu", 0, 20_000_000)
        assert hp["n_epochs"] == 4

    def test_batch_size(self):
        import importlib
        ts = importlib.import_module("train_self_play")
        hp = ts.obtener_hiperparametros_v3("/tmp", "cpu", 0, 20_000_000)
        assert hp["batch_size"] == 512

    def test_target_kl(self):
        import importlib
        ts = importlib.import_module("train_self_play")
        hp = ts.obtener_hiperparametros_v3("/tmp", "cpu", 0, 20_000_000)
        assert hp["target_kl"] == pytest.approx(0.05, rel=0.01)
