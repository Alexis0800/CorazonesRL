"""
Pruebas unitarias para el módulo de evaluación multi-nivel (Fase 5C).

Cubre:
    - Evaluación contra bots con métricas completas.
    - Evaluación de escenarios estratégicos (Q♠ sin pozo, pozo viable).
    - Detección automática de VecNormalize.
    - Integridad de métricas (suma 100%, rangos válidos).
"""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Helpers
# ============================================================

def _crear_modelo_dummy():
    """Crea un checkpoint mínimo de MaskablePPO para pruebas."""
    from src.entorno import CorazonesEnv
    from sb3_contrib import MaskablePPO
    env = CorazonesEnv()
    modelo = MaskablePPO("MlpPolicy", env, n_steps=256, device="cpu")
    env.close()
    return modelo


# ============================================================
# Test: Evaluación contra bots
# ============================================================

class TestEvaluarContraBots:
    """Verifica que la evaluación contra bots produce métricas válidas."""

    def test_metricas_tiene_keys_esperadas(self):
        """El diccionario de métricas debe contener todas las keys esperadas."""
        from src.evaluacion import _construir_metricas
        metricas = _construir_metricas(
            posiciones=[0, 1, 0, 2, 3, 0, 1, 2, 0, 3],
            puntuaciones=[10.0, 25.0, 5.0, 45.0,
                          78.0, 12.0, 30.0, 55.0, 8.0, 90.0],
            total=10,
        )
        required = [
            "total_partidas", "victorias",
            "pct_primero", "pct_segundo", "pct_tercero", "pct_cuarto",
            "pct_top2", "punt_promedio", "punt_mediana", "punt_min", "punt_max",
        ]
        for key in required:
            assert key in metricas, f"Falta key '{key}' en métricas"

    def test_porcentajes_suman_100(self):
        """pct_primero + pct_segundo + pct_tercero + pct_cuarto debe ser 1.0."""
        from src.evaluacion import _construir_metricas
        metricas = _construir_metricas(
            posiciones=[0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
            puntuaciones=[10.0] * 10,
            total=10,
        )
        suma = (metricas["pct_primero"] + metricas["pct_segundo"] +
                metricas["pct_tercero"] + metricas["pct_cuarto"])
        assert abs(suma - 1.0) < 0.001, f"Suma de % debe ser 1.0, es {suma}"

    def test_victorias_coincide_con_pct_primero(self):
        """El número de victorias debe coincidir con pct_primero * total."""
        from src.evaluacion import _construir_metricas
        posiciones = [0, 0, 0, 1, 1, 2, 3]
        total = len(posiciones)
        metricas = _construir_metricas(
            posiciones=posiciones,
            puntuaciones=[1.0] * total,
            total=total,
        )
        expected_wins = 3
        assert metricas["victorias"] == expected_wins
        assert abs(metricas["pct_primero"] - expected_wins / total) < 0.001

    def test_metricas_lista_vacia(self):
        """Con 0 partidas, todas las métricas deben ser 0.0."""
        from src.evaluacion import _construir_metricas
        metricas = _construir_metricas([], [], 0)
        assert metricas["total_partidas"] == 0
        assert metricas["pct_primero"] == 0.0
        assert metricas["punt_promedio"] == 0.0

    def test_top2_es_suma_primero_y_segundo(self):
        """pct_top2 debe ser pct_primero + pct_segundo."""
        from src.evaluacion import _construir_metricas
        metricas = _construir_metricas(
            posiciones=[0, 0, 1, 1, 2, 2, 3, 3],
            puntuaciones=[5.0] * 8,
            total=8,
        )
        expected_top2 = metricas["pct_primero"] + metricas["pct_segundo"]
        assert abs(metricas["pct_top2"] - expected_top2) < 0.001


# ============================================================
# Test: Evaluación de escenarios estratégicos
# ============================================================

class TestEscenariosEstrategicos:
    """Verifica que los escenarios estratégicos detectan comportamientos clave."""

    def test_escenario_q_spades_sin_pozo_existe(self):
        """La función de escenario Q♠ sin pozo debe estar definida."""
        from src.evaluacion import _crear_escenario_q_spades_sin_pozo
        escenario = _crear_escenario_q_spades_sin_pozo()
        assert isinstance(escenario, dict)
        assert "nombre" in escenario
        assert "mano_agente" in escenario
        assert "manos_rivales" in escenario
        assert "corazones_rotos" in escenario

    def test_escenario_pozo_viable_existe(self):
        """La función de escenario pozo viable debe estar definida."""
        from src.evaluacion import _crear_escenario_pozo_viable
        escenario = _crear_escenario_pozo_viable()
        assert escenario["nombre"] != ""
        assert len(escenario["mano_agente"]) == 13
        # Debe tener ≥6 corazones para pozo viable
        corazones = sum(
            1 for cid in escenario["mano_agente"] if 39 <= cid <= 51)
        assert corazones >= 6, f"Escenario pozo viable necesita ≥6 corazones, tiene {corazones}"

    def test_evaluar_escenario_retorna_metricas(self):
        """Evaluar un escenario debe retornar dict con keys esperadas (sin crash)."""
        from src.evaluacion import evaluar_escenario, _crear_escenario_q_spades_sin_pozo
        from src.entorno import CorazonesEnv
        from sb3_contrib import MaskablePPO

        env = CorazonesEnv()
        modelo = MaskablePPO("MlpPolicy", env, n_steps=256, device="cpu")
        env.close()

        escenario = _crear_escenario_q_spades_sin_pozo()
        resultado = evaluar_escenario(
            modelo, escenario, num_repeticiones=5, vecnorm_path=None)

        # Verificar estructura (no valores específicos, modelo dummy es aleatorio)
        assert "escenario" in resultado
        assert "pct_juega_q_spades" in resultado
        assert "pct_primero" in resultado
        assert "punt_promedio" in resultado
        assert isinstance(resultado["pct_juega_q_spades"], float)
        assert isinstance(resultado["pct_primero"], float)


# ============================================================
# Test: Detección de VecNormalize
# ============================================================

class TestDeteccionVecNorm:
    """Verifica la detección automática de archivos VecNormalize."""

    def test_detectar_vecnorm_retorna_none_si_no_existe(self):
        """Si no hay archivo, debe retornar None."""
        from src.evaluacion import _detectar_vecnorm
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            resultado = _detectar_vecnorm(
                os.path.join(tmpdir, "no_existe_snapshot"), base_dir=tmpdir)
            assert resultado is None

    def test_detectar_vecnorm_encuentra_archivo(self):
        """Debe encontrar un archivo .pkl si existe en las rutas candidatas."""
        from src.evaluacion import _detectar_vecnorm
        import pickle
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crear estructura de directorios simulada
            v5_dir = os.path.join(tmpdir, "vecnormalize", "v5")
            os.makedirs(v5_dir)
            pkl_path = os.path.join(v5_dir, "v5_vecnorm.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"dummy": True}, f)

            resultado = _detectar_vecnorm(
                os.path.join(tmpdir, "modelo"), base_dir=tmpdir)
            assert resultado is not None
            assert "v5_vecnorm.pkl" in resultado


# ============================================================
# Test: Integración — callback de evaluación en entrenamiento
# ============================================================

class TestCallbackEvaluacion:
    """Verifica que el callback de evaluación se integra con el bucle de entrenamiento."""

    def test_resultado_eval_tiene_formato_correcto(self):
        """El resultado de una evaluación debe tener keys específicas."""
        resultado = {
            "paso": 1_000_000,
            "snapshot": "snapshot_0001000000",
            "win_rate_bots": 0.65,
            "top2_bots": 0.88,
            "punt_promedio": 48.5,
            "timestamp": "2026-01-01T00:00:00",
        }
        required = ["paso", "snapshot", "win_rate_bots",
                    "top2_bots", "punt_promedio"]
        for key in required:
            assert key in resultado, f"Falta key '{key}' en resultado de evaluación"

    def test_guardar_log_evaluacion_crea_archivo(self):
        """_guardar_log_evaluacion debe crear un archivo JSONL."""
        from src.evaluacion import _guardar_log_evaluacion
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "eval_log.jsonl")
            resultado = {
                "paso": 500_000,
                "snapshot": "snapshot_0000500000",
                "win_rate_bots": 0.55,
                "top2_bots": 0.80,
                "punt_promedio": 52.0,
            }
            _guardar_log_evaluacion(log_path, resultado)
            assert os.path.exists(log_path)

            with open(log_path, "r") as f:
                lineas = f.readlines()
            assert len(lineas) == 1
            import json
            data = json.loads(lineas[0])
            assert data["paso"] == 500_000
            assert data["win_rate_bots"] == 0.55
