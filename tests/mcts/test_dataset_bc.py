"""
Tests TDD para dataset BC con PIMC y MCTS mejorado.

Verifica:
  - mcts_mejor_jugada usa bots frescos por simulación
  - generar_dataset_bc produce (obs, action) pairs correctos
  - Multiprocessing no produce errores
"""
import pytest
import numpy as np
import os
import tempfile

from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
from src.agentes.heuristicos import bot_evasivo
from src.entorno.dimensiones import DIM_ENTRENAMIENTO


# ============================================================
# Helpers
# ============================================================

def _motor_con_mano_fija(seed: int = 42) -> MotorCorazones:
    """Motor reproducible con seed fijo."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    motor = MotorCorazones()
    motor.repartir()
    return motor


# ============================================================
# TestMCTSMejorJugada
# ============================================================

class TestMCTSMejorJugada:
    """Verifica que mcts_mejor_jugada funcione correctamente."""

    def test_retorna_carta_legal(self):
        """MCTS debe retornar una carta de la lista de legales."""
        from src.mcts.pimc import mcts_mejor_jugada
        motor = _motor_con_mano_fija()
        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        carta = mcts_mejor_jugada(
            motor, idx, legales,
            num_simulaciones=20,
            rollout_tipo="evasivo",
        )
        assert carta in legales, f"MCTS retornó carta ilegal: {carta}"

    def test_una_sola_legal_retorna_esa(self):
        """Con una sola carta legal, debe retornarla sin simular."""
        from src.mcts.pimc import mcts_mejor_jugada
        motor = _motor_con_mano_fija()
        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        unica = [legales[0]]
        carta = mcts_mejor_jugada(
            motor, idx, unica,
            num_simulaciones=10,
            rollout_tipo="evasivo",
        )
        assert carta is unica[0]

    def test_legales_vacia_lanza_error(self):
        """Lista vacía debe lanzar ValueError."""
        from src.mcts.pimc import mcts_mejor_jugada
        motor = _motor_con_mano_fija()
        with pytest.raises(ValueError):
            mcts_mejor_jugada(motor, 0, [])

    def test_bots_frescos_por_simulacion_experto(self):
        """Con rollout 'experto', MCTS debe funcionar sin errores de estado.

        BotExperto tiene estado interno (vacios, sospecha_pozo) que se
        acumula entre simulaciones. Si se reutilizan bots, el estado
        contaminado produce decisiones incorrectas o errores.
        Verificamos que MCTS produce resultados válidos con rollout experto.
        """
        from src.mcts.pimc import mcts_mejor_jugada

        motor = _motor_con_mano_fija()
        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)

        # Ejecutar MCTS con rollout experto
        carta = mcts_mejor_jugada(
            motor, idx, legales,
            num_simulaciones=30,
            rollout_tipo="experto",
        )
        assert carta in legales, "MCTS con experto debe retornar carta legal"

        # Verificar que podemos ejecutar múltiples veces sin error
        for _ in range(3):
            carta2 = mcts_mejor_jugada(
                motor, idx, legales,
                num_simulaciones=10,
                rollout_tipo="experto",
            )
            assert carta2 in legales

    def test_con_rollout_mixto(self):
        """MCTS con rollout mixto debe funcionar."""
        from src.mcts.pimc import mcts_mejor_jugada
        motor = _motor_con_mano_fija()
        idx = motor.obtener_jugador_actual()
        legales = motor.obtener_jugadas_legales(idx)
        carta = mcts_mejor_jugada(
            motor, idx, legales,
            num_simulaciones=20,
            rollout_tipo="mixto",
        )
        assert carta in legales


# ============================================================
# TestGenerarDatasetBC
# ============================================================

class TestGenerarDatasetBC:
    """Verifica la generación de dataset BC con PIMC."""

    def test_genera_pares_obs_accion(self):
        """Una mano debe generar pares (obs, action) con dimensiones correctas."""
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="evasivo",
            tipo_oponentes="heuristicos",
        )
        assert len(pares) > 0, "Debe generar al menos un par por mano"
        for obs, action_id in pares:
            assert obs.shape == (DIM_ENTRENAMIENTO,), \
                f"Obs shape {obs.shape} != ({DIM_ENTRENAMIENTO},)"
            assert obs.dtype == np.float32
            assert 0 <= action_id < 52
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0), \
                "Observación fuera de rango [0, 1]"

    def test_tipo_oponentes_mixto(self):
        """Dataset con oponentes mixtos (heuristicos + experto) debe funcionar."""
        from src.mcts.dataset import generar_dataset_una_mano
        pares = generar_dataset_una_mano(
            seed=42, num_mundos=10, rollout_tipo="evasivo",
            tipo_oponentes="mixto",
        )
        assert len(pares) > 0

    def test_tipo_oponentes_experto(self):
        """Dataset con oponentes 100% experto debe funcionar."""
        from src.mcts.dataset import generar_dataset_una_mano
        pares = generar_dataset_una_mano(
            seed=42, num_mundos=10, rollout_tipo="evasivo",
            tipo_oponentes="experto",
        )
        assert len(pares) > 0

    def test_con_mcts(self):
        """Dataset con MCTS en vez de PIMC debe funcionar."""
        from src.mcts.dataset import generar_dataset_una_mano
        pares = generar_dataset_una_mano(
            seed=42, num_mundos=5, rollout_tipo="evasivo",
            tipo_oponentes="experto", use_mcts=True, mcts_simulaciones=20,
        )
        assert len(pares) > 0
        for obs, action_id in pares:
            assert obs.shape == (DIM_ENTRENAMIENTO,)
            assert 0 <= action_id < 52

    def test_dataset_multiprocessing(self):
        """Generación con 2 workers no debe tirar errores."""
        from src.mcts.dataset import generar_dataset

        obs, actions, meta = generar_dataset(
            num_manos=4,
            num_mundos=10,
            rollout_tipo="evasivo",
            num_workers=2,
            seed=123,
        )
        assert len(obs) == len(actions)
        assert len(obs) > 0
        assert obs.shape[1] == DIM_ENTRENAMIENTO
        assert meta["num_manos"] == 4
        assert meta["rollout_tipo"] == "evasivo"

    def test_guardar_y_cargar_dataset(self):
        """Dataset guardado como .npz + .json debe ser recuperable."""
        from src.mcts.dataset import generar_dataset, guardar_dataset, cargar_dataset

        obs, actions, meta = generar_dataset(
            num_manos=2,
            num_mundos=8,
            rollout_tipo="evasivo",
            num_workers=1,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "test_bc")
            guardar_dataset(prefix, obs, actions, meta)
            assert os.path.exists(prefix + ".npz")
            assert os.path.exists(prefix + ".json")

            obs2, actions2, meta2 = cargar_dataset(prefix)
            assert len(obs2) == len(obs)
            assert np.array_equal(actions2, actions)
            assert meta2["num_manos"] == meta["num_manos"]

    def test_reproducibilidad_con_seed(self):
        """Misma seed debe producir el mismo dataset."""
        from src.mcts.dataset import generar_dataset

        obs1, act1, _ = generar_dataset(
            num_manos=2, num_mundos=8,
            rollout_tipo="evasivo", num_workers=1, seed=42,
        )
        obs2, act2, _ = generar_dataset(
            num_manos=2, num_mundos=8,
            rollout_tipo="evasivo", num_workers=1, seed=42,
        )
        assert np.array_equal(
            act1, act2), "Misma seed debe dar mismas acciones"
