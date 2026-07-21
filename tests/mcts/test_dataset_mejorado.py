"""
Tests TDD para dataset generator mejorado (v2): MCTS + soft labels + multi-agente.
"""
import numpy as np
import pytest
from src.entorno.dimensiones import DIM_ENTORNO


class TestDatasetSoftLabels:
    """Verifica que el dataset con soft labels tiene formato correcto."""

    def test_generar_una_mano_soft_labels(self):
        """generar_dataset_una_mano con soft_labels retorna scores (N,52)."""
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="evasivo",
            tipo_oponentes="heuristicos",
            use_mcts=False,
            soft_labels=True,
        )
        # Cada par es (obs, scores_array) donde scores_array es (52,) float32
        assert len(pares) > 0, "Debe generar al menos un par"
        obs, scores = pares[0]
        assert obs.shape == (DIM_ENTORNO,)
        assert scores.shape == (52,)
        assert scores.dtype == np.float32

        # Las acciones legales deben tener scores finitos
        n_finitos = np.sum(np.isfinite(scores))
        assert n_finitos >= 1, "Al menos una accion legal con score"
        # Las acciones ilegales deben tener score = +inf o NaN
        n_inf = np.sum(np.isinf(scores) | np.isnan(scores))
        assert n_inf >= 52 - n_finitos, "Acciones ilegales marcadas"

    def test_soft_labels_multi_agente(self):
        """Multi-agente + soft labels: 4× mas datos, todos con scores (52,)."""
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="evasivo",
            tipo_oponentes="heuristicos",
            use_mcts=False,
            soft_labels=True,
            multi_agente=True,
        )
        # Con multi_agente, genera para las 4 posiciones
        # Cada mano tiene ~13 bazas × 4 jugadores ≈ 52 decisiones
        assert len(pares) >= 13 * \
            4, f"Multi-agente debe generar ~52 pares, genero {len(pares)}"
        # Verificar formato
        for obs, scores in pares[:5]:
            assert obs.shape == (DIM_ENTORNO,)
            assert scores.shape == (52,)
            assert scores.dtype == np.float32

    def test_soft_labels_con_mcts(self):
        """Soft labels + MCTS multi-step (requiere profundidad>=2).

        Antes este test NO pasaba `profundidad`, así que usaba el default 1 y
        recibía PIMC pese a su nombre: la rama use_mcts de `_score_una_carta`
        llamaba a `_pimc_score_carta`. Los asserts (shape/finitud) no podían
        detectarlo. Ahora pide profundidad=2 y ejerce el MCTS real.
        """
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="evasivo",
            tipo_oponentes="heuristicos",
            use_mcts=True,
            mcts_simulaciones=20,
            soft_labels=True,
            profundidad=2,
        )
        assert len(pares) > 0
        obs, scores = pares[0]
        assert scores.shape == (52,)
        # Scores deben ser finitos para acciones legales
        assert np.any(np.isfinite(scores))

    def test_soft_labels_mcts_profundidad_1_falla_explicito(self):
        """soft_labels + use_mcts + profundidad=1 debe FALLAR, no degradar a PIMC.

        Contrato del fix: el MCTS de raíz reparte simulaciones adaptativamente y
        no da score comparable por carta, así que no se puede servir como soft
        label. Antes devolvía PIMC en silencio (etiquetado como MCTS).
        """
        import pytest
        from src.mcts.dataset import generar_dataset_una_mano

        with pytest.raises(ValueError, match="profundidad"):
            generar_dataset_una_mano(
                seed=42,
                agente_idx=0,
                num_mundos=10,
                rollout_tipo="evasivo",
                tipo_oponentes="heuristicos",
                use_mcts=True,
                mcts_simulaciones=20,
                soft_labels=True,
                profundidad=1,
            )


class TestDatasetCompatibilidad:
    """Verifica retrocompatibilidad con formato antiguo."""

    def test_formato_antiguo_sigue_funcionando(self):
        """Sin soft_labels, retorna (obs, action_id) como antes."""
        from src.mcts.dataset import generar_dataset_una_mano

        pares = generar_dataset_una_mano(
            seed=42,
            agente_idx=0,
            num_mundos=10,
            rollout_tipo="evasivo",
            tipo_oponentes="heuristicos",
            use_mcts=False,
            soft_labels=False,
        )
        obs, action_id = pares[0]
        assert isinstance(action_id, (int, np.integer))
        assert 0 <= action_id <= 51

    def test_generar_dataset_con_soft_labels(self):
        """generar_dataset() con soft_labels retorna scores (N,52)."""
        from src.mcts.dataset import generar_dataset

        obs, actions, meta = generar_dataset(
            num_manos=3,
            num_mundos=10,
            rollout_tipo="evasivo",
            tipo_oponentes="heuristicos",
            num_workers=1,
            seed=42,
            use_mcts=False,
            soft_labels=True,
        )
        assert obs.shape[0] >= 3  # al menos 3 pares
        assert actions.shape == (obs.shape[0], 52)
        assert meta["soft_labels"] is True
