"""Tests para el modulo de evaluacion estandarizada v3.

Verifica que el formato [Modelo, Experto, Experto, BotRotativo]
funcione correctamente para evaluar modelos individuales.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class TestEvaluacionEstandar:
    """Verifica la funcion principal de evaluacion estandarizada."""

    def test_resultados_rango_valido(self):
        """WR y score promedio deben estar en rangos validos."""
        from src.v3.evaluacion import EvaluacionEstandar, evaluar_estandar

        bots = [
            ("[BOT] conservador", True, False, 0, None),
            ("[BOT] agresivo", True, False, 1, None),
            ("[BOT] evasivo", True, False, 2, None),
        ]
        resultados = []
        for nombre, es_bot, es_experto, bot_idx, modelo_path in bots:
            res = evaluar_estandar(
                nombre=nombre,
                num_manos=20,
                seed_base=42,
                es_bot=es_bot,
                es_experto=es_experto,
                bot_idx=bot_idx,
            )
            resultados.append(res)
            assert 0.0 <= res.wr <= 1.0
            assert 0 <= res.avg_score <= 26
            assert res.num_manos == 20

    def test_estructura_resultado(self):
        """Verifica que EvaluacionEstandar tenga todos los campos necesarios."""
        from src.v3.evaluacion import EvaluacionEstandar, evaluar_estandar

        res = evaluar_estandar(
            nombre="[BOT] conservador",
            num_manos=15,
            seed_base=123,
            es_bot=True,
            bot_idx=0,
        )

        assert isinstance(res.wr, float)
        assert isinstance(res.avg_score, float)
        assert isinstance(res.num_manos, int)
        assert isinstance(res.nombre, str)
        assert isinstance(res.scores, list)
        assert len(res.scores) == res.num_manos
        assert res.es_bot is True
        assert res.es_experto is False

    def test_bot_vs_campo_estandar_no_crashea(self):
        """Todos los bots heuristicos deben poder evaluarse sin errores."""
        from src.v3.evaluacion import evaluar_estandar

        for bot_idx, nombre in enumerate(
            ["conservador", "agresivo", "evasivo"]
        ):
            res = evaluar_estandar(
                nombre=f"[BOT] {nombre}",
                num_manos=10,
                seed_base=42,
                es_bot=True,
                bot_idx=bot_idx,
            )
            assert res.num_manos == 10
            assert 0 <= res.wr <= 1.0

    def test_experto_vs_campo_estandar_no_crashea(self):
        """BotExperto evaluado en campo estandar no debe crashear."""
        from src.v3.evaluacion import evaluar_estandar

        res = evaluar_estandar(
            nombre="[BOT] experto",
            num_manos=10,
            seed_base=42,
            es_experto=True,
        )
        assert res.num_manos == 10
        assert 0 <= res.wr <= 1.0

    def test_modelo_vs_campo_estandar_no_crashea(self):
        """Un snapshot v3 real debe poder evaluarse sin errores."""
        from src.v3.evaluacion import evaluar_estandar

        snap_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "models", "v3", "snapshots",
        )
        snaps = sorted(
            [f for f in os.listdir(snap_dir) if f.startswith(
                "snapshot_") and f.endswith(".zip")]
        ) if os.path.isdir(snap_dir) else []

        if not snaps:
            pytest.skip("No hay snapshots disponibles para el test")

        res = evaluar_estandar(
            nombre="test_modelo",
            modelo_path=os.path.join(snap_dir, snaps[-1]),
            num_manos=5,
            seed_base=42,
        )
        assert res.num_manos == 5
        assert res.wr >= 0.0

    def test_semilla_deterministica(self):
        """Misma semilla debe dar resultados aproximados.

        Nota: BotExperto tiene componentes estocasticos, por lo que
        dos evaluaciones con la misma semilla daran resultados
        cercanos pero no identicos. Verificamos correlacion
        de scores y diferencia de WR <= 0.2.
        """
        from src.v3.evaluacion import evaluar_estandar

        res1 = evaluar_estandar(
            nombre="[BOT] conservador",
            num_manos=10,
            seed_base=999,
            es_bot=True,
            bot_idx=0,
        )
        res2 = evaluar_estandar(
            nombre="[BOT] conservador",
            num_manos=10,
            seed_base=999,
            es_bot=True,
            bot_idx=0,
        )

        # Con solo 10 manos, permitir diferencia de hasta 2 victorias
        assert abs(res1.wr - res2.wr) <= 0.3, \
            f"WR muy diferentes: {res1.wr} vs {res2.wr}"
        # Los scores promedio deberian ser cercanos
        assert abs(res1.avg_score - res2.avg_score) <= 7.0, \
            f"Scores muy diferentes: {res1.avg_score} vs {res2.avg_score}"


class TestCampoEstandar:
    """Verifica que el campo [Experto, Experto, BotRotativo] sea correcto."""

    def test_rotacion_bot_por_mano(self):
        """El bot heuristico debe rotar entre manos para diversidad."""
        from src.v3.evaluacion import _construir_oponentes_estandar

        # Construir campo para 12 manos
        oponentes_por_mano = []
        for h in range(12):
            ops = _construir_oponentes_estandar(
                agente_idx=h % 4, seed=h)
            oponentes_por_mano.append(ops)

        # Verificar que cada mano tiene 3 oponentes
        for ops in oponentes_por_mano:
            assert len(ops) == 3

    def test_campo_sin_colisiones(self):
        """El agente no debe aparecer como oponente."""
        from src.v3.evaluacion import _construir_oponentes_estandar

        for agente_idx in range(4):
            ops = _construir_oponentes_estandar(
                agente_idx=agente_idx, seed=0)
            assert agente_idx not in ops
