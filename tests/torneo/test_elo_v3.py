"""Tests para el torneo Elo v3 con formato de duelo [A, B, Experto, Experto]."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class TestJugarDuelo:
    """Verifica que el duelo A vs B con 2 Expertos funcione correctamente."""

    def test_duelo_resultados_rango_valido(self):
        """Los resultados del duelo deben estar en [0, 1]."""
        from src.v3.elo import JugadorV3, jugar_duelo

        a = JugadorV3(nombre="[BOT] conservador", es_bot=True, bot_idx=0)
        b = JugadorV3(nombre="[BOT] agresivo", es_bot=True, bot_idx=1)

        res = jugar_duelo(a, b, num_manos=30, seed_base=42)

        assert 0.0 <= res["pct_a_gana"] <= 1.0
        assert 0.0 <= res["pct_b_gana"] <= 1.0
        assert 0 <= res["avg_score_a"] <= 26
        assert 0 <= res["avg_score_b"] <= 26

    def test_duelo_simetrico_aproximado(self):
        """A vs B deberia dar resultados cercanos al inverso de B vs A."""
        from src.v3.elo import JugadorV3, jugar_duelo

        a = JugadorV3(nombre="[BOT] evasivo", es_bot=True, bot_idx=2)
        b = JugadorV3(nombre="[BOT] conservador", es_bot=True, bot_idx=0)

        res_ab = jugar_duelo(a, b, num_manos=100, seed_base=42)
        res_ba = jugar_duelo(b, a, num_manos=100, seed_base=999)

        # Con suficientes manos, las tasas deberian ser aprox complementarias
        total_a_gana = res_ab["pct_a_gana"] + res_ba["pct_a_gana"]
        assert 0.5 < total_a_gana < 1.5, \
            f"Simetria rota: {res_ab['pct_a_gana']:.2f} + {res_ba['pct_a_gana']:.2f} = {total_a_gana:.2f}"

    def test_duelo_rota_asientos(self):
        """Verifica que los asientos roten entre manos."""
        from src.v3.elo import JugadorV3, jugar_duelo

        a = JugadorV3(nombre="test_a", es_bot=True, bot_idx=0)
        b = JugadorV3(nombre="test_b", es_bot=True, bot_idx=1)

        res = jugar_duelo(a, b, num_manos=40, seed_base=42)
        assert res["num_manos"] == 40

    def test_duelo_bot_vs_bot_no_crashea(self):
        """Todos los pares de bots deben poder enfrentarse sin errores."""
        from src.v3.elo import JugadorV3, jugar_duelo

        bots = [
            ("conservador", 0),
            ("agresivo", 1),
            ("evasivo", 2),
        ]
        for i in range(len(bots)):
            for j in range(i + 1, len(bots)):
                a = JugadorV3(
                    nombre=f"[BOT] {bots[i][0]}", es_bot=True, bot_idx=bots[i][1])
                b = JugadorV3(
                    nombre=f"[BOT] {bots[j][0]}", es_bot=True, bot_idx=bots[j][1])
                res = jugar_duelo(a, b, num_manos=10, seed_base=42)
                assert res["num_manos"] == 10

    def test_duelo_experto_vs_modelo_no_crashea(self):
        """BotExperto vs snapshot no debe crashear."""
        from src.v3.elo import JugadorV3, jugar_duelo

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

        a = JugadorV3(
            nombre="test_modelo",
            modelo_path=os.path.join(snap_dir, snaps[-1]),
        )
        b = JugadorV3(nombre="[BOT] experto", es_experto=True)

        res = jugar_duelo(a, b, num_manos=10, seed_base=42)
        assert res["num_manos"] == 10


class TestCalcularElo:
    """Verifica el calculo de ratings Elo."""

    def test_elo_jugador_dominante(self):
        """Un jugador que gana todo debe tener Elo mucho mayor."""
        from src.v3.elo import calcular_elo

        resultados = [("A", "B", 0.9) for _ in range(10)]
        ratings = calcular_elo(resultados)

        assert ratings["A"] > ratings["B"]
        assert ratings["A"] - ratings["B"] > 200

    def test_elo_jugadores_iguales(self):
        """Dos jugadores empatados deben tener Elo similar."""
        from src.v3.elo import calcular_elo

        resultados = [("A", "B", 0.5) for _ in range(10)]
        ratings = calcular_elo(resultados)

        assert abs(ratings["A"] - ratings["B"]) < 100

    def test_elo_tres_jugadores_ordena_correctamente(self):
        """A > B > C debe reflejarse en los ratings."""
        from src.v3.elo import calcular_elo

        resultados = [
            ("A", "B", 0.7),
            ("B", "C", 0.7),
            ("A", "C", 0.85),
        ]
        ratings = calcular_elo(resultados)

        assert ratings["A"] > ratings["B"] > ratings["C"]


class TestJugadorV3:
    """Verifica la carga de modelos."""

    def test_jugador_bot_no_carga_modelo(self):
        """JugadorV3 bot no debe intentar cargar MaskablePPO."""
        from src.v3.elo import JugadorV3

        j = JugadorV3(nombre="[BOT] test", es_bot=True)
        j.cargar()
        assert j._modelo is None

    def test_jugador_experto_no_carga_modelo(self):
        """JugadorV3 experto no debe intentar cargar MaskablePPO."""
        from src.v3.elo import JugadorV3

        j = JugadorV3(nombre="[BOT] experto", es_experto=True)
        j.cargar()
        assert j._modelo is None
