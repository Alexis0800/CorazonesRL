"""
Pruebas unitarias para el sistema de rating Elo entre snapshots (Fase 5C).

Cubre:
    - Cálculo matemático de Elo (esperado, actualización, K-factor).
    - Función de match 1v1 entre dos snapshots.
    - Torneo round-robin completo con ratings ordenados.
    - Integridad de ratings (no negativos, orden consistente).
"""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================
# Test: Cálculo matemático de Elo
# ============================================================

class TestEloMatematico:
    """Verifica las fórmulas de Elo."""

    def test_expected_score_iguales(self):
        """Dos ratings iguales deben dar expected_score=0.5."""
        from src.torneo.elo import _expected_score
        e = _expected_score(1500, 1500)
        assert abs(e - 0.5) < 0.001, f"Expected 0.5, got {e}"

    def test_expected_score_favorito(self):
        """Rating mayor debe tener expected > 0.5."""
        from src.torneo.elo import _expected_score
        e = _expected_score(1600, 1400)
        assert e > 0.5, f"Expected >0.5, got {e}"
        assert e < 1.0, f"Expected <1.0, got {e}"

    def test_expected_score_underdog(self):
        """Rating menor debe tener expected < 0.5."""
        from src.torneo.elo import _expected_score
        e = _expected_score(1400, 1600)
        assert e < 0.5, f"Expected <0.5, got {e}"
        assert e > 0.0, f"Expected >0.0, got {e}"

    def test_expected_score_400_diff(self):
        """Diferencia de 400 pts debe dar expected ~0.91 para el favorito."""
        from src.torneo.elo import _expected_score
        e = _expected_score(1900, 1500)
        assert 0.90 < e < 0.92, f"Expected ~0.91 for 400 diff, got {e}"

    def test_update_elo_victoria(self):
        """Una victoria debe aumentar el rating del ganador."""
        from src.torneo.elo import _update_elo
        nuevo_a, nuevo_b = _update_elo(1500, 1500, score_a=1.0, k=32)
        assert nuevo_a > 1500, f"Ganador debe subir, got {nuevo_a}"
        assert nuevo_b < 1500, f"Perdedor debe bajar, got {nuevo_b}"
        assert abs((nuevo_a - 1500) + (nuevo_b - 1500)) < 0.01, "Suma cero"

    def test_update_elo_empate(self):
        """Un empate debe acercar ratings si son distintos."""
        from src.torneo.elo import _update_elo
        nuevo_a, nuevo_b = _update_elo(1600, 1400, score_a=0.5, k=32)
        assert nuevo_a < 1600, "Favorito debe bajar en empate"
        assert nuevo_b > 1400, "Underdog debe subir en empate"

    def test_update_elo_k_factor(self):
        """K=16 debe producir la mitad del cambio que K=32."""
        from src.torneo.elo import _update_elo
        _, nuevo_b_k32 = _update_elo(1500, 1500, score_a=1.0, k=32)
        _, nuevo_b_k16 = _update_elo(1500, 1500, score_a=1.0, k=16)
        diff_k32 = 1500 - nuevo_b_k32
        diff_k16 = 1500 - nuevo_b_k16
        assert abs(diff_k32 - 2 * diff_k16) < 0.01, \
            f"K=32 diff={diff_k32}, K=16 diff={diff_k16}"

    def test_elo_sum_zero(self):
        """La suma de ratings debe conservarse (zero-sum)."""
        from src.torneo.elo import _update_elo
        for ra, rb in [(1500, 1500), (1600, 1400), (1200, 1800)]:
            for sa in [1.0, 0.0, 0.5]:
                na, nb = _update_elo(ra, rb, score_a=sa, k=32)
                assert abs((na + nb) - (ra + rb)) < 0.01, \
                    f"Suma no conservada: {ra}+{rb}={ra+rb} → {na}+{nb}={na+nb}"


# ============================================================
# Test: Torneo y rating de snapshots
# ============================================================

class TestTorneoElo:
    """Verifica el torneo round-robin entre snapshots."""

    def test_torneo_resultados_tiene_ratings(self):
        """El resultado del torneo debe incluir ratings para cada snapshot."""
        from src.torneo.elo import _simular_resultado_torneo
        snaps = ["snap_A", "snap_B", "snap_C"]
        ratings_iniciales = {"snap_A": 1500, "snap_B": 1500, "snap_C": 1500}
        resultados = {
            ("snap_A", "snap_B"): (3, 2),  # A ganó 3, B ganó 2
            ("snap_A", "snap_C"): (4, 1),
            ("snap_B", "snap_C"): (2, 3),
        }
        ratings = _simular_resultado_torneo(
            snaps, ratings_iniciales, resultados)
        assert len(ratings) == 3
        for snap in snaps:
            assert snap in ratings
            assert ratings[snap] > 0

    def test_ganador_tiene_mayor_rating(self):
        """El snapshot con más victorias debe tener mayor rating."""
        from src.torneo.elo import _simular_resultado_torneo
        snaps = ["fuerte", "debil"]
        ratings_iniciales = {"fuerte": 1500, "debil": 1500}
        resultados = {("fuerte", "debil"): (10, 0)}
        ratings = _simular_resultado_torneo(
            snaps, ratings_iniciales, resultados)
        assert ratings["fuerte"] > ratings["debil"], \
            f"Fuerte={ratings['fuerte']}, Debil={ratings['debil']}"

    def test_orden_ratings_con_sentido(self):
        """Tres snapshots con desempeño claro deben ordenarse correctamente."""
        from src.torneo.elo import _simular_resultado_torneo
        snaps = ["A", "B", "C"]
        ratings_iniciales = {"A": 1500, "B": 1500, "C": 1500}
        # A domina a B, B domina a C → A > B > C
        resultados = {
            ("A", "B"): (8, 2),
            ("B", "C"): (7, 3),
            ("A", "C"): (9, 1),
        }
        ratings = _simular_resultado_torneo(
            snaps, ratings_iniciales, resultados)
        assert ratings["A"] > ratings["B"] > ratings["C"], \
            f"Orden incorrecto: A={ratings['A']:.0f} B={ratings['B']:.0f} C={ratings['C']:.0f}"


# ============================================================
# Test: Integración — torneo con snapshots reales
# ============================================================

class TestTorneoConSnapshots:
    """Verifica la integración del torneo con snapshots .zip reales."""

    def test_listar_snapshots_para_torneo(self):
        """Debe listar snapshots del directorio v5."""
        from src.torneo.elo import _listar_snapshots_torneo
        snaps = _listar_snapshots_torneo(
            [os.path.join(os.path.dirname(__file__), "..",
                          "modelos", "v5", "snapshots")],
            min_paso=0,
            max_snapshots=10,
        )
        assert isinstance(snaps, list)
        if snaps:
            # Debe retornar tuplas (ruta, label)
            assert isinstance(snaps[0], tuple)
            assert len(snaps[0]) == 2

    def test_extraer_paso_de_snapshot(self):
        """Debe extraer el número de paso del nombre del snapshot."""
        from src.torneo.elo import _extraer_paso_snapshot
        assert _extraer_paso_snapshot("snapshot_0005000000.zip") == 5000000
        assert _extraer_paso_snapshot(
            "D:\\path\\snapshot_0001000000.zip") == 1000000
        assert _extraer_paso_snapshot("snapshot_0000100000") == 100000

    def test_muestreo_estratificado_multi_dir(self):
        """Con múltiples directorios, cada uno debe estar representado."""
        from src.torneo.elo import _listar_snapshots_torneo, _extraer_paso_snapshot
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "vA")
            dir_b = os.path.join(tmpdir, "vB")
            os.makedirs(dir_a)
            os.makedirs(dir_b)
            # Crear snapshots: 3 en A, 10 en B
            for i in range(3):
                fname = f"snapshot_{1000000 + i*100000:010d}.zip"
                with open(os.path.join(dir_a, fname), "w") as f:
                    f.write("")
            for i in range(10):
                fname = f"snapshot_{2000000 + i*100000:010d}.zip"
                with open(os.path.join(dir_b, fname), "w") as f:
                    f.write("")

            snaps = _listar_snapshots_torneo(
                [dir_a, dir_b], min_paso=0, max_snapshots=8)

            # Verificar que ambos directorios están representados
            labels = {label for _, label in snaps}
            assert "vA" in labels, f"Falta vA en labels: {labels}"
            assert "vB" in labels, f"Falta vB en labels: {labels}"
            # Al menos 1 snapshot de cada directorio
            count_a = sum(1 for _, l in snaps if l == "vA")
            count_b = sum(1 for _, l in snaps if l == "vB")
            assert count_a >= 1
            assert count_b >= 1


class TestAdyacentes:
    """Verifica la búsqueda de snapshots adyacentes para Elo puro."""

    def test_adyacentes_encuentra_vecinos(self):
        """Debe encontrar snapshots cercanos en pasos de entrenamiento."""
        from src.torneo.elo import _encontrar_adyacentes, _extraer_paso_snapshot
        pool = [
            "/path/snapshot_0001000000.zip",
            "/path/snapshot_0002000000.zip",
            "/path/snapshot_0003000000.zip",
            "/path/snapshot_0004000000.zip",
            "/path/snapshot_0005000000.zip",
        ]
        resultado = _encontrar_adyacentes(
            "/path/snapshot_0003000000.zip", pool)
        assert len(resultado) == 2
        pasos = sorted([_extraer_paso_snapshot(s) for s in resultado])
        # Debería encontrar 2M y 4M (los más cercanos a 3M)
        assert pasos == [2000000, 4000000]

    def test_adyacentes_borde_inferior(self):
        """En el borde inferior, solo encuentra vecinos hacia arriba."""
        from src.torneo.elo import _encontrar_adyacentes, _extraer_paso_snapshot
        pool = [
            "/path/snapshot_0001000000.zip",
            "/path/snapshot_0002000000.zip",
            "/path/snapshot_0003000000.zip",
        ]
        resultado = _encontrar_adyacentes(
            "/path/snapshot_0001000000.zip", pool)
        pasos = [_extraer_paso_snapshot(s) for s in resultado]
        assert 1000000 not in pasos, "No debe incluirse a sí mismo"
        assert len(resultado) >= 1  # Al menos un vecino

    def test_adyacentes_pool_pequeno(self):
        """Con pool de solo 2 snapshots, retorna el otro."""
        from src.torneo.elo import _encontrar_adyacentes
        pool = [
            "/path/snapshot_0001000000.zip",
            "/path/snapshot_0002000000.zip",
        ]
        resultado = _encontrar_adyacentes(
            "/path/snapshot_0001000000.zip", pool)
        assert len(resultado) == 1


# ============================================================
# Test: Cálculo Elo convergente (sin order bias)
# ============================================================

class TestEloConvergente:
    """Verifica que el recálculo iterativo elimina el order bias."""

    def test_convergente_mismo_resultado_que_simular(self):
        """Con un solo oponente, el ranking debe coincidir con el método no-convergente."""
        from src.torneo.elo import _simular_resultado_torneo, _calcular_elo_convergente
        snaps = ["A", "B"]
        resultados = {("A", "B"): (10, 0)}
        ratings_sim = _simular_resultado_torneo(
            snaps, {"A": 1500, "B": 1500}, resultados)
        ratings_conv, _ = _calcular_elo_convergente(snaps, resultados)
        # El método convergente usa la matriz completa y da ratings más extremos,
        # pero el ORDEN del ranking debe ser el mismo
        assert ratings_conv["A"] > ratings_conv["B"], "A debe seguir siendo mejor que B"
        # Verificar que el ganador tiene rating >1500 y perdedor <1500
        assert ratings_conv["A"] > 1500
        assert ratings_conv["B"] < 1500

    def test_convergente_orden_correcto(self):
        """Tres jugadores con desempeño claro deben ordenarse A > B > C."""
        from src.torneo.elo import _calcular_elo_convergente
        snaps = ["A", "B", "C"]
        resultados = {
            ("A", "B"): (8, 2),
            ("B", "C"): (7, 3),
            ("A", "C"): (9, 1),
        }
        ratings, _ = _calcular_elo_convergente(snaps, resultados)
        assert ratings["A"] > ratings["B"] > ratings["C"], \
            f"Orden incorrecto: A={ratings['A']:.0f} B={ratings['B']:.0f} C={ratings['C']:.0f}"

    def test_convergente_simetrico(self):
        """Si A y B son idénticos (5-5), deben tener ratings casi iguales."""
        from src.torneo.elo import _calcular_elo_convergente
        snaps = ["X", "Y"]
        resultados = {("X", "Y"): (5, 5)}
        ratings, _ = _calcular_elo_convergente(snaps, resultados)
        assert abs(ratings["X"] - ratings["Y"]) < 1.0, \
            f"Deben ser casi iguales: X={ratings['X']:.1f} Y={ratings['Y']:.1f}"

    def test_convergente_immunidad_order_bias(self):
        """El orden de las entradas en resultados no debe afectar el rating final."""
        from src.torneo.elo import _calcular_elo_convergente
        snaps = ["P1", "P2", "P3", "P4"]
        resultados = {
            ("P1", "P2"): (8, 2),
            ("P1", "P3"): (9, 1),
            ("P1", "P4"): (10, 0),
            ("P2", "P3"): (6, 4),
            ("P2", "P4"): (8, 2),
            ("P3", "P4"): (7, 3),
        }
        ratings1, iters1 = _calcular_elo_convergente(snaps, resultados)
        # Invertir orden del dict
        resultados_inv = dict(reversed(list(resultados.items())))
        ratings2, iters2 = _calcular_elo_convergente(snaps, resultados_inv)
        for snap in snaps:
            assert abs(ratings1[snap] - ratings2[snap]) < 0.01, \
                f"Order bias en {snap}: {ratings1[snap]:.6f} vs {ratings2[snap]:.6f}"

    def test_convergente_converge_rapido(self):
        """Gauss-Seidel con permutación: converge en <50 iters."""
        from src.torneo.elo import _calcular_elo_convergente
        snaps = [f"P{i}" for i in range(6)]
        resultados = {}
        for i in range(6):
            for j in range(i + 1, 6):
                gap = j - i
                wins_i = min(10, 5 + gap)  # 6-4, 7-3, 8-2, 9-1, 10-0
                wins_j = 10 - wins_i
                resultados[(snaps[i], snaps[j])] = (wins_i, wins_j)
        _, iters = _calcular_elo_convergente(snaps, resultados)
        assert iters < 50, f"Debe converger en <50 iters, usó {iters}"

    def test_convergente_vacio_no_falla(self):
        """Con 0 matches no debe fallar."""
        from src.torneo.elo import _calcular_elo_convergente
        snaps = ["A", "B"]
        ratings, iters = _calcular_elo_convergente(snaps, {})
        assert ratings["A"] == 1500
        assert ratings["B"] == 1500
        assert iters == 0

    def test_convergente_un_solo_jugador(self):
        """Edge case: un solo jugador sin matches."""
        from src.torneo.elo import _calcular_elo_convergente
        ratings, iters = _calcular_elo_convergente(["Solo"], {})
        assert ratings["Solo"] == 1500
        assert iters == 0
