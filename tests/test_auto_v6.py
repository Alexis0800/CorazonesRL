"""
Pruebas unitarias para train.py: entrenamiento autónomo.

Cubre:
  - Decaimiento progresivo de prob_bot
  - Formato de logging JSON
  - Construcción de comandos de torneo ELO
"""
import pytest
import json
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock


# ============================================================
# Helpers
# ============================================================

def _decaimiento_prob_bot(
    paso_actual: int, total_pasos: int,
    inicio: float = 0.50, fin: float = 0.10,
) -> float:
    """Calcula prob_bot con decaimiento lineal según el progreso.

    Args:
        paso_actual: Paso global actual.
        total_pasos: Pasos totales planeados.
        inicio: prob_bot inicial.
        fin: prob_bot final.

    Returns:
        Probabilidad de usar bot heurístico.
    """
    if total_pasos <= 0:
        return fin
    progreso = min(paso_actual / total_pasos, 1.0)
    return inicio + (fin - inicio) * progreso


def _formato_eval_jsonl(
    paso: int, win_rate: float, avg_score: float,
    num_partidas: int, prob_bot: float,
) -> dict:
    """Construye entrada de evaluación en formato JSONL.

    Args:
        paso: Paso de entrenamiento.
        win_rate: Tasa de victorias (0.0 a 1.0).
        avg_score: Puntuación promedio del agente.
        num_partidas: Número de partidas evaluadas.
        prob_bot: prob_bot usado en el entrenamiento.

    Returns:
        Diccionario con los campos de evaluación.
    """
    return {
        "timestamp": "auto",  # se rellena en producción
        "paso": paso,
        "win_rate_bots": round(win_rate, 4),
        "avg_score": round(avg_score, 2),
        "num_partidas": num_partidas,
        "prob_bot": round(prob_bot, 4),
    }


def _comando_elo(
    directorio: str, min_paso: int, max_snapshots: int,
    partidas: int, output_file: str,
) -> str:
    """Construye el comando de torneo ELO como string.

    Args:
        directorio: Directorio de snapshots.
        min_paso: Paso mínimo para incluir snapshots.
        max_snapshots: Máximo de snapshots a incluir.
        partidas: Partidas por enfrentamiento.
        output_file: Archivo de salida para resultados.

    Returns:
        Comando listo para ejecutar en terminal.
    """
    return (
        f'python -m src.torneo.elo '
        f'--directorio {directorio} '
        f'--partidas {partidas} '
        f'--min-paso {min_paso} '
        f'--max-snapshots {max_snapshots} '
        f'--elo-puro '
        f'--incluir-bots '
        f'> {output_file} 2>&1'
    )


# ============================================================
# Tests de decaimiento de prob_bot
# ============================================================

class TestDecaimientoProbBot:
    """Verifica el decaimiento progresivo de prob_bot."""

    def test_inicio_es_prob_bot_inicial(self):
        """Al inicio (paso 0), prob_bot debe ser el valor inicial."""
        resultado = _decaimiento_prob_bot(0, 20_000_000, inicio=0.50, fin=0.10)
        assert resultado == pytest.approx(0.50)

    def test_final_es_prob_bot_final(self):
        """Al final (total_pasos), prob_bot debe ser el valor final."""
        resultado = _decaimiento_prob_bot(
            20_000_000, 20_000_000, inicio=0.50, fin=0.10)
        assert resultado == pytest.approx(0.10)

    def test_mitad_es_promedio(self):
        """A la mitad del entrenamiento, prob_bot debe ser el promedio."""
        resultado = _decaimiento_prob_bot(
            10_000_000, 20_000_000, inicio=0.50, fin=0.10)
        assert resultado == pytest.approx(0.30)

    def test_pasado_el_total_no_baja_del_final(self):
        """Más allá del total, no debe bajar del valor final."""
        resultado = _decaimiento_prob_bot(
            30_000_000, 20_000_000, inicio=0.50, fin=0.10)
        assert resultado == pytest.approx(0.10)

    def test_con_rango_personalizado(self):
        """Con valores personalizados, sigue la fórmula lineal."""
        # 0.80 → 0.20, al 25% debe ser 0.65
        resultado = _decaimiento_prob_bot(
            5_000_000, 20_000_000, inicio=0.80, fin=0.20)
        assert resultado == pytest.approx(0.65)

    def test_total_pasos_cero_devuelve_fin(self):
        """Edge case: total_pasos=0 devuelve el valor final."""
        resultado = _decaimiento_prob_bot(0, 0, inicio=0.50, fin=0.10)
        assert resultado == pytest.approx(0.10)


# ============================================================
# Tests de decaimiento coseno de prob_bot
# ============================================================

def _decaimiento_coseno_prob_bot(
    paso_actual: int, total_pasos: int,
    inicio: float = 0.50, fin: float = 0.20,
) -> float:
    """Calcula prob_bot con decaimiento coseno según el progreso.

    El decaimiento coseno mantiene valores más altos durante más tiempo,
    reduciendo el riesgo de sobreajuste al self-play al inicio.
    Fórmula: fin + 0.5 * (inicio - fin) * (1 + cos(pi * progreso))

    Args:
        paso_actual: Paso global actual.
        total_pasos: Pasos totales planeados.
        inicio: prob_bot inicial.
        fin: prob_bot final (piso).

    Returns:
        Probabilidad de usar bot heurístico.
    """
    import math
    if total_pasos <= 0:
        return fin
    progreso = min(paso_actual / total_pasos, 1.0)
    return fin + 0.5 * (inicio - fin) * (1.0 + math.cos(math.pi * progreso))


class TestDecaimientoCosenoProbBot:
    """Verifica el decaimiento coseno de prob_bot."""

    def test_inicio_es_prob_bot_inicial(self):
        """Al inicio (paso 0), prob_bot debe ser el valor inicial."""
        resultado = _decaimiento_coseno_prob_bot(
            0, 20_000_000, inicio=0.50, fin=0.20)
        assert resultado == pytest.approx(0.50)

    def test_final_es_prob_bot_final(self):
        """Al final (total_pasos), prob_bot debe ser el valor final (piso)."""
        resultado = _decaimiento_coseno_prob_bot(
            20_000_000, 20_000_000, inicio=0.50, fin=0.20)
        assert resultado == pytest.approx(0.20)

    def test_mitad_es_promedio(self):
        """A la mitad, cos(pi/2)=0, así que es el promedio."""
        resultado = _decaimiento_coseno_prob_bot(
            10_000_000, 20_000_000, inicio=0.50, fin=0.20)
        assert resultado == pytest.approx(0.35)

    def test_coseno_mayor_que_lineal_en_25pct(self):
        """Al 25%, coseno > lineal (mantiene más exposición a bots)."""
        coseno = _decaimiento_coseno_prob_bot(
            5_000_000, 20_000_000, inicio=0.50, fin=0.20)
        lineal = _decaimiento_prob_bot(
            5_000_000, 20_000_000, inicio=0.50, fin=0.20)
        assert coseno > lineal, f"coseno={coseno:.4f} debe > lineal={lineal:.4f}"

    def test_coseno_mayor_que_lineal_en_75pct(self):
        """Al 75%, coseno > lineal (el piso es más alto: 0.20 vs 0.10)."""
        coseno = _decaimiento_coseno_prob_bot(
            15_000_000, 20_000_000, inicio=0.50, fin=0.20)
        lineal = _decaimiento_prob_bot(
            15_000_000, 20_000_000, inicio=0.50, fin=0.10)
        assert coseno > lineal, f"coseno={coseno:.4f} debe > lineal={lineal:.4f}"

    def test_pasado_el_total_no_baja_del_piso(self):
        """Más allá del total, no debe bajar del piso."""
        resultado = _decaimiento_coseno_prob_bot(
            30_000_000, 20_000_000, inicio=0.50, fin=0.20)
        assert resultado == pytest.approx(0.20)

    def test_total_pasos_cero_devuelve_piso(self):
        """Edge case: total_pasos=0 devuelve el piso."""
        resultado = _decaimiento_coseno_prob_bot(0, 0, inicio=0.50, fin=0.20)
        assert resultado == pytest.approx(0.20)


# ============================================================
# Tests de formato de logging
# ============================================================

class TestFormatoEvalJSONL:
    """Verifica el formato de las entradas de evaluación."""

    def test_formato_campos_obligatorios(self):
        """La entrada debe contener todos los campos requeridos."""
        entry = _formato_eval_jsonl(1_000_000, 0.45, 55.3, 100, 0.40)
        assert "paso" in entry
        assert "win_rate_bots" in entry
        assert "avg_score" in entry
        assert "num_partidas" in entry
        assert "prob_bot" in entry
        assert "timestamp" in entry

    def test_win_rate_redondeado_a_4_decimales(self):
        """win_rate debe redondearse a 4 decimales."""
        entry = _formato_eval_jsonl(1_000_000, 0.456789, 50.0, 100, 0.30)
        assert entry["win_rate_bots"] == 0.4568

    def test_entrada_serializable_a_json(self):
        """La entrada debe ser serializable a JSON."""
        entry = _formato_eval_jsonl(5_000_000, 0.72, 42.1, 200, 0.25)
        dumped = json.dumps(entry)
        assert isinstance(dumped, str)
        # Verificar que se puede deserializar
        reloaded = json.loads(dumped)
        assert reloaded["paso"] == 5_000_000
        assert reloaded["win_rate_bots"] == 0.72

    def test_paso_cero_valido(self):
        """Paso 0 es válido (evaluación inicial)."""
        entry = _formato_eval_jsonl(0, 0.25, 75.0, 100, 0.50)
        assert entry["paso"] == 0
        assert entry["win_rate_bots"] == 0.25

    def test_avg_score_redondeado_a_2_decimales(self):
        """avg_score debe redondearse a 2 decimales."""
        entry = _formato_eval_jsonl(1_000_000, 0.50, 42.567, 100, 0.30)
        assert entry["avg_score"] == 42.57


# ============================================================
# Tests de comando ELO
# ============================================================

class TestComandoELO:
    """Verifica la construcción del comando de torneo ELO."""

    def test_comando_contiene_parametros_clave(self):
        """El comando debe contener todos los flags necesarios."""
        cmd = _comando_elo(
            "modelos/v6/snapshots", 5_000_000, 10, 30, "v6/elo_out.txt")
        assert "--directorio modelos/v6/snapshots" in cmd
        assert "--partidas 30" in cmd
        assert "--min-paso 5000000" in cmd
        assert "--max-snapshots 10" in cmd
        assert "--elo-puro" in cmd
        assert "--incluir-bots" in cmd
        assert "> v6/elo_out.txt" in cmd

    def test_comando_redirecciona_salida(self):
        """El comando debe redirigir stdout+stderr al archivo de salida."""
        cmd = _comando_elo("dir", 0, 5, 20, "out.txt")
        assert "2>&1" in cmd
        assert "> out.txt" in cmd

    def test_comando_con_min_paso_cero(self):
        """min_paso=0 debe aceptarse (sin filtro de paso)."""
        cmd = _comando_elo("dir", 0, 5, 20, "out.txt")
        assert "--min-paso 0" in cmd
