"""
Pruebas unitarias para el sistema de diagnóstico automático de entrenamiento.

Cubre:
  - _capturar_diagnosticos: extrae métricas del logger de SB3.
  - _alertas_diagnostico: genera alertas cuando hay valores peligrosos.
  - Formato de entrada en eval_log.jsonl.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock


# ============================================================
# Helpers simulados
# ============================================================

class _FakeLogger:
    """Simula el logger de SB3."""

    def __init__(self, values=None):
        self.name_to_value = values or {}

    def get_dir(self):
        return "/fake/logdir"


# ============================================================
# Tests de captura de diagnósticos
# ============================================================

class TestCapturarDiagnosticos:
    """Verifica la extracción de métricas del logger de SB3."""

    def test_extrae_entropy_loss(self):
        """Debe extraer entropy_loss del logger."""
        from train import _capturar_diagnosticos
        modelo = MagicMock()
        modelo.logger = _FakeLogger({"train/entropy_loss": -0.35})
        diag = _capturar_diagnosticos(modelo)
        assert "entropy_loss" in diag
        assert diag["entropy_loss"] == -0.35

    def test_extrae_value_loss(self):
        """Debe extraer value_loss del logger."""
        from train import _capturar_diagnosticos
        modelo = MagicMock()
        modelo.logger = _FakeLogger({"train/value_loss": 2.5})
        diag = _capturar_diagnosticos(modelo)
        assert diag["value_loss"] == 2.5

    def test_extrae_approx_kl(self):
        """Debe extraer approx_kl del logger."""
        from train import _capturar_diagnosticos
        modelo = MagicMock()
        modelo.logger = _FakeLogger({"train/approx_kl": 0.015})
        diag = _capturar_diagnosticos(modelo)
        assert diag["approx_kl"] == 0.015

    def test_extrae_clip_fraction(self):
        """Debe extraer clip_fraction del logger."""
        from train import _capturar_diagnosticos
        modelo = MagicMock()
        modelo.logger = _FakeLogger({"train/clip_fraction": 0.12})
        diag = _capturar_diagnosticos(modelo)
        assert diag["clip_fraction"] == 0.12

    def test_logger_vacio_retorna_dict_vacio(self):
        """Si el logger no tiene datos, retorna dict vacío."""
        from train import _capturar_diagnosticos
        modelo = MagicMock()
        modelo.logger = _FakeLogger({})
        diag = _capturar_diagnosticos(modelo)
        assert diag == {}

    def test_logger_sin_atributo_no_crashea(self):
        """Si el modelo no tiene logger.name_to_value, no crashea."""
        from train import _capturar_diagnosticos
        modelo = MagicMock()
        modelo.logger = MagicMock()
        del modelo.logger.name_to_value
        diag = _capturar_diagnosticos(modelo)
        assert diag == {}

    def test_ignora_metricas_no_relevantes(self):
        """Solo extrae métricas con prefijo train/."""
        from train import _capturar_diagnosticos
        modelo = MagicMock()
        modelo.logger = _FakeLogger({
            "train/entropy_loss": -0.1,
            "time/fps": 500,
            "rollout/ep_len_mean": 52,
            "train/value_loss": 1.0,
        })
        diag = _capturar_diagnosticos(modelo)
        assert "entropy_loss" in diag
        assert "value_loss" in diag
        assert "fps" not in diag
        assert "ep_len_mean" not in diag


# ============================================================
# Tests de alertas de diagnóstico
# ============================================================

class TestAlertasDiagnostico:
    """Verifica el sistema de alertas automáticas."""

    def test_alerta_entropy_muy_negativa(self):
        """entropy_loss < -0.5 debe generar alerta."""
        from train import _alertas_diagnostico
        alertas = _alertas_diagnostico({"entropy_loss": -0.63})
        assert len(alertas) >= 1
        assert any("entropía" in a.lower() for a in alertas)

    def test_no_alerta_entropy_normal(self):
        """entropy_loss > -0.3 no debe generar alerta."""
        from train import _alertas_diagnostico
        alertas = _alertas_diagnostico({"entropy_loss": -0.25})
        assert len(alertas) == 0

    def test_alerta_approx_kl_alto(self):
        """approx_kl > 0.03 debe generar alerta."""
        from train import _alertas_diagnostico
        alertas = _alertas_diagnostico({"approx_kl": 0.05})
        assert len(alertas) >= 1
        assert any("kl" in a.lower() for a in alertas)

    def test_alerta_clip_fraction_alto(self):
        """clip_fraction > 0.5 debe generar alerta."""
        from train import _alertas_diagnostico
        alertas = _alertas_diagnostico({"clip_fraction": 0.7})
        assert len(alertas) >= 1
        assert any("clip" in a.lower() for a in alertas)

    def test_dict_vacio_no_alertas(self):
        """Sin métricas, sin alertas."""
        from train import _alertas_diagnostico
        alertas = _alertas_diagnostico({})
        assert alertas == []

    def test_todas_metricas_normales_sin_alertas(self):
        """Todas las métricas en rangos normales no generan alertas."""
        from train import _alertas_diagnostico
        alertas = _alertas_diagnostico({
            "entropy_loss": -0.2,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
            "value_loss": 1.5,
        })
        assert alertas == []
