"""Fixtures y configuración global para todos los tests."""
import os
import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tests que requieren entrenamiento real (lentos)"
    )


# Asegurar que la raíz del proyecto está en sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
