#!/usr/bin/env python
"""
Script de evaluación de modelos (entry point).

Uso:
    python scripts/evaluar.py --modelo models/v7/snapshots/snapshot_0014900000.zip
    python scripts/evaluar.py --modelo v5_golden.zip --partidas 500
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluar_modelo import main  # noqa: E402

if __name__ == "__main__":
    main()
