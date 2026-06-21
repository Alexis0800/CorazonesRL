#!/usr/bin/env python
"""
Script de entrenamiento autónomo (entry point).

Uso:
    python scripts/entrenar.py --total-steps 20000000 --output-dir models/v9
    python scripts/entrenar.py --resume models/v8/elite/snapshot_XXX --total-steps 25000000
"""
from train import main
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Delegar al pipeline legacy (migración gradual a src.entrenamiento)

if __name__ == "__main__":
    main()
