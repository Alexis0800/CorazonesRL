#!/usr/bin/env python
"""
Script de juego interactivo (entry point).

Uso:
    python scripts/jugar.py --modelo models/v8/elite/snapshot_XXX.zip
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.jugar_contra_modelo import main  # noqa: E402

if __name__ == "__main__":
    main()
