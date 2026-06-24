"""
Entorno Gymnasium v4 — Terminal-only rewards + 228 dimensiones.

Reutiliza CorazonesEnvV31 de v3_1 (que ya tiene terminal-only rewards).
La clase se re-exporta como CorazonesEnvV4 para claridad de versión.
"""

from __future__ import annotations

from src.v3_1.entorno import CorazonesEnvV31 as CorazonesEnvV4

__all__ = ["CorazonesEnvV4"]
