"""
Interfaz de línea de comandos — Entry points para el usuario.
"""
from src.cli.evaluar import ejecutar_evaluacion
from src.cli.jugar import ejecutar_juego

__all__ = ["ejecutar_evaluacion", "ejecutar_juego"]
