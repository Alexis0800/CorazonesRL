"""
v2_1 — Entorno de Corazones por ronda individual con reward shaping táctico.

Extiende v2_ronda con 4 nuevas señales de recompensa por ronda que enseñan
las estrategias que BotExperto codifica explícitamente:

  1. Q♠ dump: +12 cuando descartas Q♠ siendo void y un rival la captura
  2. Moon block: +15 cuando ganas una baza con puntos y un rival tiene ≥6♥
  3. Early safe burn: +1.5 por ganar baza con 0 puntos en bazas 1-7
  4. Liability hold: -5 tras baza 7 si retienes A♠/K♠ con Q♠ activa

Una mano = un episodio. Sin puntajes históricos. Sin contexto multi-mano.

Paquete independiente. Comparte dominio, agentes, y observación con v1/v2.
"""
