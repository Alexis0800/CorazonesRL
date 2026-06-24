"""Dimensiones de observación para v3.1.

Vector depurado de 228 dimensiones:
  [0:52]     Mano del agente (one-hot)
  [52:104]   Mesa actual (one-hot)
  [104:156]  Cementerio (one-hot)
  [156:172]  Vacíos (4 jug × 4 palos)
  [172:176]  Puntaje histórico /100
  [176:180]  Puntos mano actual (raw 0-26)
  [180]      Corazones rotos
  [181]      Posición en baza
  [182:187]  Q♠ tracker (5 estados)
  [187]      pozo_viable
  [188]      Número de baza /13
  [189:193]  Cartas restantes por palo (raw 0-13)
  [193]      Palo de salida
  [194:198]  Peligro Q♠ por palo
  [198]      Peligro Q♠ inminente
  [199:203]  Prob Q♠ por jugador (usa voids)
  [203:207]  Altas en mi mano (J/Q/K/A) por palo
  [207:211]  Máxima absoluta por palo
  [211:215]  Control de palo (dominancia)
  [215:219]  ♥ altos (J/Q/K/A) capturados por rival
  [219:223]  ♠ altas (J/Q/K/A) jugadas por rival
  [223:227]  ¿Ya jugó ♥ cada rival?
  [227]      Forzado (1 sola carta legal)

Eliminados vs v3 (265): duplicados, features inferibles, y bajo valor.
Añadidos vs v3: patrones de rivales (12 dims).
"""

from __future__ import annotations

DIM_V3_1: int = 228

# Dimensiones válidas para compatibilidad con el ecosistema
DIMS_VALIDAS: frozenset[int] = frozenset({
    190,   # DIM_V5
    194,   # DIM_V6
    220,   # DIM_V10 / DIM_ENTORNO / DIM_ENTRENAMIENTO
    228,   # DIM_V3_1
    265,   # DIM_V3
})

__all__ = ["DIM_V3_1", "DIMS_VALIDAS"]
