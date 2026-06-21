# Hoja de Ruta — Modelos de Recompensa para Corazones RL

> **Principio:** Cada versión `vX` es un paquete independiente bajo `src/vX/`.
> Comparten únicamente `src/dominio/` (lógica pura del juego), `src/agentes/` (bots),
> `src/entorno/observacion.py` y `src/entorno/dimensiones.py`.
> Si una versión no convence, se elimina su carpeta sin afectar a las demás.

---

## v1 — Partida Total, 20 Señales Tácticas

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/entorno/`, `src/entrenamiento/` |
| **Episodio** | Partida completa (múltiples manos hasta 100 pts) |
| **Recompensas** | 20 señales tácticas por baza + posición final |
| **Observación** | 220-dim con contexto de partida |
| **Estado** | ✅ **Estable.** Mejor snapshot: v22 (1632 Elo, 33% wr_experto) |
| **Problema conocido** | Value head colapsa ~1.5M pasos (20 señales → regresión compleja) |

---

## v2 — Ronda, Score Puro (← AHORA)

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/v2_ronda/` |
| **Episodio** | **Una sola mano** (max 52 steps) |
| **Recompensas** | 4 señales: per-baza (-1 ♡, -13 Q♠), fin de mano (`26 - mis_puntos`), pozo (+78), posición (±5) |
| **Observación** | 220-dim (dims de partida en 0: sin contexto multi-mano) |
| **Hipótesis** | "Minimizar puntos propios sin ruido de rivales → política más robusta que v1" |
| **Métrica objetivo** | wr_experto ≥ 33% (igualar v22), sin colapso de value head |

### Diseño de recompensa v2

```
REWARD_CORAZON      = -1     (por baza, shaping denso)
REWARD_DAMA_PICAS   = -13    (por baza, shaping denso)
REWARD_SHOOTING_MOON = 78    (fin de mano, neto +52 tras per-baza)
REWARD_MEJOR_MANO   = +5     (fin de mano, bonus posicional)
REWARD_PEOR_MANO    = -5     (fin de mano, penalty posicional)

reward_fin_mano = (26 - mis_puntos)   # rango [0, 26]
```

Tabla de rewards netos por mano:

| Mano | Per-baza | Fin mano | Posición | **Neto** |
|------|----------|----------|----------|----------|
| 0 pts, mejor | 0 | +26 | +5 | **+31** |
| 3♡, medio | -3 | +23 | 0 | **+20** |
| Q♠+2♡, peor | -15 | +11 | -5 | **-9** |
| Pozo exitoso | -26 | +78 | — | **+52** |
| 26 pts sin pozo | -26 | 0 | -5 | **-31** |

---

## v3 — Ronda, Score + Delta (planificado)

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/v3_ronda_delta/` |
| **Episodio** | Una sola mano con puntajes históricos **simulados** |
| **Recompensas** | Base v2 + bonos por diferencial de puntos |
| **Hipótesis** | "El contexto de partida (diferenciales, near-100) mejora la estrategia multi-mano" |

### Señales adicionales planeadas

```
Bonus:  +X si bajo mi posición relativa vs el líder
Penalty: -X si alguien está cerca de 100 y le doy puntos
Penalty: -X si voy ganando por mucho y arriesgo innecesariamente
```

Las features [172:198] del vector de observación se activan con valores simulados.

---

## v4 — Ronda, Suma Cero (planificado)

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/v4_ronda_zero_sum/` |
| **Episodio** | Una sola mano con puntajes históricos **simulados** |
| **Recompensas** | Suma cero estricta: `mi_reward = -(suma rewards rivales)` |
| **Hipótesis** | "La simetría matemática de suma cero produce políticas más robustas que score-only" |

### Nota sobre suma cero

En Hearts real:

- El pozo mete 26 pts a cada rival → diferencial de +78 para el agente, -78 total para rivales
- Una mano de 0 pts da ~8.7 de ventaja promedio sobre cada rival
- La suma cero fuerza que el modelo internalice que "mi ganancia = pérdida de rivales"

---

## Comparativa rápida

| | v1 (partida) | v2 (score) | v3 (delta) | v4 (zero-sum) |
|---|---|---|---|---|
| Episodio | Multi-mano | 1 mano | 1 mano | 1 mano |
| Nº señales | 20 | 4 | 6-8 | 4 |
| Ruido rivales | Alto (tácticas) | Ninguno | Bajo | Ninguno |
| Value head | Colapsa | ¿Estable? | ¿Estable? | ¿Estable? |
| Complejidad | Alta | **Mínima** | Media | Media |

---

## Criterios de éxito por versión

Cada versión debe superar a la anterior en **al menos uno** de estos:

1. **wr_experto** > mejor versión anterior (actual: 33% v22)
2. **Elo** > mejor versión anterior (actual: 1632 v22)
3. **Estabilidad**: value_loss sin picos > 0.05 durante todo el entrenamiento
4. **Simplicidad**: menos señales de recompensa que la versión anterior

---

## Reglas de experimentación

- Cada versión tiene su propio `train.py` que importa su entorno específico
- El dataset BC se regenera si el reward cambia (necesita acciones alineadas)
- Los torneos Elo siempre incluyen bots como baseline fijo
- Early stopping: 3 evaluaciones consecutivas con wr_experto bajando → detener
