# Hoja de Ruta — Modelos de Recompensa para Corazones RL

> **Principio:** Cada versión `vX` es un paquete independiente bajo `src/vX/`.
> Comparten únicamente `src/dominio/` (lógica pura del juego), `src/agentes/` (bots),
> `src/entorno/observacion.py` y `src/entorno/dimensiones.py`.
> Si una versión no convence, se elimina su carpeta sin afectar a las demás.
>
> **Modelos legacy:** `models/v1_antiguo/` contiene snapshots de v1 (v5_golden..v28).

---

## v1 — Partida Total, 20 Señales Tácticas

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/entorno/`, `src/entrenamiento/` |
| **Episodio** | Partida completa (múltiples manos hasta 100 pts) |
| **Recompensas** | 20 señales tácticas por baza + posición final |
| **Observación** | 220-dim con contexto de partida |
| **Estado** | ✅ **Congelada.** Mejor snapshot: v22 (1632 Elo, 33% wr_experto) |
| **Problema conocido** | Value head colapsa ~1.5M pasos (20 señales → regresión compleja) |

---

## v2 — Ronda, Score Puro (← ACTUAL)

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/v2_ronda/`, `src/v2_1/` |
| **Episodio** | **Una sola mano** (max 52 steps) |
| **Recompensas** | v2_ronda: 4 señales base. v2_1: +4 tácticas (Q♠ dump, moon block, early burn, liability) |
| **Observación** | 220-dim (dims de partida en 0: sin contexto multi-mano) |
| **Arquitectura** | MLP [512,512,256] sin memoria entre bazas |
| **Estado** | ✅ Mejor snapshot: v2_1 (1535 Elo, ~30% wr_experto) |
| **Problema raíz** | **~45% error rate.** MLP no trackea fallos, no planifica, no gestiona Q♠ |

### Diseño de recompensa v2

```
REWARD_CORAZON      = -1     (por baza, shaping denso)
REWARD_DAMA_PICAS   = -13    (por baza, shaping denso)
REWARD_SHOOTING_MOON = 78    (fin de mano, neto +52 tras per-baza)
REWARD_MEJOR_MANO   = +5     (fin de mano, bonus posicional)
REWARD_PEOR_MANO    = -5     (fin de mano, penalty posicional)

reward_fin_mano = (26 - mis_puntos)   # rango [0, 26]
```

### Diagnóstico PIMC (16 manos exhaustivas, seed=42)

| Métrica | v2_1 |
|---------|------|
| Error rate global | **44.9%** (71/158 decisiones) |
| Coste total | 156.2 pts en 16 manos |
| Error rate early (bazas 1-4) | ~48%, coste medio 2.96 pts/error |
| Error rate mid (bazas 5-8) | ~46%, coste medio 4.84 pts/error |
| Error rate late (bazas 9-13) | ~30%, coste medio 1.71 pts/error |
| Error #1 | **LIDERAR mal** — jugar Q♠ cuando hay opción segura |
| Causas raíz | `planning` (33%), `risk_assessment` (46%), `card_counting` (17%) |

**Conclusión:** Errores tempranos son MÁS costosos que tardíos. MLP sin memoria
no puede trackear fallos entre bazas ni planificar a largo plazo.

---

## v3 — Arquitectura con Memoria + Observación Enriquecida (← AHORA)

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/v3/` |
| **Episodio** | Una sola mano (hereda de v2) |
| **Recompensas** | v2_1 (8 señales: 4 base + 4 tácticas) — sin cambios |
| **Objetivo** | Corregir las 3 causas raíz del diagnóstico |

### Tres mejoras simultáneas

#### A) Transformer Feature Extractor — Memoria entre bazas

Reemplaza el MLP `[512,512,256]` por un Transformer Encoder que procesa
la secuencia de 13 bazas como tokens temporales.

```
Observation(220) → Linear(256) → TransformerEncoder(4 layers, 4 heads, d=256)
                                 → Mean Pooling → Linear(256) → features_dim(256)
```

- **Memoria contextual**: Cada baza ve las anteriores vía self-attention.
- **Resuelve `planning`**: El modelo puede atender a bazas pasadas para decidir.
- **Resuelve `card_counting`**: El attention trackea qué cartas se jugaron.
- **Resuelve `void_tracking`**: Patrones de fallos visibles en la secuencia.

#### B) MCTS-Guided Training — Expert Iteration

Entrena con targets del oráculo PIMC en bazas clave (≥8) para mejorar
la calidad de decisiones tardías sin costo de inferencia en producción.

```
Para cada episodio:
  1. Self-play normal (modelo actual)
  2. Si baza ≥ 8: ejecutar PIMC exacto → obtener acción óptima
  3. Guardar (obs, acción_optima_pimc) en buffer BC
  4. Cada N pasos: BC fine-tuning con buffer + RL normal
```

- **Corrige `risk_assessment`**: PIMC ve todos los desenlaces posibles.
- **Sin costo en inferencia**: Solo se usa en entrenamiento.
- **Parámetros**: `pimc_every=4` bazas, `bc_weight=0.3`, buffer size=100K.

#### C) Observación Enriquecida — Features de tracking explícito

Añade 30 dimensiones al vector de observación (220 → 250) con features
que el MLP actual no puede inferir:

| Rango | Feature | Descripción |
|-------|---------|-------------|
| [220:224] | `cartas_restantes` | Cartas sin jugar por palo (valor real, no /13) |
| [224:228] | `peligro_qs` | Probabilidad Q♠ × peligrosidad del palo |
| [228:232] | `cartas_altas_mano` | J/Q/K/A en mi mano por palo |
| [232:236] | `riesgo_baza` | Puntos esperados si gano esta baza |
| [236:240] | `control_palo` | ¿Soy dominante en este palo? (#altas en mano / #altas restantes) |
| [240:244] | `oportunidad_descarte` | ¿Puedo vaciarme de un palo esta baza? |
| [244] | `bazas_restantes` | 13 - baza_actual |
| [245:249] | `puntos_rivales` | Puntos acumulados esta mano por cada rival |

- **Resuelve `risk_assessment` parcialmente**: Features explícitas de riesgo.
- **Ayuda al Transformer**: Features precomputadas = menos carga de inferencia.

### Métricas objetivo v3

| Métrica | v2_1 (actual) | v3 (objetivo) |
|---------|---------------|---------------|
| wr_experto | ~30% | ≥ **50%** |
| Elo | 1535 | ≥ **1700** |
| Error rate PIMC | 45% | ≤ **25%** |
| Error rate late (bazas 9-13) | 30% | ≤ **12%** |
| Coste medio/mano | 9.8 pts | ≤ **5 pts** |

---

## v4 — Ronda, Suma Cero + Contexto Multi-Mano (planificado)

| Aspecto | Valor |
|---------|-------|
| **Ubicación** | `src/v4/` |
| **Episodio** | Partida multi-mano con memoria entre manos |
| **Recompensas** | Suma cero estricta: `mi_reward = -(suma rewards rivales)` |
| **Hipótesis** | "Si v3 domina una mano, v4 aprende estrategia de partida completa" |

---

## Comparativa rápida

| | v1 (partida) | v2 (score) | v3 (memoria) | v4 (zero-sum) |
|---|---|---|---|---|
| Episodio | Multi-mano | 1 mano | 1 mano | Multi-mano |
| Nº señales | 20 | 4-8 | 8 | 8+zero-sum |
| Arquitectura | MLP | MLP | **Transformer** | Transformer |
| Observación | 220 | 220 | **250** | 250+contexto |
| MCTS-guided | No | No | **Sí** | Sí |
| Value head | Colapsa | Estable | Estable | Estable |
| wr_experto | 33% | 30% | **≥50%** | ≥55% |

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
- **Nuevo:** PIMC como oráculo de diagnóstico cada 500k pasos (eval_log.jsonl)
