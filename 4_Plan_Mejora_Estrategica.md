# Plan de Mejora Estratégica — Modelo RL de Corazones (v5)

**Fecha:** 2026-06-11
**Basado en:** Observaciones de partidas interactivas (`observaciones_modelo.md`)
**Modelo actual:** MaskablePPO, 8M steps, 86% win rate contra bots, ~4º lugar contra solitar.io difícil

---

## 1. Diagnóstico

### 1.1 Qué funciona

| Capacidad | Estado |
|---|---|
| Trackeo de voids y cementerio | ✅ Bueno |
| Control de cierre de mano | ✅ Correcto (Caso 2) |
| Seguir reglas (legales, palo, etc.) | ✅ Perfecto |
| Aprendizaje base contra bots | ✅ 86% win rate |

### 1.2 Qué falla

| Error | Causa raíz | Impacto |
|---|---|---|
| Tirar Q♠ sin pozo viable (Casos 1 y 3) | No distingue modo pozo vs minimizar | +13-26 pts/partida |
| No adaptar estrategia según puntuación | Solo recompensa terminal | Decisiones miopes |
| No planificar a nivel de mano completa | PPO es reactivo (baza a baza) | Pierde oportunidades |

### 1.3 Por qué el 86% es engañoso

El win rate mide desempeño contra **los mismos 3 bots heurísticos del entrenamiento**.
Los bots (conservador, agresivo, evasivo) no hacen:

- Planificación multi-baza
- Bloqueo de pozo
- Adaptación al puntaje histórico

Contra oponentes que sí hacen estas cosas (solitar.io difícil), el modelo colapsa.

---

## 2. Arquitectura Propuesta: Dos Niveles de Decisión

La idea de separar táctica y estrategia es correcta y está alineada con
*Hierarchical Reinforcement Learning* (HRL). En Corazones, esto se traduce en:

```
┌─────────────────────────────────────────┐
│         NIVEL ESTRATÉGICO               │
│  (1 decisión por mano o por evento)     │
│                                         │
│  Input:  puntajes históricos            │
│          mano actual (corazones altos)  │
│          voids, 💔, Q♠, bazas restantes │
│                                         │
│  Output: MODO de juego                  │
│     • MINIMIZAR — evitar todos los pts  │
│     • POZO — intentar shooting the moon │
│     • ALIMENTAR_X — dar pts a jugador X │
│       (para que pierda y vos zafes)     │
└──────────────────┬──────────────────────┘
                   │ modo (como feature adicional)
                   ▼
┌─────────────────────────────────────────┐
│         NIVEL TÁCTICO                   │
│  (1 decisión por baza)                  │
│                                         │
│  Input:  observación 187-dim actual     │
│          + features de modo estratégico │
│                                         │
│  Output: qué carta jugar (0-51)         │
└─────────────────────────────────────────┘
```

### 2.1 ¿Dos modelos separados o uno solo?

**Recomendación: un solo modelo con features estratégicas adicionales.**

| Enfoque | Ventaja | Desventaja |
|---|---|---|
| 2 modelos | Separación clara de responsabilidades | Más complejo, doble entrenamiento |
| 1 modelo + features | Simple, PPO aprende la transición | La red debe ser un poco más grande |

El nivel estratégico en Corazones es lo suficientemente simple (3-4 modos)
como para codificarlo en features binarios en vez de un modelo separado.
PPO aprenderá a usar esos features como "interruptores" de estrategia.

### 2.2 Features estratégicas a agregar

Se agregan al vector de observación (actualmente 187 dims → ~195 dims):

```python
# Feature 188: pozo_viable (bool)
# ¿Es REALMENTE posible hacer pozo?
corazones_en_mano = sum(1 for c in mano if palo == HEARTS)
corazones_altos = sum(1 for c in mano if palo == HEARTS and valor >= 11)  # J,Q,K,A
pozo_viable = (
    not corazones_rotos
    and corazones_en_mano >= 6       # mayoría de corazones
    and corazones_altos >= 3         # control de corazones altos
    and puntaje_historico[agente] < 80  # margen para fallar
)

# Feature 189: debo_arriesgar (bool)
# ¿Estoy tan atrás que debo arriesgarme?
debo_arriesgar = (
    puntaje_historico[agente] > 75
    and any(p < 30 for p in puntaje_historico if p != puntaje_historico[agente])
)

# Feature 190: puedo_alimentar (bool)
# ¿Puedo darle puntos a un jugador para que pierda?
jugador_cerca_de_100 = max(puntaje_historico) > 85
puedo_alimentar = jugador_cerca_de_100 and puntaje_historico[agente] < 70

# Features 191-194: one-hot del MODO estratégico recomendado
# [MINIMIZAR, POZO, ALIMENTAR, INDEFINIDO]
# Solo uno es 1.0, los demás 0.0
```

### 2.3 Correcciones de recompensa (reward shaping)

Las recompensas actuales no penalizan errores estratégicos documentados.
Se agregan:

```python
# === NUEVAS RECOMPENSAS v5 ===

# Penalización: Q♠ cuando el pozo NO es viable
REWARD_Q_SPADES_SIN_POZO: float = -8.0
# Se aplica si: el agente gana una baza con Q♠ y
#   (corazones_rotos OR corazones_en_mano < 6)

# Penalización: ganar baza con corazones cuando el modo es MINIMIZAR
REWARD_GANAR_BAZA_CON_CORAZON: float = -3.0
# Se aplica si: el agente gana una baza que contiene ≥1 corazón
#   y pozo_viable == False

# Penalización por puntos acumulados en la mano (suave)
REWARD_POR_PUNTO_EN_MANO: float = -0.2
# Se aplica al final de la mano: reward += puntos_mano[agente] * REWARD_POR_PUNTO_EN_MANO
# Esto da ~ -5.2 por una mano normal (26 pts), manejable
```

**Las recompensas terminales (+500 1º, +200 2º, -200 3º, -500 4º) se mantienen sin cambios.**
Son la señal dominante; las nuevas recompensas solo guían al modelo
para que no cometa los errores documentados.

---

## 3. Recolección de Datos de Partidas Reales

### 3.1 Objetivo

Los datos de solitar.io **no son para entrenar** al modelo, sino para:

1. **Encontrar patrones de error** — como hiciste manualmente con los Casos 1-3.
2. **Comparar decisiones** — ¿qué habría hecho el modelo vs. qué hizo el bot difícil?
3. **Generar casos de prueba** — escenarios concretos para tests unitarios.

### 3.2 Método: grabación manual + análisis automático

```
┌─────────────────────────────────────────────────────┐
│  Flujo de captura de partidas                        │
│                                                       │
│  1. Jugás en solitar.io/corazones (modo difícil)     │
│     mientras tenés asesor_partida.py abierto          │
│     en otra ventana como "consultor".                │
│                                                       │
│  2. asesor_partida.py ya trackea todo el estado.     │
│     Cada vez que diferís de la recomendación,         │
│     queda registrado en la UI.                        │
│                                                       │
│  3. Opcional: agregar flag --log para que             │
│     asesor_partida.py guarde cada baza en un .jsonl   │
│     (observación, legales, recomendación, tu jugada,  │
│      resultado). Para análisis post-partida.          │
└─────────────────────────────────────────────────────┘
```

### 3.3 ¿Conectar un bot a solitar.io?

**No se recomienda.** Hay tres razones:

1. **Técnico:** solitar.io usa un DOM dinámico. Mapear el estado del juego desde HTML a tu vector de 187 dimensiones es frágil y se rompe si cambian la UI.
2. **Legal:** Los ToS de la mayoría de estos sitios prohíben automatización/bots.
3. **Utilidad:** El modelo se entrena en tu `CorazonesEnv`, que ya es un simulador perfecto. No necesitás datos externos para entrenar; necesitás **mejorar el simulador** (recompensas, features) y **mejorar los oponentes de entrenamiento** (self-play más avanzado).

El valor de solitar.io es como **herramienta de evaluación manual** — jugar partidas, observar errores, documentarlos, y luego codificar la solución en el entorno de entrenamiento.

---

## 4. Plan de Implementación (3 fases)

### Fase 5A: Corrección de recompensas y features estratégicas

**Objetivo:** Que el modelo aprenda a no tirar Q♠ cuando el pozo es inviable.

**Tareas:**

- [ ] Agregar 8 features estratégicas al vector de observación (188→~195 dims).
  - Archivo: `src/entorno.py` → método `_construir_observacion()`.
- [ ] Agregar 3 nuevas recompensas de reward shaping.
  - Archivo: `src/entorno.py` → método `step()`.
- [ ] Escribir tests unitarios para las nuevas features y recompensas.
  - Archivo: `tests/test_entorno_v5.py`.
- [ ] Actualizar `construir_observacion_parcial()` en `asesor_carta.py` para que el REPL también use las nuevas features.
- [ ] Actualizar VecNormalize (nuevas dimensiones requieren nuevas estadísticas).

**DoD:**

- Tests unitarios pasan para features estratégicas.
- Tests unitarios pasan para nuevas recompensas (verificar que Q♠ con pozo inviable da -8.0).
- `evaluar_modelo.py` sigue funcionando con el nuevo espacio de observación.

### Fase 5B: Self-Play mejorado (oponentes más fuertes)

**Objetivo:** Que el modelo entrene contra versiones cada vez más fuertes de sí mismo, no contra bots fijos.

**Cambios respecto al self-play actual:**

| Aspecto | v4 (actual) | v5 (propuesto) |
|---|---|---|
| % bots heurísticos | 40% | **15%** (menos anclaje) |
| % snapshots históricos | 60% | **85%** (más auto-juego) |
| Pool máximo | 50 snapshots | **100 snapshots** |
| Snapshot mínimo | 200K steps | **500K steps** (más maduros) |
| Pesos de selección | Exponenciales | Exponenciales (sin cambio) |

**Tareas:**

- [ ] Ajustar constantes `PROB_BOT_V2`, `MAX_SNAPSHOTS_POOL`, `MIN_SNAPSHOT_STEPS` en `train_self_play.py`.
- [ ] Agregar lógica de pruning del pool: eliminar el snapshot más débil cuando se excede `MAX_SNAPSHOTS_POOL`.

**DoD:**

- El pool de oponentes crece hasta 100 snapshots sin errores de memoria.
- El entrenamiento no colapsa (win rate no cae a <25%).

### Fase 5C: Entrenamiento y evaluación

**Objetivo:** Re-entrenar desde snapshot base con las mejoras y medir el impacto.

**Tareas:**

- [ ] Lanzar entrenamiento desde `modelos_historicos/v2/snapshot_0004500000` (o desde cero si las features nuevas lo requieren).
- [ ] Entrenar 5-10M steps con las nuevas recompensas y features.
- [ ] Evaluar cada 1M steps con `evaluar_modelo.py --partidas 200`.
- [ ] Probar interactivamente con `asesor_partida.py` en los escenarios de los Casos 1 y 3.
- [ ] Jugar partidas en solitar.io para comparar desempeño.

**DoD:**

- Win rate contra bots ≥ 80%.
- En escenarios de Caso 1 y Caso 3, el modelo **no recomienda Q♠**.
- Puntuación promedio en `evaluar_modelo.py` baja respecto al modelo v4.

---

## 5. Resumen Visual

```
               AHORA (v4)                    PROPUESTO (v5)
               ─────────                     ──────────────
Estrategia     Implícita en PPO              Features explícitas
               ("aprende solo")              (pozo_viable, debo_arriesgar, etc.)

Recompensa     Solo terminal (+500/1º)       Terminal + shaping
                                             (-8 por Q♠ sin pozo, -0.2/pt en mano)

Oponentes      60% snapshots + 40% bots      85% snapshots + 15% bots
               Pool ≤ 50                     Pool ≤ 100, ≥500K steps c/u

Observación    187 dimensiones               ~195 dimensiones
                                             (8 features estratégicas nuevas)

Resultado      86% vs bots                   80%+ vs bots
esperado       4º vs solitar.io              ≥2º vs solitar.io (objetivo)
```

---

## 6. Preguntas para decidir

Antes de empezar a codificar, necesito tu confirmación en estos puntos:

| # | Pregunta |
|---|---|
| 1 | ¿Arrancamos con Fase 5A (features + recompensas)? Es la de mayor impacto. |
| 2 | ¿El modelo se re-entrena desde cero o desde el snapshot base `0004500000`? Desde cero es más limpio (las features cambiaron) pero tarda más. Desde snapshot aprovecha lo aprendido pero las features extra empiezan en cero. |
| 3 | ¿Querés que agregue el flag `--log` a `asesor_partida.py` para guardar logs de partidas en JSONL? |
| 4 | ¿Las features de modo estratégico las calculo como heurístico (fórmula) o preferís un modelo pequeño aparte? Recomiendo heurístico para la v5. |
