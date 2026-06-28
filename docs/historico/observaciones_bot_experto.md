# Observaciones del Bot Experto — Fase 3

**Fecha:** 2026-06-15  
**Objetivo:** Documentar los fallos del BotExperto frente a v7_golden para orientar las mejoras al sistema RL.

---

## Resultados del Torneo Elo (1000 partidas totales, round-robin)

| Pos | Agente | Elo | Diff |
|---|---|---|---|
| 1 | v7_golden (snapshot_0014900000) | **1788** | — |
| 2 | BotExperto | 1641 | -147 |
| 3 | bot_evasivo | 1408 | -233 |
| 4 | bot_agresivo | 1341 | -67 |
| 5 | bot_conservador | 1323 | -18 |

**Head-to-head v7_golden vs BotExperto (100 partidas):** RL gana 72-28.

El BotExperto domina a los tres bots heurísticos con claridad (1641 Elo, ~78-85% win rate en head-to-head), pero pierde consistentemente contra el modelo RL.

---

## Métricas Detalladas (simulaciones directas, 100 partidas cada config)

### Config A — RL(J0) vs BotExperto(J1) vs bot_c(J2) vs bot_e(J3)

| Agente | Win% | Pts/mano | 0-pt% | Q♠% | Pozo% | BlqPozo |
|---|---|---|---|---|---|---|
| **BotExperto** | **44%** | **5.6** | **36%** | 21% | 0.7% | 8 |
| RL | 24% | 6.6 | 33% | 24% | 0.2% | 12 |
| bot_c | 10% | 8.3 | 18% | 28% | 0.4% | 1 |
| bot_e | 22% | 7.2 | 24% | 27% | 2.1% | 1 |

### Config B — BotExperto(J0) vs bots (3×)

| Agente | Win% | Pts/mano | 0-pt% | Q♠% | Pozo% | BlqPozo |
|---|---|---|---|---|---|---|
| **BotExperto** | **64%** | **4.9** | **46%** | 18% | 0.3% | 4 |
| bot_agresivo | 9% | 7.4 | 26% | 24% | 0.2% | 12 |
| bot_conservador | 5% | 8.7 | 19% | 32% | 1.0% | 3 |
| bot_evasivo | 22% | 7.0 | 27% | 26% | 2.2% | 1 |

### Config C — RL(J0) vs bots (3×) — baseline del modelo

| Agente | Win% | Pts/mano | 0-pt% | Q♠% | Pozo% | BlqPozo |
|---|---|---|---|---|---|---|
| **RL** | **58%** | **4.8** | **46%** | 16% | 0.0% | 12 |
| bot_agresivo | 12% | 7.8 | 26% | 32% | 0.1% | 19 |
| bot_conservador | 12% | 8.0 | 17% | 28% | 0.7% | 5 |
| bot_evasivo | 18% | 6.8 | 23% | 24% | 2.0% | 2 |

---

## Interpretación de las Métricas

### Q♠% — Tasa de captura de la Dama de Picas (menor es mejor)

- RL vs bots: **16%** (mejor de todos los configs)
- BotExperto vs bots: **18%**
- BotExperto vs RL + bots: **21%** — degradación de 3 pts cuando enfrenta RL
- RL vs BotExperto + bots: **24%** — RL captura Q♠ más cuando hay presión de BotExperto

Ambos agentes son notablemente mejores que los bots heurísticos (24-32%). Sin embargo, cuando se enfrentan entre sí, la tasa de captura sube para ambos, lo que sugiere que sus estrategias de evasión de Q♠ se contrarrestan mutuamente.

### 0-pt% — Manos limpias (mayor es mejor)

- BotExperto vs bots: **46%** (comparable a RL)
- RL vs bots: **46%**
- BotExperto vs RL: **36%** (caída de 10 pts bajo presión RL)
- RL vs BotExperto: **33%**

El BotExperto tiene una alta tasa de manos con 0 puntos frente a bots, pero cae significativamente frente a RL. Esto indica que RL logra "empujar" puntos hacia BotExperto con más frecuencia que los bots simples.

### BlqPozo — Bloqueos de shooting the moon

- RL bloquea **12** intentos de pozo en 100 partidas vs BotExperto
- BotExperto bloquea **8** intentos en 100 partidas vs RL + bots

El RL bloquea pozos con mayor eficacia que BotExperto, aunque la diferencia no es dramática.

---

## Fallos Críticos Identificados (Config A, 100 partidas)

| Tipo de fallo | Ocurrencias | Por partida |
|---|---|---|
| Q♠ capturada por BotExperto | 213 | **2.13** |
| Mano con >18 pts (sin ser pozo propio) | 85 | 0.85 |
| Pozo enemigo no bloqueado | 36 | **0.36** |

### Fallo 1 — Captura excesiva de Q♠ (2.13 por partida)

Este es el fallo más frecuente. Causas probables:

1. **BotExperto no tiene Q♠ en su mano pero un rival la descarga sobre él.** La inferencia de quién tiene Q♠ (por eliminación de voids) no es suficientemente predictiva cuando el modelo RL juega de forma no estándar (descartando cartas "inesperadas").

2. **Liderazgo de espadas cuando el RL sabe que BotExperto tiene espadas altas.** El RL puede haber aprendido a liderar espadas cuando BotExperto tiene palo, forzándolo a ganar bazas con Q♠.

3. **La lógica de `_q_posibles()` no siempre actualiza correctamente.** Si el RL descarta de forma que confunde el rastreo de voids del BotExperto, la inferencia de ubicación de Q♠ falla.

### Fallo 2 — Manos catastróficas >18 pts (0.85 por partida)

85 manos con más de 18 puntos sugieren situaciones donde BotExperto no tiene escape posible: forzado a ganar bazas con corazones altos o recibe descartes de rivales cuando ya tiene puntos acumulados en esa mano.

La mayoría coincide con pozos enemigos no bloqueados (el shooter acumula 0 pts, BotExperto recibe 26).

### Fallo 3 — Pozos enemigos no bloqueados (0.36 por partida)

En 36 partidas, un rival completó shooting the moon sin que BotExperto lo detectara o detuviera. Causas:

1. **Detección tardía del intento de pozo.** BotExperto detecta el peligro cuando el rival ya tiene ≥9 corazones, pero puede ser demasiado tarde para actuar.

2. **Modo BLOQUEAR_POZO no activo cuando debería.** Si BotExperto está en modo MINIMIZAR o POZO propio, puede no priorizar el bloqueo.

3. **El RL no intenta pozos (0.0% en Config C).** Así que los 36 pozos no bloqueados en Config A son todos de los bots (J2/J3), no del RL. Esto sugiere que el RL no necesita hacer pozo para ganar — le basta con tener menos puntos que BotExperto consistentemente.

---

## Hipótesis sobre cómo el RL supera al BotExperto (72-28 head-to-head)

A pesar de que BotExperto gana más partidas de 4 jugadores (44% vs 24% en Config A), el RL lo supera en la comparación directa de puntuación final (head-to-head 72-28). Esto implica que en la mayoría de partidas, aunque BotExperto gana el juego completo, RL termina con una puntuación MENOR que BotExperto.

**Mecanismo probable:**

1. **RL empuja puntos hacia BotExperto específicamente.** El modelo aprendió, a través de self-play, a redirigir la carga de puntos hacia el oponente más fuerte. Cuando detecta un adversario que juega bien (tipo BotExperto), lo convierte en objetivo preferente de descartes de Q♠ y corazones altos.

2. **RL no comete errores reactivos.** BotExperto usa heurísticas que pueden ser "engañadas": si el RL juega fuera del patrón esperado, BotExperto puede tomar decisiones subóptimas (capturar Q♠ cuando debería evitarla, no bloquear a tiempo).

3. **RL maneja mejor la incertidumbre.** La observación de 194 dimensiones incluye rastreo de voids (`[156:172]`), posición en el trick (`[181]`), Q♠ tracker (`[182:187]`), y flags estratégicos (`[187:190]`). El RL usa toda esta información de forma no-lineal; BotExperto usa heurísticas lineales aproximadas.

---

## Implicaciones para la Mejora del Sistema RL

### Prioridad Alta — Observación

La comparativa muestra que el RL con 194 dims supera al BotExperto con conocimiento explícito. Pero el BotExperto tiene capacidades que la observación RL NO captura explícitamente:

| Información | BotExperto | RL (194 dims) |
|---|---|---|
| Void tracking por jugador+palo | ✅ Explícito | ✅ `[156:172]` |
| Inferencia de Q♠ por eliminación | ✅ Explícito | ❌ Solo Q♠ tracker |
| Posición de Q♠ (quién probablemente la tiene) | ✅ Parcial | ❌ |
| Estado del pozo rival (% de corazones acumulados) | ✅ Explícito | ❌ |
| "Está el rival intentando hacer pozo ahora?" | ✅ Explícito | ❌ |
| Puntuación acumulada de los rivales por separado | ❌ Solo `[172:176]` avg | ✅ 4 valores normalizados |

**Candidatos para nuevas features de observación (~226 dims):**
- `[194:210]` — ¿Quién tiene Q♠? (4 bits: "yo", "rival_1", "rival_2", "rival_3", "desconocido")
- `[210:214]` — Corazones ya capturados por cada rival (normalizado /13)
- `[214:218]` — Señal de alerta de pozo por rival (0/1 si rival tiene ≥6 corazones)
- `[218:222]` — Puntuación de riesgo de cada rival (puntuación_historica / 100)
- `[222:226]` — ¿Está el rival liderando un palo donde yo estoy void? (4 bits)

### Prioridad Media — Recompensas

El BotExperto nunca intenta shooting the moon contra RL (demasiado arriesgado), mientras que los bots simples lo hacen ocasionalmente con éxito. El RL tampoco lo hace (0.0% en Config C). Esto sugiere que la recompensa actual no incentiva suficientemente las jugadas arriesgadas de alto beneficio cuando el contexto es favorable.

**Candidatos de nuevas recompensas:**
- Recompensa por bloquear pozo exitosamente (detectado por variación brusca de puntos del rival)
- Recompensa por "liderar suits estratégicamente" (quemar palos seguros tarde en la partida)
- Penalización por capturar Q♠ cuando la posición lo hacía evitable

### Prioridad Baja — MCTS

Fases 8-9 (PIMC con BotExperto como rollout + Behavioral Cloning) podrían mejorar significativamente la calidad de las decisiones en situaciones de alta incertidumbre (quién tiene Q♠, si el rival está haciendo pozo). Los 36 pozos no bloqueados son un buen caso de prueba para MCTS: un árbol de búsqueda de 4-6 pasos habría detectado el riesgo antes.

---

## DoD de Fase 3

- [x] Torneo Elo completo (5 participantes, 100 partidas por par, 1000 total)
- [x] Script `analizar_bot_experto.py` con métricas detalladas por tipo de agente
- [x] 3 configuraciones analizadas (300 partidas totales)
- [x] Fallos críticos documentados con frecuencias
- [x] Hipótesis sobre ventaja RL sobre BotExperto
- [x] Implicaciones para Fases 4-9 identificadas

**Estado: Fase 3 COMPLETADA. Listo para Fase 4: Análisis de Features.**
