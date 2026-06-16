# Fase 4 — Análisis de Features para el Espacio de Observación

**Fecha:** 2026-06-15  
**Input:** Resultados de Fase 3 (observaciones_bot_experto.md), espacio actual (194 dims, observacion.py)  
**Output:** Espacio definitivo de observación para v9 (~220 dims)

---

## 1. Estado Actual: 194 dimensiones

```
[0:52]     Mano del agente (one-hot, 52 bits)
[52:104]   Mesa actual / baza en curso (one-hot, 52 bits)
[104:156]  Cementerio / bazas pasadas (one-hot, 52 bits)
[156:172]  Vacíos conocidos: 4 jugadores × 4 palos (16 bits)
[172:176]  Puntajes históricos globales (4 floats, /100)
[176:180]  Puntos acumulados en mano actual (4 floats, /26)
[180]      Corazones rotos (bool)
[181]      Posición en la baza actual (0.0/0.33/0.66/1.0)
[182:187]  Q♠ tracker: one-hot 5 estados (unknown|me|rival1|rival2|rival3)
[187]      pozo_viable (bool)
[188]      debo_arriesgar (bool)
[189]      puedo_alimentar (bool)
[190:194]  all_void_X: ¿todos los rivales son void en palo X? (4 bools)
```

### Limitaciones identificadas en Fase 3

| Limitación | Impacto en juego | Evidencia |
|---|---|---|
| Q♠ tracker no distingue probabilidad | Modelo no sabe QUIÉN *puede* tener Q♠, solo quién la tiene con certeza | 2.13 capturas de Q♠ por partida en BotExperto; el RL tampoco las evita mejor |
| Sin información de corazones capturados ESTA MANO | Modelo no detecta intentos de shooting the moon hasta el final | 36 pozos no bloqueados en 100 partidas |
| Sin indicador de fase de la partida | Estrategia no varía entre baza 1 (todo abierto) y baza 12 (crítico) | BotExperto usa número de baza explícitamente |
| Sin conteo de cartas restantes por palo | Modelo no sabe si liderar un palo es "seguro" (quedan pocas) | BotExperto quema palos en baza ≥10 |
| Tracker Q♠ no diferencia "ya capturada" de "yo la tengo" | Ambigüedad: si capturé Q♠ en modo pozo, el tracker dice "yo tengo Q♠" igual | Código en observacion.py: sin distinción |

---

## 2. Matriz de Decisión: Features Propuestos

Para cada feature candidato, se evalúan 3 criterios:
1. **¿Inferible?** — ¿Puede el modelo aprender esto de features existentes?
2. **¿Costoso?** — Si es inferible, ¿el aprendizaje consume capacidad innecesaria?
3. **¿Usa BotExperto?** — ¿Sería decisivo para el BotExperto tenerlo?

### 2.1 Features del plan original

| Feature | Inferible de features existentes | Costoso de inferir | Usa BotExperto | Decisión |
|---|---|---|---|---|
| `baza_numero / 13` | Sí, del cementerio (tamaño ÷ 4 ÷ 13 aprox.) | Sí — requiere conteo + división | Sí (activa estrategia tardía) | ✅ INCLUIR |
| `cartas_restantes_por_palo` (4) | Sí, de hand[0:52] + cementerio[104:156] | Sí — requiere suma sobre 52 índices por palo | Sí (saber si liderar palo es seguro) | ✅ INCLUIR |
| `cartas_altas_restantes_por_palo` (4) | Sí, subconjunto del anterior | Sí — requiere filtrar por valor ≥ J | Sí (evaluar si puedo ganar bazas) | ✅ INCLUIR |
| `probabilidad_Q♠_por_jugador` (4) | Parcialmente (voids[156:172] dan información) | Sí — combinación no trivial | Sí (motor de decisión de BotExperto) | ✅ INCLUIR |
| `modo_juego_one_hot` (4) | Sí, de pozo_viable + debo_arriesgar + puedo_alimentar | Moderado — ya hay 3 flags parciales | No (BotExperto lo computa internamente) | ❌ SKIP (redundante con flags existentes) |
| `diferencia_puntaje_con_lider` | Sí, de puntajes[172:176] | Trivial | Sí (ajustar estrategia) | ❌ SKIP (derivable en 1 paso) |
| `diferencia_puntaje_con_ultimo` | Sí | Trivial | Parcialmente | ❌ SKIP |
| `jugadores_cerca_de_100` | Sí, de puntajes[172:176] | Bajo — pero útil como señal explícita | Sí (urgencia de fin de partida) | ✅ INCLUIR (1 dim, cuenta de j con score≥85) |
| `direccion_del_pase` | N/A | N/A | N/A | ❌ SKIP (el motor no implementa passing) |
| `%_bazas_restantes_con_puntos` | Sí, del cementerio | Moderado | Parcialmente | ❌ SKIP |

### 2.2 Features nuevos derivados de análisis Fase 3

| Feature | Justificación | Inferible | Decisión |
|---|---|---|---|
| `corazones_capturados_esta_mano_por_jugador` (4) | Detectar intento de moon. Key para BLOQUEAR. El modelo ve puntos_mano_actual[176:180] pero no distingue corazones de Q♠ | Parcialmente (puntos_mano_actual es suma de corazones + 13*tiene_Q♠) | ✅ INCLUIR |
| `alerta_pozo_por_jugador` (4) | Señal directa: "rival tiene ≥6 corazones esta mano → peligro de pozo" | Sí, si tiene corazones_capturados_esta_mano | ✅ INCLUIR (4 bools) |
| `Q♠_ya_capturada` (1) | Distinción entre "tracker activo" y "Q♠ ya fue tomada". Actualmente ambiguo en [182:187] | Sí, del cementerio [104:156] verificando Q♠ bit | ✅ INCLUIR (hace explícito el fin del peligro) |
| `soy_lider_en_puntaje` (1) | Indica que tengo el puntaje más bajo (situación más vulnerable) | Sí, de puntajes[172:176] | ✅ INCLUIR (crítico para estrategia "no dejar ganar al lider") |
| `mano_final_posible` (1) | Esta mano puede terminar el juego (algún jugador tiene score ≥74) | Sí, de puntajes[172:176] | ✅ INCLUIR (activa estrategia terminal) |
| `rivales_void_en_palo_de_salida` (1) | ¿Todos los rivales son void en el palo liderado? | Sí, de voids[156:172] + palo_de_salida | ❌ SKIP (all_void_X[190:194] ya lo cubre) |

---

## 3. Espacio de Observación Definitivo: 220 dimensiones

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOQUE 1 — Estado del juego (sin cambios)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[0:52]     Mano del agente (one-hot)
[52:104]   Mesa actual (one-hot)
[104:156]  Cementerio (one-hot)
[156:172]  Vacíos conocidos (4 × 4)
[172:176]  Puntajes históricos (/100)
[176:180]  Puntos mano actual (/26)
[180]      Corazones rotos
[181]      Posición en baza (0.0/0.33/0.66/1.0)
[182:187]  Q♠ tracker — 5 estados (sin cambio)
[187]      pozo_viable
[188]      debo_arriesgar
[189]      puedo_alimentar
[190:194]  all_void_X (4 bools)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOQUE 2 — Features de fase y progresión  [194:199]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[194]      baza_numero / 13.0                         — fase del juego [0,1]
[195]      jugadores_cerca_de_100 / 3.0               — urgencia terminal [0,1]
[196]      Q♠_ya_capturada (bool)                     — peligro de Q♠ terminó
[197]      soy_lider_en_puntaje (bool)                — tengo puntaje más bajo
[198]      mano_terminal_posible (bool)               — alguien tiene score ≥74

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOQUE 3 — Conteo de cartas restantes  [199:207]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[199:203]  cartas_restantes_por_palo / 13.0           — (♣, ♦, ♠, ♥), relativas
[203:207]  cartas_altas_restantes_por_palo / 4.0      — (J/Q/K/A por palo), relativas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOQUE 4 — Probabilidad de Q♠ por jugador  [207:211]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[207:211]  prob_Q♠_relativa (4 floats [0,1])          — jugadores relativos al agente
           Cálculo: P(j tiene Q♠) ∝ 1 si no void en picas, 0 si void o ya conocido
           Distinto del tracker [182:187] que solo registra certeza.
           Cuando Q♠ está capturada → todos los valores a 0.0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOQUE 5 — Detección de moon / bloqueo  [211:220]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[211:215]  corazones_capturados_esta_mano / 13.0      — (4 jugadores relativos)
           Nota: derivar de bazas_ganadas de cada jugador, filtrado por es_corazon
[215:219]  alerta_pozo_por_jugador (4 bools)          — 1 si jugador ≥6 corazones
[219]      RESERVADO                                  — para futuras features

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: 220 dimensiones
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.1 Mapa de índices definitivo (v9)

| Índice | Contenido | Tipo | Rango |
|---|---|---|---|
| `[0:52]` | Mano del agente | one-hot | {0,1} |
| `[52:104]` | Mesa actual | one-hot | {0,1} |
| `[104:156]` | Cementerio | one-hot | {0,1} |
| `[156:172]` | Vacíos conocidos 4×4 | one-hot | {0,1} |
| `[172:176]` | Puntajes históricos | float | [0,1] |
| `[176:180]` | Puntos mano actual | float | [0,1] |
| `[180]` | Corazones rotos | bool | {0,1} |
| `[181]` | Posición en baza | enum | {0.0,0.33,0.66,1.0} |
| `[182:187]` | Q♠ tracker (5 estados) | one-hot | {0,1} |
| `[187]` | pozo_viable | bool | {0,1} |
| `[188]` | debo_arriesgar | bool | {0,1} |
| `[189]` | puedo_alimentar | bool | {0,1} |
| `[190:194]` | all_void_X (4 palos) | bool | {0,1} |
| `[194]` | baza_numero / 13.0 | float | [0.077, 1.0] |
| `[195]` | jugadores_cerca_100 / 3.0 | float | {0,0.33,0.67,1.0} |
| `[196]` | Q♠ ya capturada | bool | {0,1} |
| `[197]` | soy lider en puntaje | bool | {0,1} |
| `[198]` | mano terminal posible | bool | {0,1} |
| `[199:203]` | cartas restantes por palo / 13 | float | [0,1] |
| `[203:207]` | cartas altas restantes por palo / 4 | float | [0,1] |
| `[207:211]` | prob Q♠ por jugador relativo | float | [0,1] |
| `[211:215]` | corazones capturados esta mano / 13 | float | [0,1] |
| `[215:219]` | alerta pozo por jugador (≥6 cor.) | bool | {0,1} |
| `[219]` | RESERVADO | — | 0.0 |

---

## 4. Decisiones de Implementación

### 4.1 Compatibilidad hacia atrás

Los snapshots v7 y v8 (194 dims) requieren `ObservacionBuilder(dim=194)`.  
El nuevo `ObservacionBuilder(dim=220)` hereda todo el código de 194 y añade los 26 nuevos features en `_construir_bloque_v7()`.

El wrapper `_Modelo190Wrapper` (para v5, 190 dims) se mantiene sin cambios.

### 4.2 Cálculo de `prob_Q♠_relativa` [207:211]

```python
def _calcular_prob_q_picas(self, motor, vacios, agente_idx, q_capturada):
    """Distribuye la probabilidad de Q♠ según voids conocidos."""
    if q_capturada:
        return [0.0, 0.0, 0.0, 0.0]
    
    # Yo tengo Q♠ (certeza)
    mi_mano = motor.jugadores[agente_idx].mano
    if any(c.es_dama_de_picas for c in mi_mano):
        return [1.0, 0.0, 0.0, 0.0]  # relativo: yo=0
    
    # Q♠ está en mesa (certeza temporal)
    for jug_idx, carta in motor.mesa:
        if carta.es_dama_de_picas:
            rel = (jug_idx - agente_idx) % 4
            return [1.0 if r == rel else 0.0 for r in range(4)]  # no, la ya jugó
    
    # Distribuir uniformemente entre quienes NO son void en picas
    PICA = 2
    candidatos = [
        r for r in range(1, 4)  # rivales relativos (yo=0 no la tengo)
        if PICA not in vacios[(agente_idx + r) % 4]
    ]
    if not candidatos:
        return [0.0, 0.0, 0.0, 0.0]  # imposible (bug en void tracking)
    
    prob = 1.0 / len(candidatos)
    result = [0.0, 0.0, 0.0, 0.0]
    for r in candidatos:
        result[r] = prob
    return result
```

### 4.3 `construir_desde_motor()` para oponentes en self-play

El método `construir_desde_motor()` (usado por snapshots de versiones anteriores como oponentes) NO añade los nuevos features. Queda todo en 0.0 para los dims [194:220].

Esto es correcto: los snapshots v7/v8 esperan 194 dims y usan su propio `ObservacionBuilder(dim=194)`. Los nuevos snapshots v9 usarán 220 dims.

### 4.4 Normalización

Los nuevos features floats [199:219] están naturalmente en [0,1], por lo que VecNormalize no tendrá que hacer correcciones grandes. Los bools {0,1} se tratan correctamente por VecNormalize (varianza ≈ 0.25).

---

## 5. Features Descartados con Justificación

| Feature descartado | Razón |
|---|---|
| `modo_juego_one_hot` | Redundante: pozo_viable[187] + debo_arriesgar[188] + puedo_alimentar[189] ya son los 3 flags del modo. Añadir un 4° modo (BLOQUEAR) podría ser útil pero el modelo puede inferirlo de alerta_pozo[215:219]. |
| `diferencia con líder/último` | Derivable de puntajes[172:176] en un paso. La red MLP puede aprender esto en la primera capa. |
| `direccion_del_pase` | El motor no implementa la fase de pase de cartas. |
| `%_bazas_con_puntos` | Derivable del cementerio [104:156] contando corazones y Q♠ no vistos. `cartas_altas_restantes[203:207]` cubre esta necesidad. |
| `es_void_en_palo_de_salida` | Trivialmente inferible: si el agente tiene al menos una carta del palo en [0:52] y el palo liderado está en mesa[52:104]. |
| `cartas_altas_propias_por_palo` | Derivable de [0:52] filtrando J/Q/K/A. El modelo puede aprenderlo. |

---

## 6. Estimación de Impacto

| Feature nuevo | Caso donde ayuda | Frecuencia en partida |
|---|---|---|
| `baza_numero[194]` | Activar estrategia tardía (quemar palos, bloquear) | Cada baza |
| `cartas_restantes[199:203]` | Decidir si liderar un palo es "seguro" (pocas cartas) | Cada liderazgo |
| `prob_Q♠[207:211]` | Descargar Q♠ al jugador correcto; saber cuándo buscarla | Cada baza con espadas |
| `corazones_esta_mano[211:215]` | Detectar riesgo de moon en rivales | Desde baza 6-7 en mano |
| `alerta_pozo[215:219]` | Señal directa para cambiar a modo BLOQUEAR | ~2-3 veces por partida |
| `Q♠_capturada[196]` | Relajar estrategia de espadas después de capturarse | Después de baza con Q♠ |
| `mano_terminal[198]` | Estrategia agresiva si el rival líder puede ganar con 1 mano buena | ~30% de manos |

---

## 7. DoD de Fase 4

- [x] Matriz de decisión completada para todos los features propuestos
- [x] Espacio definitivo de 220 dims documentado con índices exactos
- [x] Features descartados con justificación documentados
- [x] Pseudocódigo de `_calcular_prob_q_picas()` preparado
- [x] Decisiones de compatibilidad (194↔220) documentadas

**Estado: Fase 4 COMPLETADA. Listo para Fase 5: Implementación TDD del nuevo espacio de observación.**
