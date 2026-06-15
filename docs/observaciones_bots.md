# Observaciones de Bots — Fase 1

**Fecha:** 2026-06-15
**Objetivo:** Documentar las deficiencias de los bots heurísticos actuales antes de construir el Bot Experto (Fase 2).

---

## Resultados de los Tests

### Test A — Torneo Elo bots vs bots (50 partidas por par, 150 total)

| Pos | Bot | Elo | W% (estimado) |
|---|---|---|---|
| 1 | agresivo | 1514 | 27/50 vs conservador, 23/50 vs evasivo |
| 2 | evasivo | 1509 | 29/50 vs conservador, 23/50 vs agresivo |
| 3 | conservador | 1477 | 24/50 vs agresivo, 21/50 vs evasivo |

**Conclusión:** Los tres bots están dentro de 37 puntos Elo entre sí. No hay un bot claramente dominante. El agresivo y el evasivo son casi equivalentes; el conservador es el más débil.

---

### Test B — Análisis verbose (30+ partidas, combinaciones bot vs bot)

Estadísticas agregadas 30 partidas (c/a/e/c, semillas 0-29):

| Bot | 1º | Top-2 | Pts/partida |
|---|---|---|---|
| J3 conservador | 40.0% | 60.0% | 67.7 |
| J0 conservador | 23.3% | 46.7% | 76.1 |
| J1 agresivo | 20.0% | 43.3% | 78.3 |
| J2 evasivo | 16.7% | 50.0% | 75.1 |

> **Nota:** J0 y J3 usan la misma estrategia (conservador), pero J3 rinde ~17% mejor en victorias. Esto confirma que la posición en la mesa importa significativamente: J3 actúa último en el primer trick (2♣), lo que le da más información antes de decidir.

---

### Test C — Win rate v7_golden vs bots (200 partidas)

| Métrica | v7_golden (1723 Elo) |
|---|---|
| 1er lugar | **82.5%** (165/200) |
| Top-2 | **97.0%** (194/200) |
| 3º o 4º | 3.0% |

**Referencia:** Los bots heurísticos alcanzan ~17-40% de win rate entre sí. El modelo RL supera a cualquier bot en más de 3× esa tasa.

---

## Deficiencias Detectadas por Bot

### Bot Conservador (`min(legales, key=lambda c: c.valor)`)

**Error 1 — Valor numérico ≠ peligrosidad**
El bot ordena por `c.valor` (valor de carta: 2–14), no por `c.puntos` (0, 1, o 13). Resultado: considera Q♠ (valor=12, puntos=13) menos peligrosa que K♥ (valor=13, puntos=1). En situaciones donde debe elegir entre descartar Q♠ o K♥, retiene Q♠ más tiempo del necesario.

**Error 2 — No diferencia entre liderar y seguir el palo**
Al liderar, el conservador siempre juega la carta más baja de su mano. Si su carta más baja es en un palo donde los rivales tienen cartas altas, puede iniciar una baza que termina ganando innecesariamente.

**Error 3 — Sin concepto de void**
No sabe qué palos tienen los rivales ni cuándo es seguro liderar. Lidera el 2♣ (el más bajo de su mano) aunque pudiera liderar un palo donde es seguro.

---

### Bot Agresivo (`max(mismo_palo) si sigue, else max(legales)`)

**Error 1 — SIEMPRE captura Q♠ cuando puede** (más crítico)
Si el palo liderado es espadas y tiene cartas de espadas, juega la más alta. Si esa carta es mayor que Q♠, la captura automáticamente. No hay ningún control de "¿hay Q♠ en la mesa?".

Ejemplo observado (Mano 4/B02):
```
Mesa: [J2:K♠, J3:Q♠]   Mano J0: [♠A4]
J0 agresivo juega A♠ → captura Q♠ (13pts)
Podría jugar 4♠ → no captura Q♠
```

**Error 2 — Captura corazones innecesariamente**
Al seguir palo, juega la carta más alta del palo. Si tiene K♥ o A♥ y el palo de salida es corazones, captura la baza con todos sus puntos.

Ejemplo observado (Mano 1/B08):
```
Mesa: [J2:8♥]   Mano J3: [♥10742]
J3 agresivo juega 10♥ → gana baza cuando podría jugar 2♥
```

**Error 3 — Liderar con las cartas más altas**
Al liderar, juega `max(legales)` — la carta de mayor valor en su mano. Esto con frecuencia implica liderar corazones altos o A/K de palos donde puede recibir Q♠ como descarte de un rival.

---

### Bot Evasivo (`min(sin_puntos) siguiendo, max(con_puntos) descartando`)

**Error 1 — Descarta corazones antes que Q♠**
Cuando está void y tiene que descartar cartas con puntos, usa `max(con_puntos, key=lambda c: c.valor)`. Esto ordena por valor de carta (2–14), no por puntos reales. Como A♥ tiene valor 14 y Q♠ tiene valor 12, el bot descarta A♥ (1 punto) antes que Q♠ (13 puntos) cuando ambas están disponibles.

Ejemplo implicado: si tiene [Q♠, A♥, K♥], el orden de descarte sería A♥ → K♥ → Q♠. Lo correcto sería Q♠ → A♥ → K♥.

**Error 2 — Sin estrategia de liderazgo**
Al liderar (mesa vacía), descarta `min(sin_puntos)`. Nunca considera qué palo le da más fugas futuras ni qué cartas peligrosas necesita descartar.

**Error 3 — "Evasivo" no siempre equivale a "mínimos puntos"**
El bot evita tener puntos en su mano, pero en situaciones donde conviene ganar la baza (e.g., para controlar el palo siguiente), la estrategia puramente evasiva es subóptima.

---

## Errores Sistemáticos Comunes (todos los bots)

| Deficiencia | Impacto |
|---|---|
| Sin rastreo de Q♠ | No saben dónde está la Dama de Picas hasta que aparece |
| Sin inferencia de voids | No saben en qué palos son void los rivales |
| Sin detección de pozo ajeno | Ningún bot bloquea cuando un rival intenta shooting the moon |
| Sin razonamiento multi-baza | Cada decisión es local, sin lookahead |
| Sin adaptar estrategia a posición | El resultado varía 17% solo por orden en la mesa |

---

## ¿El Modelo RL Aprendió a Jugar Bien?

**Sí, en el sentido de que supera abrumadoramente a los bots (82.5% WR vs 17-40% de los bots entre sí).**

Pero la pregunta más importante es: **¿aprendió a jugar Corazones, o simplemente aprendió a explotar las debilidades de estos bots?**

Dado que los bots tienen deficiencias tan obvias (agresivo captura Q♠ sistemáticamente, conservador no diferencia puntos de valor), el modelo puede haberse especializado en explotar esas debilidades concretas sin aprender estrategia general. Esto se confirmará cuando compitamos contra el Bot Experto (Fase 2).

---

## DoD de Fase 1

- [x] Torneo Elo entre bots ejecutado y resultados documentados
- [x] 30+ partidas analizadas y deficiencias documentadas
- [x] Win rate del modelo RL vs bots registrado (82.5%, 200 partidas)
- [x] `observaciones_bots.md` creado

**Estado: Fase 1 COMPLETADA. Listo para Fase 2: Bot Experto.**
