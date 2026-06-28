# Análisis de Errores — v4 snapshot_0008100000

**Fecha**: 2026-06-24 13:01
**Modelo**: `models/v4/snapshots/snapshot_0008100000`
**Manos analizadas**: 80
**Tiempo de análisis**: 127.3s
**Baza mínima para PIMC exacto**: 10

## 1. Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Puntuación media | 6.2 |
| Manos con 0 pts | 42/80 (52%) |
| Manos donde capturó Q♠ | 18/80 (22%) |
| **Total errores detectados** | **29** |
| **Coste total acumulado** | **50.4 pts** |
| Coste medio por mano | 0.6 pts |
| Coste medio por error | 1.74 pts |

## 2. Errores por Etapa de la Mano

| Baza | Errores | Coste Total | Coste Medio | ¿Evitable? |
|------|---------|-------------|-------------|------------|
| 10 | 16 | 31.8 | 1.98 | ✅ Sí (PIMC exacto) |
| 11 | 10 | 13.9 | 1.39 | ✅ Sí (PIMC exacto) |
| 12 | 3 | 4.8 | 1.59 | ✅ Sí (PIMC exacto) |

## 3. Errores por Situación de Juego

| Situación | Errores | Coste Total | Coste Medio |
|-----------|---------|-------------|-------------|
| liderar | 4 | 12.6 | 3.14 |
| seguir | 7 | 12.4 | 1.77 |
| descartar | 18 | 25.4 | 1.41 |

## 4. Top 10 Errores Más Costosos

| # | Baza | Situación | Modelo eligió | Óptimo PIMC | Coste | ΔScore |
|---|------|-----------|---------------|-------------|-------|--------|
| 1 | 10 | liderar | 3♦ | 7♠ | 9.0 | 9.0 |
| 2 | 10 | descartar | 3♠ | A♠ | 5.3 | 5.3 |
| 3 | 11 | descartar | 2♠ | A♠ | 4.2 | 4.2 |
| 4 | 10 | seguir | 3♥ | 10♥ | 4.0 | 4.0 |
| 5 | 12 | descartar | 3♦ | 9♥ | 3.7 | 3.7 |
| 6 | 10 | seguir | 3♥ | 10♥ | 3.4 | 3.4 |
| 7 | 11 | liderar | 10♥ | 6♥ | 2.7 | 2.9 |
| 8 | 10 | seguir | 9♣ | 3♣ | 2.7 | 2.7 |
| 9 | 10 | descartar | A♥ | 3♣ | 2.5 | 9.1 |
| 10 | 11 | descartar | 3♥ | 8♥ | 1.7 | 1.7 |

## 5. Distribución del Coste de Errores

| Percentil | Coste (pts) |
|-----------|-------------|
| P50 | 1.04 |
| P75 | 2.66 |
| P90 | 4.03 |
| P95 | 4.83 |
| P99 | 7.96 |
| P100 | 9.00 |

## 6. Análisis Cualitativo

### Errores al Liderar (4 casos)

- Coste medio: 3.14 pts
- Suele ocurrir cuando el modelo **no identifica la carta óptima para iniciar la baza**.
- El modelo puede estar jugando cartas demasiado altas o demasiado bajas para la fase de la mano.

### Errores al Seguir Palo (7 casos)

- Coste medio: 1.77 pts
- El modelo **no ajusta bien la altura de la carta** al seguir el palo.
- Posiblemente juega cartas muy altas cuando debería jugar bajas (o viceversa).

### Errores al Descartar (18 casos)

- Coste medio: 1.41 pts
- El modelo **elige mal qué palo descartar** o **tira una carta incorrecta**.
- Posiblemente no está considerando correctamente los vacíos de los oponentes.

## 7. Implicaciones para v5

- **Coste total por mano**: 0.6 pts de error evitable.
- Si v5 logra reducir este coste a la mitad, la puntuación media mejoraría ~0.3 pts.
- Las bazas críticas son: 10, 11, 12.
- v5 con self-play puro debería mostrar **menor coste en bazas tardías** (10+) donde PIMC tiene ground truth.

## 8. Manos con Más Errores

| Seed | Puntos | Q♠ | Errores | Coste Total |
|------|--------|-----|---------|-------------|
| 58 | 0 | Baza 7 | 2 | 1.0 |
| 69 | 0 | No | 2 | 2.1 |
| 71 | 11 | Baza 6 | 2 | 0.9 |
| 91 | 4 | No | 2 | 6.7 |
| 114 | 16 | No | 2 | 9.4 |
| 42 | 0 | No | 1 | 1.0 |
| 44 | 0 | No | 1 | 0.4 |
| 45 | 26 | No | 1 | 0.2 |
| 55 | 4 | No | 1 | 1.7 |
| 61 | 13 | Baza 9 | 1 | 0.0 |

---

## 9. Patrones de Error Detectados (Análisis Detallado)

### 9.1 Errores de «No Vaciar Picas con Q♠ Pendiente» (Coste: 9.0 pts, baza 10)

El error más costoso: el modelo lideró con **3♦** en baza 10 cuando aún tenía **7♠** en mano
y la Q♠ seguía sin aparecer. PIMC identificó que liderar con 7♠ era la jugada óptima (+9.0 pts mejor).

**Diagnóstico**: El modelo no reconoce que en baza 10 es crítico forzar la salida de Q♠ vaciando
el palo de picas. Retener una pica baja con Q♠ aún en juego es un error grave.

**Tipo**: Error de **planificación de fin de mano** — el modelo prioriza mal el orden de descarte.

### 9.2 Errores de «Jugar Pica Baja en Lugar de Alta» (Coste: 5.3 + 4.2 pts)

El modelo descartó **3♠** en lugar de **A♠** (baza 10, coste 5.3 pts), y **2♠** en lugar de
**A♠** (baza 11, coste 4.2 pts).

**Diagnóstico**: El modelo retiene cartas altas de picas por miedo a capturar Q♠, pero cuando
Q♠ ya salió (o el riesgo es bajo en baza avanzada), jugar el A♠ es más seguro que retener
picas bajas que pueden forzar capturas indeseadas.

**Tipo**: Error de **gestión de riesgo Q♠** — sobreprotección contra Q♠ cuando ya no es necesario.

### 9.3 Errores de «Jugar Corazón Alto en Lugar de Bajo» (Coste: 4.0 + 3.4 pts)

El modelo jugó **3♥** en lugar de **10♥** al seguir el palo de corazones (2 ocasiones).

**Diagnóstico**: El modelo no ajusta la altura de la carta al seguir corazones.
Jugar un corazón bajo cuando tienes uno más alto disponible aumenta el riesgo de
ganar la baza y capturar todos los corazones acumulados.

**Tipo**: Error de **ajuste de altura al seguir palo**.

### 9.4 Errores de «Descartar Corazón en Vez de Trébol o Diamante» (Coste: 2.5 + 1.2 + 0.7 pts)

El modelo descartó **A♥** en lugar de **3♣** (coste 2.5 pts) y otros corazones en vez de
cartas de palos sin puntos.

**Diagnóstico**: El modelo tira corazones demasiado pronto en bazas 10-12 cuando debería
retenerlos para evitar ganar bazas futuras con corazones altos.

**Tipo**: Error de **prioridad de descarte** — no evalúa correctamente el riesgo relativo
de cada palo.

## 10. Comparación con Evaluación Estándar

| Métrica | Evaluación estándar (200 partidas, EvalCallback) | PIMC exacto (80 manos) |
|---------|---------------------------------------------------|------------------------|
| Score medio | 5.66 (histórico) — 5.69 (último snapshot) | 6.2 |
| WR ≤8 | 76% | ~52% manos con ≤8 |
| Top1 | 22% | 52% manos con 0 pts (contra bots evasivos) |

**Nota**: La evaluación estándar usa BotExperto como oponentes (más difíciles),
mientras que el análisis PIMC usó bots evasivos (más fáciles). Esto explica el
score más bajo (6.2 vs 5.66) — los oponentes más débiles deberían resultar en
mejor score para el modelo.

## 11. Línea Base para Comparación con v5

Al finalizar el entrenamiento de v5, se debe ejecutar el mismo análisis:

```bash
python scripts/analizar_errores_v4.py --modelo models/v5/snapshots/snapshot_XXXXXXXXXX --manos 80 --output docs/analisis_v5_errores.md
```

**Métricas clave para comparar v4 vs v5**:

| Métrica | v4 actual | v5 objetivo | Mejora esperada |
|---------|-----------|-------------|-----------------|
| Total errores / 80 manos | 29 | <20 | -31% |
| Coste total acumulado | 50.4 pts | <30 pts | -40% |
| Coste medio por error | 1.74 pts | <1.2 pts | -31% |
| Errores al descartar | 18 | <12 | -33% |
| Error máximo individual | 9.0 pts | <5.0 pts | -44% |

## 12. Conclusiones

1. **v4 comete 29 errores evitables en 80 manos** según PIMC con ground truth (bazas 10-12).
2. **62% de los errores son al descartar** (18/29) — el modelo no prioriza correctamente
   qué palo vaciar primero.
3. **El error más grave (9.0 pts)** es no vaciar picas cuando Q♠ está pendiente en baza 10.
4. **Coste total de 50.4 pts en 80 manos** — si v5 elimina solo los errores de descarte,
   el score medio mejoraría ~0.3 pts.
5. **El modelo muestra sobreprotección contra Q♠** en bazas tardías, reteniendo picas
   bajas que son más peligrosas que las altas.
6. **v5 con self-play puro 4-way debería reducir estos errores** porque el modelo
   aprende simultáneamente las 4 perspectivas (tirar Q♠, cazar Q♠, defender, atacar).
