# Análisis de Errores — v4 snapshot_0011800000

**Fecha**: 2026-06-24 19:00
**Modelo**: `models/v5/snapshots/snapshot_0011800000`
**Manos analizadas**: 50
**Tiempo de análisis**: 87.1s
**Baza mínima para PIMC exacto**: 10

## 1. Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Puntuación media | 4.3 |
| Manos con 0 pts | 32/50 (64%) |
| Manos donde capturó Q♠ | 13/50 (26%) |
| **Total errores detectados** | **12** |
| **Coste total acumulado** | **21.4 pts** |
| Coste medio por mano | 0.4 pts |
| Coste medio por error | 1.78 pts |

## 2. Errores por Etapa de la Mano

| Baza | Errores | Coste Total | Coste Medio | ¿Evitable? |
|------|---------|-------------|-------------|------------|
| 10 | 9 | 12.9 | 1.43 | ✅ Sí (PIMC exacto) |
| 11 | 3 | 8.5 | 2.83 | ✅ Sí (PIMC exacto) |

## 3. Errores por Situación de Juego

| Situación | Errores | Coste Total | Coste Medio |
|-----------|---------|-------------|-------------|
| liderar | 2 | 2.4 | 1.18 |
| seguir | 5 | 9.7 | 1.94 |
| descartar | 5 | 9.3 | 1.86 |

## 4. Top 10 Errores Más Costosos

| # | Baza | Situación | Modelo eligió | Óptimo PIMC | Coste | ΔScore |
|---|------|-----------|---------------|-------------|-------|--------|
| 1 | 11 | descartar | 3♦ | 4♦ | 4.6 | 4.6 |
| 2 | 10 | seguir | 3♥ | 10♥ | 4.0 | 4.0 |
| 3 | 10 | seguir | 3♥ | 10♥ | 3.4 | 3.4 |
| 4 | 10 | descartar | A♥ | 3♣ | 2.5 | 9.1 |
| 5 | 10 | liderar | 6♥ | 4♥ | 2.3 | 5.2 |
| 6 | 11 | seguir | 3♥ | 10♥ | 2.1 | 2.1 |
| 7 | 11 | descartar | 3♥ | 8♥ | 1.7 | 1.7 |
| 8 | 10 | descartar | 10♥ | 3♣ | 0.4 | 5.7 |
| 9 | 10 | seguir | 8♠ | 4♠ | 0.1 | 0.1 |
| 10 | 10 | descartar | 7♦ | A♣ | 0.0 | 0.4 |

## 5. Distribución del Coste de Errores

| Percentil | Coste (pts) |
|-----------|-------------|
| P50 | 1.93 |
| P75 | 2.74 |
| P90 | 3.94 |
| P95 | 4.29 |
| P99 | 4.57 |
| P100 | 4.64 |

## 6. Análisis Cualitativo

### Errores al Liderar (2 casos)

- Coste medio: 1.18 pts
- Suele ocurrir cuando el modelo **no identifica la carta óptima para iniciar la baza**.
- El modelo puede estar jugando cartas demasiado altas o demasiado bajas para la fase de la mano.

### Errores al Seguir Palo (5 casos)

- Coste medio: 1.94 pts
- El modelo **no ajusta bien la altura de la carta** al seguir el palo.
- Posiblemente juega cartas muy altas cuando debería jugar bajas (o viceversa).

### Errores al Descartar (5 casos)

- Coste medio: 1.86 pts
- El modelo **elige mal qué palo descartar** o **tira una carta incorrecta**.
- Posiblemente no está considerando correctamente los vacíos de los oponentes.

## 7. Implicaciones para v5

- **Coste total por mano**: 0.4 pts de error evitable.
- Si v5 logra reducir este coste a la mitad, la puntuación media mejoraría ~0.2 pts.
- Las bazas críticas son: 10, 11.
- v5 con self-play puro debería mostrar **menor coste en bazas tardías** (10+) donde PIMC tiene ground truth.

## 8. Manos con Más Errores

| Seed | Puntos | Q♠ | Errores | Coste Total |
|------|--------|-----|---------|-------------|
| 43 | 0 | Baza 6 | 1 | 0.1 |
| 51 | 13 | No | 1 | 0.0 |
| 53 | 26 | No | 1 | 4.6 |
| 54 | 0 | No | 1 | 0.4 |
| 55 | 4 | No | 1 | 1.7 |
| 61 | 13 | Baza 9 | 1 | 0.0 |
| 68 | 0 | Baza 10 | 1 | 2.5 |
| 72 | 0 | No | 1 | 2.1 |
| 78 | 4 | No | 1 | 3.4 |
| 82 | 14 | No | 1 | 2.3 |