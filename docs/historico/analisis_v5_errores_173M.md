# Análisis de Errores — v4 snapshot_0017300000

**Fecha**: 2026-06-24 19:01
**Modelo**: `models/v5/snapshots/snapshot_0017300000`
**Manos analizadas**: 50
**Tiempo de análisis**: 77.0s
**Baza mínima para PIMC exacto**: 10

## 1. Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Puntuación media | 4.5 |
| Manos con 0 pts | 30/50 (60%) |
| Manos donde capturó Q♠ | 13/50 (26%) |
| **Total errores detectados** | **18** |
| **Coste total acumulado** | **65.9 pts** |
| Coste medio por mano | 1.3 pts |
| Coste medio por error | 3.66 pts |

## 2. Errores por Etapa de la Mano

| Baza | Errores | Coste Total | Coste Medio | ¿Evitable? |
|------|---------|-------------|-------------|------------|
| 10 | 12 | 54.1 | 4.51 | ✅ Sí (PIMC exacto) |
| 11 | 4 | 8.0 | 2.01 | ✅ Sí (PIMC exacto) |
| 12 | 2 | 3.8 | 1.88 | ✅ Sí (PIMC exacto) |

## 3. Errores por Situación de Juego

| Situación | Errores | Coste Total | Coste Medio |
|-----------|---------|-------------|-------------|
| liderar | 2 | 7.3 | 3.64 |
| seguir | 8 | 49.5 | 6.19 |
| descartar | 8 | 9.1 | 1.13 |

## 4. Top 10 Errores Más Costosos

| # | Baza | Situación | Modelo eligió | Óptimo PIMC | Coste | ΔScore |
|---|------|-----------|---------------|-------------|-------|--------|
| 1 | 10 | seguir | 10♠ | 6♠ | 17.3 | 17.3 |
| 2 | 10 | seguir | 11♥ | 5♥ | 12.8 | 12.8 |
| 3 | 10 | seguir | 3♦ | 3♦ | 9.0 | 9.0 |
| 4 | 10 | seguir | 3♥ | 10♥ | 4.0 | 4.0 |
| 5 | 11 | liderar | 10♥ | 2♥ | 3.9 | 4.0 |
| 6 | 10 | seguir | 3♥ | 10♥ | 3.4 | 3.4 |
| 7 | 10 | liderar | 11♥ | 2♥ | 3.4 | 3.4 |
| 8 | 12 | descartar | 3♦ | 3♦ | 3.3 | 3.3 |
| 9 | 10 | descartar | 7♥ | 11♥ | 2.8 | 3.5 |
| 10 | 11 | seguir | 3♥ | 11♥ | 2.1 | 2.1 |

## 5. Distribución del Coste de Errores

| Percentil | Coste (pts) |
|-----------|-------------|
| P50 | 2.45 |
| P75 | 3.80 |
| P90 | 10.14 |
| P95 | 13.46 |
| P99 | 16.51 |
| P100 | 17.27 |

## 6. Análisis Cualitativo

### Errores al Liderar (2 casos)

- Coste medio: 3.64 pts
- Suele ocurrir cuando el modelo **no identifica la carta óptima para iniciar la baza**.
- El modelo puede estar jugando cartas demasiado altas o demasiado bajas para la fase de la mano.

### Errores al Seguir Palo (8 casos)

- Coste medio: 6.19 pts
- El modelo **no ajusta bien la altura de la carta** al seguir el palo.
- Posiblemente juega cartas muy altas cuando debería jugar bajas (o viceversa).

### Errores al Descartar (8 casos)

- Coste medio: 1.13 pts
- El modelo **elige mal qué palo descartar** o **tira una carta incorrecta**.
- Posiblemente no está considerando correctamente los vacíos de los oponentes.

## 7. Implicaciones para v5

- **Coste total por mano**: 1.3 pts de error evitable.
- Si v5 logra reducir este coste a la mitad, la puntuación media mejoraría ~0.7 pts.
- Las bazas críticas son: 10, 11, 12.
- v5 con self-play puro debería mostrar **menor coste en bazas tardías** (10+) donde PIMC tiene ground truth.

## 8. Manos con Más Errores

| Seed | Puntos | Q♠ | Errores | Coste Total |
|------|--------|-----|---------|-------------|
| 78 | 7 | No | 2 | 7.4 |
| 90 | 0 | No | 2 | 4.5 |
| 43 | 0 | Baza 6 | 1 | 0.3 |
| 53 | 1 | No | 1 | 0.2 |
| 56 | 0 | No | 1 | 0.4 |
| 63 | 17 | No | 1 | 3.4 |
| 67 | 0 | No | 1 | 0.6 |
| 68 | 4 | Baza 9 | 1 | 12.8 |
| 72 | 0 | No | 1 | 2.1 |
| 77 | 2 | Baza 4 | 1 | 17.3 |