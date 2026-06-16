# Catálogo de Patrones de Error — BotExperto

**Manos analizadas:** 1000  
**Fecha:** 2026-06-16 14:04  
**Total patrones detectados:** 19  

---

## `liderar_otro`

- **Ocurrencias:** 729
- **Coste total:** 1470.3 pts
- **Coste medio:** 2.02 pts
- **Coste máximo:** 19.24 pts
- **Situación:** liderar

**Descripción:** Error al liderar no clasificado en categorías específicas.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 11 | 3T | 9P | 3.3 | ✓ |
| 2 | 6D | 9T | 2.6 | ~ |
| 2 | 2P | 8P | 1.0 | ~ |
| 10 | 13C | 8P | 0.7 | ✓ |
| 6 | 3T | 3D | 0.8 | ~ |

## `seguir_otro`

- **Ocurrencias:** 570
- **Coste total:** 694.9 pts
- **Coste medio:** 1.22 pts
- **Coste máximo:** 7.60 pts
- **Situación:** seguir

**Descripción:** Error al seguir palo no clasificado en categorías específicas.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 1 | 9T | 3T | 0.4 | ~ |
| 3 | 10T | 3T | 0.8 | ~ |
| 4 | 8C | 12C | 2.1 | ~ |
| 5 | 9T | 6T | 0.5 | ~ |
| 1 | 10T | 8T | 2.5 | ~ |

## `seguir_no_quema_alta_en_baza_limpia`

- **Ocurrencias:** 428
- **Coste total:** 620.2 pts
- **Coste medio:** 1.45 pts
- **Coste máximo:** 6.93 pts
- **Situación:** seguir

**Descripción:** En baza sin puntos, jugó carta baja/media en vez de A/K del palo. Debía quemar la carta alta en baza limpia para eliminar liability futura.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 2 | 4P | 11P | 0.6 | ~ |
| 4 | 4D | 14D | 1.0 | ~ |
| 3 | 6D | 13D | 0.3 | ~ |
| 5 | 7T | 13T | 1.0 | ~ |
| 2 | 4D | 14D | 5.7 | ~ |

## `descarte_otro`

- **Ocurrencias:** 323
- **Coste total:** 425.2 pts
- **Coste medio:** 1.32 pts
- **Coste máximo:** 14.87 pts
- **Situación:** descartar

**Descripción:** Error al descartar no clasificado en categorías específicas.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 7 | 14C | 10P | 0.4 | ~ |
| 9 | 11C | 10D | 2.9 | ~ |
| 5 | 7C | 14D | 1.5 | ~ |
| 8 | 6D | 7C | 1.6 | ~ |
| 1 | 13P | 13D | 2.6 | ~ |

## `liderar_pica_innecesaria`

- **Ocurrencias:** 208
- **Coste total:** 342.0 pts
- **Coste medio:** 1.64 pts
- **Coste máximo:** 11.07 pts
- **Situación:** liderar

**Descripción:** Lideró pica (no Q♠) cuando había mejores opciones. Las picas son peligrosas mientras Q♠ está activa.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 3 | 2P | 14D | 5.7 | ~ |
| 5 | 2P | 10T | 0.8 | ~ |
| 2 | 4P | 5T | 1.6 | ~ |
| 5 | 2P | 5T | 1.0 | ~ |
| 3 | 2P | 12T | 2.2 | ~ |

## `descarte_no_suelta_q_espadas`

- **Ocurrencias:** 56
- **Coste total:** 325.8 pts
- **Coste medio:** 5.82 pts
- **Coste máximo:** 13.30 pts
- **Situación:** descartar

**Descripción:** Pudiendo descartar Q♠ en baza con puntos, no lo hizo. Perdió oportunidad de endosar 13 pts a un rival.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 11C | 12P | 7.8 | ~ |
| 3 | 13C | 12P | 6.8 | ~ |
| 3 | 13P | 12P | 2.6 | ~ |
| 4 | 13C | 12P | 2.5 | ~ |
| 2 | 11C | 12P | 6.8 | ~ |

## `liderar_no_quema_maxima`

- **Ocurrencias:** 142
- **Coste total:** 275.8 pts
- **Coste medio:** 1.94 pts
- **Coste máximo:** 8.50 pts
- **Situación:** liderar

**Descripción:** Tenía la máxima de un palo seguro y no la lideró. Perdió oportunidad de ganar baza limpia y quemar liability.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 4D | 11P | 0.3 | ~ |
| 6 | 3D | 13T | 3.1 | ~ |
| 8 | 4T | 11P | 2.1 | ~ |
| 10 | 8T | 11P | 5.3 | ✓ |
| 11 | 10D | 11P | 6.0 | ✓ |

## `seguir_q_espadas_mal_momento`

- **Ocurrencias:** 35
- **Coste total:** 171.9 pts
- **Coste medio:** 4.91 pts
- **Coste máximo:** 9.23 pts
- **Situación:** seguir

**Descripción:** Jugó Q♠ siguiendo picas cuando no era óptimo.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 12P | 9P | 7.4 | ~ |
| 11 | 12P | 7P | 5.7 | ✓ |
| 6 | 12P | 7P | 6.4 | ~ |
| 3 | 12P | 4P | 2.3 | ~ |
| 2 | 12P | 5P | 4.7 | ~ |

## `liderar_q_espadas_mal_momento`

- **Ocurrencias:** 25
- **Coste total:** 159.9 pts
- **Coste medio:** 6.40 pts
- **Coste máximo:** 10.27 pts
- **Situación:** liderar

**Descripción:** Lideró Q♠ en momento inadecuado (demasiado pronto o cuando era máxima en picas).

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 7 | 12P | 11T | 2.8 | ~ |
| 9 | 12P | 7P | 3.8 | ~ |
| 10 | 12P | 13D | 9.7 | ✓ |
| 7 | 12P | 11T | 9.2 | ~ |
| 7 | 12P | 12T | 3.9 | ~ |

## `seguir_no_suelta_q_con_altas`

- **Ocurrencias:** 16
- **Coste total:** 140.7 pts
- **Coste medio:** 8.79 pts
- **Coste máximo:** 13.40 pts
- **Situación:** seguir

**Descripción:** Tenía Q♠ y A♠/K♠, y jugó A♠/K♠ en vez de Q♠. PIMC dice que era mejor soltar Q♠ para que otro la capture.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 3 | 14P | 12P | 12.2 | ~ |
| 5 | 13P | 12P | 7.4 | ~ |
| 6 | 14P | 12P | 2.8 | ~ |
| 3 | 14P | 12P | 12.5 | ~ |
| 2 | 13P | 12P | 10.4 | ~ |

## `seguir_no_quema_maxima_forzada`

- **Ocurrencias:** 115
- **Coste total:** 137.9 pts
- **Coste medio:** 1.20 pts
- **Coste máximo:** 8.67 pts
- **Situación:** seguir

**Descripción:** Forzado a ganar la baza (todas sus cartas > ganadora actual), pero jugó la más baja en vez de la más alta. Debía quemar la máxima.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 1 | 5T | 10T | 1.0 | ~ |
| 4 | 5P | 10P | 1.0 | ~ |
| 1 | 3T | 8T | 0.9 | ~ |
| 2 | 3D | 10D | 1.1 | ~ |
| 5 | 7T | 10T | 1.9 | ~ |

## `seguir_gana_baza_con_puntos`

- **Ocurrencias:** 57
- **Coste total:** 77.2 pts
- **Coste medio:** 1.35 pts
- **Coste máximo:** 10.96 pts
- **Situación:** seguir

**Descripción:** Jugó carta que gana la baza cuando había puntos en mesa, pudiendo jugar una perdedora. Capturó puntos innecesariamente.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 4 | 10D | 8D | 0.4 | ~ |
| 4 | 10T | 8T | 0.6 | ~ |
| 2 | 9P | 5P | 0.3 | ~ |
| 3 | 6D | 4D | 1.0 | ~ |
| 3 | 10P | 5P | 1.3 | ~ |

## `liderar_quemar_cuando_debe_ceder`

- **Ocurrencias:** 19
- **Coste total:** 65.2 pts
- **Coste medio:** 3.43 pts
- **Coste máximo:** 19.27 pts
- **Situación:** liderar

**Descripción:** En baza ≥9, lideró carta alta de palo seguro (gana la baza) cuando debía liderar baja (ceder el lead). Ganar fuerza a liderar de nuevo.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 10 | 11P | 6P | 0.7 | ✓ |
| 9 | 13P | 11T | 1.8 | ~ |
| 9 | 14D | 5D | 1.3 | ~ |
| 11 | 13P | 4P | 1.5 | ✓ |
| 12 | 10P | 4P | 3.0 | ✓ |

## `descarte_no_suelta_picas_altas`

- **Ocurrencias:** 25
- **Coste total:** 55.8 pts
- **Coste medio:** 2.23 pts
- **Coste máximo:** 7.13 pts
- **Situación:** descartar

**Descripción:** Con Q♠ activa, no descartó K♠/A♠ en baza limpia. Estas cartas son peligrosas: ganan bazas de ♠ donde Q♠ puede caer encima.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 12C | 13P | 3.3 | ~ |
| 7 | 14C | 14P | 0.7 | ~ |
| 5 | 14C | 13P | 1.5 | ~ |
| 3 | 14C | 14P | 1.5 | ~ |
| 8 | 13C | 14P | 0.9 | ~ |

## `liderar_no_suelta_q_espadas`

- **Ocurrencias:** 4
- **Coste total:** 33.9 pts
- **Coste medio:** 8.48 pts
- **Coste máximo:** 15.50 pts
- **Situación:** liderar

**Descripción:** No lideró Q♠ cuando debía (baza tardía con K♠/A♠ en circulación).

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 7 | 13T | 12P | 14.1 | ~ |
| 8 | 4P | 12P | 15.5 | ~ |
| 4 | 3T | 12P | 0.6 | ~ |
| 7 | 5D | 12P | 3.7 | ~ |

## `descarte_no_suelta_corazon_bajo`

- **Ocurrencias:** 16
- **Coste total:** 22.0 pts
- **Coste medio:** 1.37 pts
- **Coste máximo:** 3.78 pts
- **Situación:** descartar

**Descripción:** Con corazones rotos, no descartó corazón bajo en baza limpia. Los corazones bajos pueden forzar ganar bazas de ♥ con puntos.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 12 | 2P | 11C | 0.7 | ✓ |
| 7 | 13D | 6C | 0.9 | ~ |
| 10 | 11T | 2C | 0.8 | ✓ |
| 11 | 14T | 10C | 2.8 | ✓ |
| 7 | 14T | 10C | 3.0 | ~ |

## `liderar_corazon_tardio`

- **Ocurrencias:** 7
- **Coste total:** 20.5 pts
- **Coste medio:** 2.92 pts
- **Coste máximo:** 9.43 pts
- **Situación:** liderar

**Descripción:** En baza ≥8, lideró carta sin puntos cuando debió liderar corazón medio. PIMC dice que dump de corazón es más seguro que guardarlo.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 8 | 3D | 7C | 2.2 | ~ |
| 8 | 14T | 7C | 1.2 | ~ |
| 8 | 6D | 6C | 2.0 | ~ |
| 8 | 8D | 6C | 1.0 | ~ |
| 8 | 13D | 7C | 1.5 | ~ |

## `descarte_q_espadas_prematuro`

- **Ocurrencias:** 7
- **Coste total:** 9.0 pts
- **Coste medio:** 1.28 pts
- **Coste máximo:** 2.07 pts
- **Situación:** descartar

**Descripción:** Descartó Q♠ en momento inadecuado (baza sin puntos, demasiado pronto).

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 8 | 12P | 5D | 2.0 | ~ |
| 7 | 12P | 9D | 1.8 | ~ |
| 7 | 12P | 8T | 2.1 | ~ |
| 10 | 12P | 10T | 0.9 | ✓ |
| 8 | 12P | 10C | 0.9 | ~ |

## `descarte_ases_en_vez_de_corazones`

- **Ocurrencias:** 1
- **Coste total:** 1.5 pts
- **Coste medio:** 1.45 pts
- **Coste máximo:** 1.45 pts
- **Situación:** descartar

**Descripción:** Descartó A♣/A♦ en vez de corazón alto. Los ases de palos seguros ganan bazas y atraen corazones descartados.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 10 | 14D | 11C | 1.4 | ✓ |
