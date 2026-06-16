# Catálogo de Patrones de Error — BotExperto

**Manos analizadas:** 200  
**Fecha:** 2026-06-16 13:35  
**Total patrones detectados:** 16  

---

## `liderar_otro`

- **Ocurrencias:** 143
- **Coste total:** 308.4 pts
- **Coste medio:** 2.16 pts
- **Coste máximo:** 9.80 pts
- **Situación:** liderar

**Descripción:** Error al liderar no clasificado en categorías específicas.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 9 | 12C | 11D | 0.3 | ~ |
| 10 | 13C | 11D | 1.1 | ✓ |
| 8 | 7T | 14C | 4.0 | ~ |
| 10 | 8P | 2P | 7.0 | ✓ |
| 2 | 3D | 8D | 2.0 | ~ |

## `seguir_otro`

- **Ocurrencias:** 127
- **Coste total:** 185.5 pts
- **Coste medio:** 1.46 pts
- **Coste máximo:** 8.47 pts
- **Situación:** seguir

**Descripción:** Error al seguir palo no clasificado en categorías específicas.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 1 | 12T | 3T | 0.4 | ~ |
| 7 | 10C | 13C | 1.9 | ~ |
| 1 | 12T | 5T | 0.8 | ~ |
| 1 | 14T | 9T | 1.1 | ~ |
| 7 | 11C | 14C | 0.8 | ~ |

## `seguir_no_quema_alta_en_baza_limpia`

- **Ocurrencias:** 78
- **Coste total:** 114.3 pts
- **Coste medio:** 1.47 pts
- **Coste máximo:** 4.87 pts
- **Situación:** seguir

**Descripción:** En baza sin puntos, jugó carta baja/media en vez de A/K del palo. Debía quemar la carta alta en baza limpia para eliminar liability futura.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 3 | 7D | 14D | 0.6 | ~ |
| 5 | 4D | 12D | 2.1 | ~ |
| 6 | 4D | 13D | 2.0 | ~ |
| 4 | 7D | 12D | 0.6 | ~ |
| 1 | 10T | 14T | 0.7 | ~ |

## `liderar_pica_innecesaria`

- **Ocurrencias:** 41
- **Coste total:** 85.7 pts
- **Coste medio:** 2.09 pts
- **Coste máximo:** 7.53 pts
- **Situación:** liderar

**Descripción:** Lideró pica (no Q♠) cuando había mejores opciones. Las picas son peligrosas mientras Q♠ está activa.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 4P | 8T | 0.7 | ~ |
| 8 | 7P | 8T | 0.7 | ~ |
| 3 | 2P | 14T | 3.1 | ~ |
| 6 | 14P | 10T | 6.7 | ~ |
| 7 | 10P | 10T | 6.5 | ~ |

## `descarte_no_suelta_q_espadas`

- **Ocurrencias:** 9
- **Coste total:** 69.9 pts
- **Coste medio:** 7.76 pts
- **Coste máximo:** 13.00 pts
- **Situación:** descartar

**Descripción:** Pudiendo descartar Q♠ en baza con puntos, no lo hizo. Perdió oportunidad de endosar 13 pts a un rival.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 11P | 12P | 13.0 | ~ |
| 5 | 14T | 12P | 7.3 | ~ |
| 5 | 14C | 12P | 7.2 | ~ |
| 5 | 10T | 12P | 2.5 | ~ |
| 4 | 13C | 12P | 6.1 | ~ |

## `descarte_otro`

- **Ocurrencias:** 56
- **Coste total:** 55.8 pts
- **Coste medio:** 1.00 pts
- **Coste máximo:** 2.73 pts
- **Situación:** descartar

**Descripción:** Error al descartar no clasificado en categorías específicas.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 12D | 8C | 1.4 | ~ |
| 7 | 8C | 8T | 0.6 | ~ |
| 8 | 11C | 10T | 0.6 | ~ |
| 9 | 7C | 10T | 0.4 | ~ |
| 3 | 14P | 14D | 0.4 | ~ |

## `liderar_no_quema_maxima`

- **Ocurrencias:** 28
- **Coste total:** 39.8 pts
- **Coste medio:** 1.42 pts
- **Coste máximo:** 4.87 pts
- **Situación:** liderar

**Descripción:** Tenía la máxima de un palo seguro y no la lideró. Perdió oportunidad de ganar baza limpia y quemar liability.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 5 | 4P | 13T | 2.2 | ~ |
| 6 | 5D | 11T | 0.7 | ~ |
| 6 | 2D | 11P | 1.3 | ~ |
| 5 | 4D | 11D | 1.7 | ~ |
| 7 | 3D | 13P | 2.1 | ~ |

## `seguir_q_espadas_mal_momento`

- **Ocurrencias:** 6
- **Coste total:** 26.2 pts
- **Coste medio:** 4.36 pts
- **Coste máximo:** 8.47 pts
- **Situación:** seguir

**Descripción:** Jugó Q♠ siguiendo picas cuando no era óptimo.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 7 | 12P | 7P | 8.3 | ~ |
| 3 | 12P | 9P | 1.1 | ~ |
| 6 | 12P | 9P | 3.3 | ~ |
| 3 | 12P | 10P | 4.0 | ~ |
| 7 | 12P | 11P | 8.5 | ~ |

## `liderar_q_espadas_mal_momento`

- **Ocurrencias:** 4
- **Coste total:** 24.8 pts
- **Coste medio:** 6.19 pts
- **Coste máximo:** 9.57 pts
- **Situación:** liderar

**Descripción:** Lideró Q♠ en momento inadecuado (demasiado pronto o cuando era máxima en picas).

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 7 | 12P | 11P | 1.4 | ~ |
| 9 | 12P | 13D | 5.9 | ~ |
| 7 | 12P | 14C | 9.6 | ~ |
| 7 | 12P | 3D | 7.9 | ~ |

## `seguir_gana_baza_con_puntos`

- **Ocurrencias:** 12
- **Coste total:** 16.5 pts
- **Coste medio:** 1.38 pts
- **Coste máximo:** 5.15 pts
- **Situación:** seguir

**Descripción:** Jugó carta que gana la baza cuando había puntos en mesa, pudiendo jugar una perdedora. Capturó puntos innecesariamente.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 4 | 10T | 7T | 0.3 | ~ |
| 2 | 8T | 4T | 1.2 | ~ |
| 6 | 12D | 4D | 0.6 | ~ |
| 10 | 14C | 6C | 2.0 | ✓ |
| 11 | 8C | 4C | 5.2 | ✓ |

## `seguir_no_quema_maxima_forzada`

- **Ocurrencias:** 18
- **Coste total:** 14.4 pts
- **Coste medio:** 0.80 pts
- **Coste máximo:** 1.70 pts
- **Situación:** seguir

**Descripción:** Forzado a ganar la baza (todas sus cartas > ganadora actual), pero jugó la más baja en vez de la más alta. Debía quemar la máxima.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 1 | 3T | 8T | 0.6 | ~ |
| 3 | 5P | 10P | 1.2 | ~ |
| 4 | 2D | 10D | 0.7 | ~ |
| 4 | 3T | 8T | 1.7 | ~ |
| 1 | 3T | 7T | 0.6 | ~ |

## `descarte_no_suelta_corazon_bajo`

- **Ocurrencias:** 7
- **Coste total:** 14.0 pts
- **Coste medio:** 2.00 pts
- **Coste máximo:** 5.20 pts
- **Situación:** descartar

**Descripción:** Con corazones rotos, no descartó corazón bajo en baza limpia. Los corazones bajos pueden forzar ganar bazas de ♥ con puntos.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 8 | 14D | 9C | 0.6 | ~ |
| 9 | 8D | 14C | 4.5 | ~ |
| 9 | 11P | 7C | 2.0 | ~ |
| 10 | 9T | 7C | 0.6 | ✓ |
| 12 | 5T | 7C | 0.7 | ✓ |

## `seguir_no_suelta_q_con_altas`

- **Ocurrencias:** 1
- **Coste total:** 13.1 pts
- **Coste medio:** 13.13 pts
- **Coste máximo:** 13.13 pts
- **Situación:** seguir

**Descripción:** Tenía Q♠ y A♠/K♠, y jugó A♠/K♠ en vez de Q♠. PIMC dice que era mejor soltar Q♠ para que otro la capture.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 3 | 13P | 12P | 13.1 | ~ |

## `descarte_ases_en_vez_de_corazones`

- **Ocurrencias:** 1
- **Coste total:** 6.9 pts
- **Coste medio:** 6.93 pts
- **Coste máximo:** 6.93 pts
- **Situación:** descartar

**Descripción:** Descartó A♣/A♦ en vez de corazón alto. Los ases de palos seguros ganan bazas y atraen corazones descartados.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 7 | 14T | 13C | 6.9 | ~ |

## `descarte_no_suelta_picas_altas`

- **Ocurrencias:** 3
- **Coste total:** 4.6 pts
- **Coste medio:** 1.52 pts
- **Coste máximo:** 1.97 pts
- **Situación:** descartar

**Descripción:** Con Q♠ activa, no descartó K♠/A♠ en baza limpia. Estas cartas son peligrosas: ganan bazas de ♠ donde Q♠ puede caer encima.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 2 | 11C | 14P | 2.0 | ~ |
| 10 | 14C | 14P | 1.8 | ✓ |
| 4 | 12C | 13P | 0.8 | ~ |

## `liderar_quemar_cuando_debe_ceder`

- **Ocurrencias:** 3
- **Coste total:** 4.5 pts
- **Coste medio:** 1.50 pts
- **Coste máximo:** 2.00 pts
- **Situación:** liderar

**Descripción:** En baza ≥9, lideró carta alta de palo seguro (gana la baza) cuando debía liderar baja (ceder el lead). Ganar fuerza a liderar de nuevo.

**Ejemplos:**

| Baza | Bot | PIMC | Coste | Exacto |
|------|-----|------|-------|--------|
| 11 | 13P | 6P | 2.0 | ✓ |
| 10 | 10P | 3P | 0.5 | ✓ |
| 10 | 14P | 12D | 2.0 | ✓ |
