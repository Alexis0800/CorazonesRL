# Mapeo Patrones → Recompensas/Castigos RL

Se sugiere añadir las siguientes señales al `RewardConfig` y
a `CalculadoraRecompensas` para que el modelo RL aprenda estas
distinciones que el BotExperto actualmente no hace.

---

## 🟢 `REWARD_DESCARTAR_ASES_SEGUROS` — ⬜ PENDIENTE

- **Patrón asociado:** `descarte_ases_en_vez_de_corazones` (1 ocurrencia, 1.5 pts) — muy bajo volumen
- **Tipo:** recompensa
- **Valor sugerido:** +1.0
- **Condición:** `Se descarta A♣/A♦ en baza limpia antes que corazones`
- **Justificación:** Ases ganan bazas y atraen descartes de corazones.

## 🟢 `REWARD_DESCARTAR_CORAZON_BAJO_ROTO` — ⬜ PENDIENTE

- **Patrón asociado:** `descarte_no_suelta_corazon_bajo` (16 ocurrencias, 22 pts) — bajo volumen
- **Tipo:** recompensa
- **Valor sugerido:** +0.5
- **Condición:** `corazones_rotos AND se descarta corazón en baza limpia`
- **Justificación:** Con corazones rotos, cualquier corazón es peligroso.

## 🟢 `REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA` — ✅ IMPLEMENTADO

- **Patrón asociado:** `descarte_no_suelta_picas_altas` (25 ocurrencias, 56 pts)
- **Tipo:** recompensa
- **Valor:** +2.0
- **Condición:** `puntos_baza == 0 AND Q♠ activa AND se descarta K♠/A♠ (void en palo de salida)`
- **Justificación:** K♠/A♠ son liability con Q♠ activa.

## 🟢 `REWARD_DESCARTAR_DAMA_SEGURO` — ✅ SUBIDO A 8.0

- **Patrón asociado:** `descarte_no_suelta_q_espadas` (56 ocurrencias, 326 pts)
- **Tipo:** recompensa
- **Valor:** ~~5.0~~ → **8.0**
- **Condición:** `YA EXISTÍA. Se subió de 3.0 → 8.0.`
- **Justificación:** Descartar Q♠ en baza con puntos es crítico.

## 🔴 `PENALTY_DESCARTAR_Q_PREMATURO` — ❌ DESCARTADA

- **Patrón asociado:** `descarte_q_espadas_prematuro`
- **Tipo:** penalización
- **Valor sugerido:** ~~-3.0~~
- **Condición:** ~~`baza < 6 AND se descarta Q♠ en baza sin puntos`~~
- **Justificación del rechazo:** Datos muestran bajo impacto (7 ocurrencias, coste medio 1.28 pts).
  Soltar Q♠ pronto y sin recibir puntos es una estrategia válida. La señal ya existe
  en positivo (`REWARD_DESCARTAR_DAMA_SEGURO = 5.0` en bazas con puntos).

## 🔴 `PENALTY_LIDERAR_CORAZON_ALTO_TARDIO` — ⬜ PENDIENTE

- **Patrón asociado:** `liderar_corazon_demasiado_alto` — patrón no detectado en v2
- **Tipo:** penalización
- **Valor sugerido:** -1.5
- **Condición:** `baza ≥ 9 AND se lidera corazón J/Q/K/A teniendo corazones ≤10`
- **Justificación:** Corazones altos son para control, no para dump.

## 🟢 `REWARD_LIDERAR_CORAZON_TARDIO` — ⬜ PENDIENTE

- **Patrón asociado:** `liderar_corazon_tardio` (7 ocurrencias, 21 pts) — bajo volumen
- **Tipo:** recompensa
- **Valor sugerido:** +1.0
- **Condición:** `baza ≥ 8 AND corazones_rotos AND se lidera corazón medio (≤10)`
- **Justificación:** Dump seguro de corazones en final de mano.

## 🟢 `REWARD_QUEMAR_MAXIMA_PALO_SEGURO` — ✅ IMPLEMENTADO

- **Patrón asociado:** `liderar_no_quema_maxima` (142 ocurrencias, 276 pts)
- **Tipo:** recompensa
- **Valor:** +0.5
- **Condición:** `posición == 0 AND palo in (♣,♦) AND es máxima en mano`
- **Tipo:** recompensa
- **Valor sugerido:** +0.5
- **Condición:** `baza ≥ 6 AND se lidera la máxima de ♣/♦`
- **Justificación:** Quemar máxima en palo seguro garantiza baza limpia.

## 🟢 `REWARD_LIDERAR_Q_DUMP_SEGURO` — ⬜ PENDIENTE (cubierto parcialmente)

- **Patrón asociado:** `liderar_no_suelta_q_espadas` (4 ocurrencias, 34 pts) — bajo volumen
- **Nota:** `PENALTY_LIDERAR_Q_EQUIVOCADO` cubre el caso negativo.
- **Tipo:** recompensa
- **Valor sugerido:** +3.0
- **Condición:** `baza ≥ 7 AND K♠/A♠ en circulación AND se lidera Q♠`
- **Justificación:** Dump de Q♠ cuando hay cobertura de K♠/A♠.

## 🔴 `PENALTY_LIDERAR_PICA_CON_Q_ACTIVA` — ✅ IMPLEMENTADO

- **Patrón asociado:** `liderar_pica_innecesaria` (208 ocurrencias, coste medio 1.64 pts)
- **Tipo:** penalización
- **Valor:** -1.0
- **Condición:** `Q♠ activa AND posición == 0 (lidera) AND palo == picas AND NO es máxima pica del agente`
- **Guard clause:** No aplica si el agente es último en jugar (posición 3) — liderar ahí es 100% seguro.
- **Justificación:** Liderar picas no-máximas con Q♠ activa invita a que te la tiren encima.

## 🔴 `PENALTY_LIDERAR_Q_EQUIVOCADO` — ✅ IMPLEMENTADO

- **Patrón asociado:** `liderar_q_espadas_mal_momento` (25 ocurrencias, 160 pts, media 6.40)
- **Tipo:** penalización
- **Valor:** -3.0
- **Condición:** `posición == 0 AND es Q♠ AND (baza < 7 OR soy máxima en picas)`
- **Tipo:** penalización
- **Valor sugerido:** -3.0
- **Condición:** `Se lidera Q♠ cuando soy máxima en picas o baza < 7`
- **Justificación:** Liderar Q♠ sin K♠/A♠ en circulación = auto-13pts.

## 🔴 `PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD` — ⬜ PENDIENTE

- **Patrón asociado:** `liderar_quemar_cuando_debe_ceder` (19 ocurrencias, 65 pts) — medio volumen, alto coste medio
- **Tipo:** penalización
- **Valor sugerido:** -2.0
- **Condición:** `baza ≥ 9 AND se gana baza sin puntos con carta alta de palo seguro`
- **Justificación:** Ganar baza en final de mano fuerza a liderar de nuevo.

## 🔴 `PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE` — ✅ IMPLEMENTADO

- **Patrón asociado:** `seguir_gana_baza_con_puntos` (57 ocurrencias, 77 pts)
- **Tipo:** penalización
- **Valor:** -3.0
- **Condición:** `ganador == agente AND puntos_baza > 0 AND siguió palo AND no pozo_viable`
- **Tipo:** penalización
- **Valor sugerido:** -3.0
- **Condición:** `puntos_mesa > 0 AND se juega carta > ganadora teniendo perdedora`
- **Justificación:** Capturar puntos cuando había alternativa de perder.

## 🟢 `REWARD_QUEMAR_ALTA_SIGUIENDO_PALO` — ✅ IMPLEMENTADO

- **Patrón asociado:** `seguir_no_quema_alta_en_baza_limpia` (428 ocurrencias, 620 pts) ⭐ MÁXIMO IMPACTO
- **Tipo:** recompensa
- **Valor:** +0.5
- **Condición:** `puntos_baza == 0 AND siguió palo AND valor >= 13 (A/K)`
- **Tipo:** recompensa
- **Valor sugerido:** +0.5
- **Condición:** `puntos_mesa == 0 AND se juega A/K del palo`
- **Justificación:** Ya cubierto parcialmente. Refuerzo adicional.

## 🟢 `REWARD_QUEMAR_MAXIMA_FORZADA` — ✅ IMPLEMENTADO

- **Patrón asociado:** `seguir_no_quema_maxima_forzada` (115 ocurrencias, 138 pts)
- **Tipo:** recompensa
- **Valor:** +0.3
- **Condición:** `siguiendo palo AND forzado a ganar AND jugó la máxima`
- **Tipo:** recompensa
- **Valor sugerido:** +0.3
- **Condición:** `Todas las cartas ganan AND se juega la máxima`
- **Justificación:** Si vas a ganar igual, quema la más alta.

## 🟢 `REWARD_DUMP_Q_SIGUIENDO_PICAS` — ✅ IMPLEMENTADO

- **Patrón asociado:** `seguir_no_suelta_q_con_altas` (16 ocurrencias, 141 pts, media 8.79) ⭐ ALTO COSTE
- **Tipo:** recompensa
- **Valor:** +2.0
- **Condición:** `es Q♠ AND no ganó baza AND palo_salida == picas`
- **Tipo:** recompensa
- **Valor sugerido:** +2.0
- **Condición:** `siguiendo ♠ AND se juega Q♠ AND K♠/A♠ en circulación`
- **Justificación:** Soltar Q♠ cuando hay cobertura.

---

## Tabla Resumen

| Reward/Penalty | Estado | Valor | Ocurrencias | Coste Total |
|---------------|--------|-------|-------------|-------------|
| `REWARD_QUEMAR_ALTA_SIGUIENDO_PALO` | ✅ | +0.5 | 428 | 620 pts |
| `REWARD_DESCARTAR_DAMA_SEGURO` | ✅ 8.0 | +8.0 | 56 | 326 pts |
| `REWARD_QUEMAR_MAXIMA_PALO_SEGURO` | ✅ | +0.5 | 142 | 276 pts |
| `PENALTY_LIDERAR_PICA_CON_Q_ACTIVA` | ✅ | -1.0 | 208 | 342 pts |
| `PENALTY_LIDERAR_Q_EQUIVOCADO` | ✅ | -3.0 | 25 | 160 pts |
| `REWARD_DUMP_Q_SIGUIENDO_PICAS` | ✅ | +2.0 | 16 | 141 pts |
| `REWARD_QUEMAR_MAXIMA_FORZADA` | ✅ | +0.3 | 115 | 138 pts |
| `PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE` | ✅ | -3.0 | 57 | 77 pts |
| `REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA` | ✅ | +2.0 | 25 | 56 pts |
| `PENALTY_DESCARTAR_Q_PREMATURO` | ❌ | — | 7 | 9 pts |
| `REWARD_DESCARTAR_CORAZON_BAJO_ROTO` | ⬜ | +0.5 | 16 | 22 pts |
| `REWARD_LIDERAR_CORAZON_TARDIO` | ⬜ | +1.0 | 7 | 21 pts |
| `PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD` | ⬜ | -2.0 | 19 | 65 pts |
| `REWARD_LIDERAR_Q_DUMP_SEGURO` | ⬜ | +3.0 | 4 | 34 pts |
| `PENALTY_LIDERAR_CORAZON_ALTO_TARDIO` | ⬜ | -1.5 | — | — |
| `REWARD_DESCARTAR_ASES_SEGUROS` | ⬜ | +1.0 | 1 | 1.5 pts |

✅ = Implementado | ❌ = Descartado | ⬜ = Pendiente
