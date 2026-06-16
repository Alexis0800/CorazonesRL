# Mapeo Patrones → Recompensas/Castigos RL

Se sugiere añadir las siguientes señales al `RewardConfig` y
a `CalculadoraRecompensas` para que el modelo RL aprenda estas
distinciones que el BotExperto actualmente no hace.

---

## 🟢 `REWARD_DESCARTAR_ASES_SEGUROS`

- **Patrón asociado:** `descarte_ases_en_vez_de_corazones`
- **Tipo:** recompensa
- **Valor sugerido:** +1.0
- **Condición:** `Se descarta A♣/A♦ en baza limpia antes que corazones`
- **Justificación:** Ases ganan bazas y atraen descartes de corazones.

## 🟢 `REWARD_DESCARTAR_CORAZON_BAJO_ROTO`

- **Patrón asociado:** `descarte_no_suelta_corazon_bajo`
- **Tipo:** recompensa
- **Valor sugerido:** +0.5
- **Condición:** `corazones_rotos AND se descarta corazón en baza limpia`
- **Justificación:** Con corazones rotos, cualquier corazón es peligroso.

## 🟢 `REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA`

- **Patrón asociado:** `descarte_no_suelta_picas_altas`
- **Tipo:** recompensa
- **Valor sugerido:** +2.0
- **Condición:** `Q♠ activa AND se descarta K♠/A♠ en baza limpia`
- **Justificación:** K♠/A♠ son liability con Q♠ activa.

## 🟢 `REWARD_DESCARTAR_DAMA_SEGURO`

- **Patrón asociado:** `descarte_no_suelta_q_espadas`
- **Tipo:** recompensa
- **Valor sugerido:** +5.0
- **Condición:** `YA EXISTE. Subir valor a 8.0.`
- **Justificación:** Descartar Q♠ en baza con puntos es crítico.

## 🔴 `PENALTY_DESCARTAR_Q_PREMATURO`

- **Patrón asociado:** `descarte_q_espadas_prematuro`
- **Tipo:** penalización
- **Valor sugerido:** -3.0
- **Condición:** `baza < 6 AND se descarta Q♠ en baza sin puntos`
- **Justificación:** Q♠ debe guardarse para bazas con puntos.

## 🔴 `PENALTY_LIDERAR_CORAZON_ALTO_TARDIO`

- **Patrón asociado:** `liderar_corazon_demasiado_alto`
- **Tipo:** penalización
- **Valor sugerido:** -1.5
- **Condición:** `baza ≥ 9 AND se lidera corazón J/Q/K/A teniendo corazones ≤10`
- **Justificación:** Corazones altos son para control, no para dump.

## 🟢 `REWARD_LIDERAR_CORAZON_TARDIO`

- **Patrón asociado:** `liderar_corazon_tardio`
- **Tipo:** recompensa
- **Valor sugerido:** +1.0
- **Condición:** `baza ≥ 8 AND corazones_rotos AND se lidera corazón medio (≤10)`
- **Justificación:** Dump seguro de corazones en final de mano.

## 🟢 `REWARD_QUEMAR_MAXIMA_PALO_SEGURO`

- **Patrón asociado:** `liderar_no_quema_maxima`
- **Tipo:** recompensa
- **Valor sugerido:** +0.5
- **Condición:** `baza ≥ 6 AND se lidera la máxima de ♣/♦`
- **Justificación:** Quemar máxima en palo seguro garantiza baza limpia.

## 🟢 `REWARD_LIDERAR_Q_DUMP_SEGURO`

- **Patrón asociado:** `liderar_no_suelta_q_espadas`
- **Tipo:** recompensa
- **Valor sugerido:** +3.0
- **Condición:** `baza ≥ 7 AND K♠/A♠ en circulación AND se lidera Q♠`
- **Justificación:** Dump de Q♠ cuando hay cobertura de K♠/A♠.

## 🔴 `PENALTY_LIDERAR_PICA_CON_Q_ACTIVA`

- **Patrón asociado:** `liderar_pica_innecesaria`
- **Tipo:** penalización
- **Valor sugerido:** -1.0
- **Condición:** `Q♠ activa AND se lidera pica sin ser la máxima`
- **Justificación:** Liderar picas con Q♠ activa es peligroso.

## 🔴 `PENALTY_LIDERAR_Q_EQUIVOCADO`

- **Patrón asociado:** `liderar_q_espadas_mal_momento`
- **Tipo:** penalización
- **Valor sugerido:** -3.0
- **Condición:** `Se lidera Q♠ cuando soy máxima en picas o baza < 7`
- **Justificación:** Liderar Q♠ sin K♠/A♠ en circulación = auto-13pts.

## 🔴 `PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD`

- **Patrón asociado:** `liderar_quemar_cuando_debe_ceder`
- **Tipo:** penalización
- **Valor sugerido:** -2.0
- **Condición:** `baza ≥ 9 AND se gana baza sin puntos con carta alta de palo seguro`
- **Justificación:** Ganar baza en final de mano fuerza a liderar de nuevo.

## 🔴 `PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE`

- **Patrón asociado:** `seguir_gana_baza_con_puntos`
- **Tipo:** penalización
- **Valor sugerido:** -3.0
- **Condición:** `puntos_mesa > 0 AND se juega carta > ganadora teniendo perdedora`
- **Justificación:** Capturar puntos cuando había alternativa de perder.

## 🟢 `REWARD_QUEMAR_ALTA_SIGUIENDO_PALO`

- **Patrón asociado:** `seguir_no_quema_alta_en_baza_limpia`
- **Tipo:** recompensa
- **Valor sugerido:** +0.5
- **Condición:** `puntos_mesa == 0 AND se juega A/K del palo`
- **Justificación:** Ya cubierto parcialmente. Refuerzo adicional.

## 🟢 `REWARD_QUEMAR_MAXIMA_FORZADA`

- **Patrón asociado:** `seguir_no_quema_maxima_forzada`
- **Tipo:** recompensa
- **Valor sugerido:** +0.3
- **Condición:** `Todas las cartas ganan AND se juega la máxima`
- **Justificación:** Si vas a ganar igual, quema la más alta.

## 🟢 `REWARD_DUMP_Q_SIGUIENDO_PICAS`

- **Patrón asociado:** `seguir_no_suelta_q_con_altas`
- **Tipo:** recompensa
- **Valor sugerido:** +2.0
- **Condición:** `siguiendo ♠ AND se juega Q♠ AND K♠/A♠ en circulación`
- **Justificación:** Soltar Q♠ cuando hay cobertura.

---

## Tabla Resumen

| Reward/Penalty | Tipo | Valor | Patrón |
|---------------|------|-------|--------|
| `REWARD_DESCARTAR_ASES_SEGUROS` | recompensa | +1.0 | `descarte_ases_en_vez_de_corazones` |
| `REWARD_DESCARTAR_CORAZON_BAJO_ROTO` | recompensa | +0.5 | `descarte_no_suelta_corazon_bajo` |
| `REWARD_DESCARTAR_K_A_PICAS_CON_Q_ACTIVA` | recompensa | +2.0 | `descarte_no_suelta_picas_altas` |
| `REWARD_DESCARTAR_DAMA_SEGURO` | recompensa | +5.0 | `descarte_no_suelta_q_espadas` |
| `PENALTY_DESCARTAR_Q_PREMATURO` | penalización | -3.0 | `descarte_q_espadas_prematuro` |
| `PENALTY_LIDERAR_CORAZON_ALTO_TARDIO` | penalización | -1.5 | `liderar_corazon_demasiado_alto` |
| `REWARD_LIDERAR_CORAZON_TARDIO` | recompensa | +1.0 | `liderar_corazon_tardio` |
| `REWARD_QUEMAR_MAXIMA_PALO_SEGURO` | recompensa | +0.5 | `liderar_no_quema_maxima` |
| `REWARD_LIDERAR_Q_DUMP_SEGURO` | recompensa | +3.0 | `liderar_no_suelta_q_espadas` |
| `PENALTY_LIDERAR_PICA_CON_Q_ACTIVA` | penalización | -1.0 | `liderar_pica_innecesaria` |
| `PENALTY_LIDERAR_Q_EQUIVOCADO` | penalización | -3.0 | `liderar_q_espadas_mal_momento` |
| `PENALTY_GANAR_BAZA_TARDIA_SIN_NECESIDAD` | penalización | -2.0 | `liderar_quemar_cuando_debe_ceder` |
| `PENALTY_GANAR_BAZA_CON_PUNTOS_EVITABLE` | penalización | -3.0 | `seguir_gana_baza_con_puntos` |
| `REWARD_QUEMAR_ALTA_SIGUIENDO_PALO` | recompensa | +0.5 | `seguir_no_quema_alta_en_baza_limpia` |
| `REWARD_QUEMAR_MAXIMA_FORZADA` | recompensa | +0.3 | `seguir_no_quema_maxima_forzada` |
| `REWARD_DUMP_Q_SIGUIENDO_PICAS` | recompensa | +2.0 | `seguir_no_suelta_q_con_altas` |