# Análisis interino del periodo v12_campeon (154 partidas) — 2026-07-26

Primer periodo real del sucesor (v12_campeon + FiltroQS). Cerrado y archivado
(`archived-2026-07-26-v12-filtro-bug`); réplica limpia:
`data/partidas_bridge_v12a.jsonl` (153 partidas — la 145 excluida por
etiqueta terminal contaminada). Dos forenses paralelos (filtro + anomalías).

## Señales del modelo (excluyendo partidas corruptas)

| Métrica | v10c (556p) | modolunar (288p) | **v12+FQS (153p)** |
|---|---|---|---|
| pts rel/mano sin lunas | −0.486 | −0.106 | **−0.845 ± 0.28 — mejor del histórico** |
| 4º puestos | 25.5 % | 28.8 % | **20.3 %** |
| 1º / 2º | 23.9/27.9 | 21.9/28.5 | 20.3/**31.4** (perfil conservador) |
| Rating inicio→fin | 2696→2167 | 2208→2091 | **2010→2966 (+956, único periodo positivo)** |
| Lunas nuestras | 0.27 % | 0.53 % | 0 (bug del filtro, corregido) |

Win-rate 19.7 ± 3.2 % — no concluyente a este n; el resto de señales es la
mejor foto real que ha dado un modelo de la casa.

## FiltroQS en producción (forense)

- Fidelidad 99.7 % (770 intervenciones, 5.3 % de decisiones).
- **Clase Q♠-evitable: 15.2 % → 0.7 % — gate (<8 %) batido.**
- Tasa TOTAL de Q comidas sin cambio (22.7 % ≈ baseline): el 85 % nunca fue
  filtrable y el 31.5 % de los vetos de la propia Q solo la posponen
  (auto-comida 28.7 % de las manos con Q). Expectativa del filtro calibrada:
  mata la clase evitable, no la tasa total.
- **Bug de lunas** (corregido 2026-07-26): 48 manos estranguladas, 16 con
  perfil de luna abortada (~20 pts comidos) ≈ 1.3 pts/partida + upside
  perdido. El fix (excepción luna-propia) queda validado por estos datos.

## Bug del bridge encontrado: doble-mesa por /resume a mitad de partida

29 manos corruptas (partidas 145-147) NO son regresión del reveal: /pause
durante partida + /resume con partida en curso → joinDefaultRules() con la
mesa viva → multi-room join de SFS2X → cliente sentado en DOS mesas mezclando
eventos; leaveRoom() salía de la sala equivocada (this.room = lobby). La
validación estructural rechazó toda la basura (reconstructable=0 correcto).
**Corregido en el bridge (commit 840b5b9)**: guard de resume + gameRoom
trackeado aparte + joinDefaultRules se rehúsa con mesa activa. La "violación
de filtro" detectada por el otro forense era esta misma contaminación.

## Estado para el periodo limpio

- Fix de lunas del filtro: desplegado (reiniciar servidor si no se hizo).
- Fixes del bridge: commiteados — cargar con el próximo reinicio del bridge.
- `hearts.db` viva aún contiene el periodo viejo: BORRARLA en el próximo
  reinicio del bridge (la copia archivada está a salvo).
- Pendiente menor: decidir si [test-ready] (opcode 3) se queda o se retira.
