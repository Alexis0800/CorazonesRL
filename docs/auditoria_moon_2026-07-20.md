# Auditoría de datos reales + moon-prob (2026-07-20)

Auditoría senior sobre las partidas reales capturadas por el bridge SFS
(`hearts-sfs-bridge`) para entender por qué v10c **no le gana de forma
consistente a los humanos** (win-rate 23.9 % ≈ azar) y qué palanca lo mejora.

## TL;DR (estado FINAL de la campaña, 2026-07-21 — el doc es un log cronológico; ante contradicción manda la sección más reciente)

- **Modelos moon**: de azar a señal real (propio AUC **0.945** val / rival
  **0.707**), pesos vivos `models/moon_realfull` (= `RUTA_MOON`).
- **Defensa de luna: CERRADA.** El regret elevado en manos de luna es brecha
  estructural de información (BotExperto igual de elevado). El regret vs
  oráculo está saturado como métrica entre jugadores competentes — no usar
  como guía de mejora.
- **RL vs clon humano: REFUTADO 3×** (fine-tune ×2, desde-cero ×1). El techo
  es la fidelidad del clon, no el algoritmo.
- **La fuga real es la OFENSIVA de luna** (§Auditoría de OFENSIVA): en manos
  normales el campeón es MEJOR que los humanos (−3.97 pts/partida); todo el
  déficit es el diferencial de luna (+5.81). Luneamos 0.27 %/mano vs 2.52 %
  cada humano (9.4×); nuestro pase 100 %-defensivo mata nuestras manos de luna
  (verdad del servidor: pase → ×3.6 la luna rival, ×0.9 la nuestra).
- **ModoLunar** (composición gate+BotLunatico, rama `feature/modo-lunar`):
  **EV-neutro exacto vs clon** (+0.2 ± 0.7 pp, A/B pareado 1000 partidas).
  Seguro pero no paga con persecución cruda. Gate pendiente: bridge real (el
  clon no defiende lunas → no mide transferencia).
- **Lección de instrumento**: A/B contra clones estocásticos exige resembrar
  TODOS los RNG por partida (torch incluido) y delta pareado; si no, piso de
  ruido ~±3 pp (≈ los efectos buscados).

## Diagnóstico (datos)

Sobre `D:/Github/Personal/hearts-sfs-bridge/data/hearts.db` (556 partidas
completas, 5360 manos, 10–20 jul 2026):

| Métrica | Modelo | Azar |
|---|---|---|
| Win-rate (1er puesto) | 23.9 % | 25 % |
| Top-2 | 51.8 % | 50 % |
| Puesto 1/2/3/4 | 133 / 155 / 126 / 142 | plano |
| Puntos/mano recibidos | 6.92 | 6.5 |
| **Lunas en contra** | **368** (en 54 % de las partidas) | — |
| Lunas a favor | 13 | — |

Juega a nivel de **azar de colocación**. Fuga dominante: te lunean en más de la
mitad de las partidas (+26 de golpe cada vez) → esa es la varianza que se ve
como "oscilación". Nota: el campo `rating` de la DB está sin decodificar
(0–4402, salta salvajemente); **la señal confiable es el win-rate, no el
rating**.

## Fase 0 — Desbloqueo de datos (hecho)

`reconstruct-partidas.js` daba 0 manos: los `.log` recientes (sesiones autoplay)
no traen el volcado crudo `cmd=g opcode=…` que ese script necesita. Pero los
eventos estructurados (`handDealt`/`trickEnd`/`handRevealed`/`pass`/`handFinal`)
sí están en los `session-*.jsonl` — es de ahí que se construyó la DB.

- **Nuevo**: `hearts-sfs-bridge/src/export-reconstructed-from-jsonl.js` — reusa el
  parser de `ingest-to-sqlite.js` y emite `logs/reconstructed/*.reconstructed.jsonl`.
- Importado con el script Python vigente sin modificarlo →
  **`data/partidas_bridge_full.jsonl`: 558 partidas / 4168 manos reconstruibles
  al 100 %** (antes ~483; cifra ya LIMPIA tras el fix de §Verificación).
  *(ACTUALIZADO 2026-07-21: el archivo se regeneró tras recuperar los reveals →
  556 partidas / 4910 manos / 385 lunas; ver §Recuperación.)*
- Positivos de luna: **371 rival + 1 propia** (vs 11 antes; = el conteo del
  servidor, tras quitar 97 falsos). Las lunas propias reconstruibles son ~0 → el
  modelo "propio" debe seguir con datos simulados
  (`generar_dataset_moon_simulado.py`, ya existe).

Regenerar tras nuevas capturas:
```
node src/export-reconstructed-from-jsonl.js logs logs/reconstructed          # en el bridge
python scripts/importar_sesiones_bridge_reconstruidas.py \                    # en este repo
    --dir D:/Github/Personal/hearts-sfs-bridge/logs/reconstructed \
    --out data/partidas_bridge_full.jsonl
```

## Fase 2 — Reentrenar modelos moon (hecho)

`python scripts/entrenar_moon_prob.py --partidas data/partidas_bridge_full.jsonl --out-dir models/moon_realfull`

| Modelo | AUC antes | AUC ahora (limpio) | Recientes (overfit) |
|---|---|---|---|
| propio | 0.44–0.46 | **0.945** | 0.958 ✅ |
| rival  | 0.578     | **0.707** | 0.750 ✅ |

Guardados en `models/moon_realfull/` — que LUEGO pasó a ser el canónico vivo
(`RUTA_MOON`); `models/moon/` quedó como legado que nada lee.

## Fase 1 — Regret real del campeón (hecho)

`pimc_regret_real.py --modelo models/produccion/v10c_campeon --partidas data/partidas_bridge_full.jsonl --max-decisiones 4000 --rollouts 20` (volcado en `data/regret_v10c_full.jsonl`):

Global (limpio): regret medio **1.002**, mediana 0, **59.9 % óptimo**,
≥3 pts = 10.5 %, ≥10 pts = 1.5 %. El campeón decide MEJOR que el BotExperto en
global (1.002 vs 1.137).

**Cruce con etiquetas de luna, campeón vs BotExperto** (la prueba decisiva):

| Tipo de mano | n | campeón regret | experto regret | campeón ≥10 pts |
|---|---|---|---|---|
| Normal | 3675 | 0.92 | 1.05 | 1.1 % |
| **Luna rival** | 325 | **1.93** | **2.07** | **6.5 %** |

El regret en manos de luna está ~2× elevado — **pero por igual en el campeón y en
un heurístico fuerte** (1.93 vs 2.07). No es un defecto de política: es una
**brecha estructural de información** (el oráculo ve las manos ocultas; la defensa
de luna depende de info que no está en el estado público). El campeón ya juega
las manos de luna a nivel heurístico-fuerte. → El regret NO puede validar Fase 3.

## Verificación adversarial (errores encontrados y corregidos)

Antes de justificar Fase 3 se re-auditó todo buscando errores.

1. **BUG encontrado y corregido — manos fantasma.** Mi exportador sumaba las
   cartas de la ÚLTIMA mano de cada partida, que el servidor marca `[0,0,0,0]`
   (no la contabiliza — el marcador final NO se mueve; verificado 5360/5360
   contra `scores_after`). Eran 490 manos no-contabilizadas (489 = última mano
   de su partida), y mi recálculo les fabricaba puntos → **97 pozos falsos**.
   Fix en `export-reconstructed-from-jsonl.js`: descartar toda mano con
   `sum(puntos)==0` (una mano real siempre suma 26 o 78, nunca 0). Todas las
   cifras de este doc son POST-fix.
2. **El regret sobre-estima la fuga recuperable.** El oráculo usa info completa
   (ve las manos reales), así que en manos de luna siempre halla la defensa
   perfecta. La referencia BotExperto (arriba) muestra que la brecha en luna es
   estructural, no de política.
3. **Sano:** el diagnóstico base (win-rate 23.9 %, 368 lunas en contra) usa
   campos del SERVIDOR, no mi recálculo → no contaminado.

Gaps NO cubiertos (pendientes): el **pase** no se audita (posible facilitador de
lunas; el oráculo de regret no lo evalúa); el regret cubre las primeras ~54
partidas hasta agotar 4000 decisiones (la comparación luna-vs-normal es
intra-muestra, así que la comparación relativa es válida, pero el absoluto no
representa las 558).

## Fase 3 — Plan (gated, NO ejecutado; decisión del operador)

**Justificación honesta:** hay UNA razón nueva y real para intentarlo — el modelo
rival dejó de ser ruido (0.578→0.707), así que hay una señal pública de riesgo de
luna que el entrenamiento de v10c NUNCA tuvo (hoy `[188]` usa la heurística de
azar). PERO no es un slam-dunk: el campeón ya juega la defensa de luna a nivel
heurístico-fuerte, la parte recuperable por info pública es limitada (AUC 0.707,
no 1.0), y el intento previo de mover el regret falló por diseño (el regret no
mide esto). **Riesgo real de resultado nulo.**

Métrica de éxito correcta: **tasa de lunas-en-contra y win-rate en evaluación**
(no regret). Orden por ROI:

1. **Experimento barato primero (sin RL)**: eval de simulación del campeón vs
   `BotLunatico` con el modelo rival conectado vs desconectado, midiendo
   lunas-en-contra. Si no baja, PARAR — no gastar cómputo de RL.
2. Si baja: **conectar `EstimadorMoonProb` rival (models/moon_realfull) en
   `corazones_rllib.py`** (`[188]` en entrenamiento) + **shaping de recompensa
   por defensa de luna**, y **fine-tune** con `BotLunatico` reforzado (lo único
   que ya movió métricas: regret 1.261→1.227).
3. **Gate final**: win-rate y lunas-en-contra en partidas reales held-out (como
   hizo `models/produccion/README.md`), NO regret.

## Veredicto de datos

- Pozo-rival y liderazgo: **suficientes** (467 positivos / 4635 manos sin
  explotar). No hace falta capturar más.
- Pozo-propio: escaso (~0 reconstruibles) → usar dataset simulado, no capturar
  más real por eso. *(ACTUALIZADO 2026-07-21: tras recuperar los reveals hay
  14 lunas propias reconstruibles — siguen siendo pocas para entrenar, el
  dataset simulado sigue vigente para "propio"; ver §Recuperación.)*

## Implementación Fase 3 (construida y validada 2026-07-20)

Pipeline completo montado y probado end-to-end (no el fine-tune largo en sí):

1. **Encoder alineado** (`src/captura/replay.py`): `_TrackerTactico` replica el
   bookkeeping del env; `ejemplos_de_partida(..., seats_de="rivales")` emite la
   obs COMPLETA (228) por perspectiva. Test: `tests/captura/test_replay_obs_completa.py`.
2. **Bot de imitación humana** (`scripts/entrenar_bc_humano.py` → `models/humano_bc/`):
   `HeartsActionMaskModel` BC sobre los 3 asientos humanos. **val top-1 66%.**
3. **Cableado al pool** (`src/rllib/opponent_pool.py`, `scripts/train_rllib.py`):
   flags `--humano-bc` / `--prob-humano`. Cubre una fracción de los slots de
   oponente "duro". Cloudpicklable (Ray). **Contraste medido — campeón vs bot
   humano: 46% win-rate** (vs 92% bots simples, 52% mixto, 24% humanos reales):
   el oponente más duro y el mejor proxy de humano.
4. **Modelo rival reentrenado conectado** (`--moon-dir models/moon_realfull`,
   `src/rllib/config.py` + `corazones_rllib.py`): `[188]` deja de ser la
   heurística de azar. **Shaping de defensa de luna: DIFERIDO** — la verificación
   mostró que la brecha es estructural (experto empata al campeón); un cambio de
   reward mal hecho desestabiliza horas de entrenamiento. Es un experimento
   opcional posterior, no el lever principal.
5. **Fine-tune**: pipeline validado (smoke corrió, reanudó del campeón, guardó
   snapshot). Comando del fine-tune completo:

```bash
python scripts/train_rllib.py --total-steps 30000000 --workers 4 --gpus 1 \
    --obs-dim 228 --con-pase \
    --humano-bc models/humano_bc/pesos.npz --prob-humano 0.5 \
    --moon-dir models/moon_realfull --pool-diverso --ancla-experto \
    --resume models/produccion/v10c_campeon --output-dir models/v10d_humano
```

Evaluación before/after (métrica de éxito, NO regret):

```bash
python scripts/evaluar_vs_humano_bc.py --modelo <snapshot> --partidas 300
python scripts/pimc_regret_real.py --modelo <snapshot> \
    --partidas data/partidas_bridge_full.jsonl --max-decisiones 4000   # no-regresión
```

Greenlight si el win-rate vs bot humano sube desde 0.46 sin regresión de regret.

**Consistencia (auditada 2026-07-20):** todo el vector es 228 (DIM_V12) —
campeón, bot humano, env, pool, config. NO usar 332 (v13): el fine-tune no puede
cambiar el input del campeón, y para oponentes humanos la memoria de pase v13
`[228:332]` es desconocida (el puente no revela el pase rival) → iría en cero. El
modelo moon de `[187:188]` debe ser el MISMO en el BC humano y en el env: por eso
`entrenar_bc_humano.py --moon-dir` (default `models/moon_realfull`) y el fine-tune
`--moon-dir models/moon_realfull` coinciden (bug corregido: antes el BC usaba
`models/moon`).

**Fine-tune vs desde cero:** empezar con **fine-tune** del campeón — su
competencia (juego legal, Q♠, marcador) es independiente del oponente y transfiere;
solo hay que adaptar la capa de explotación al meta humano. Más largo que el +5M
previo (el salto bots→humanos es mayor): ~10M pasos, monitoreando win-rate vs bot
humano. **Fallback a desde-cero** solo si ese win-rate se estanca cerca de 0.46
(señal de que la burbuja es un atractor fuerte que el fine-tune no escapa).

## Experimento v10e — fine-tune corregido (2026-07-20 noche): REFUTADO

Se corrió el fine-tune corto con TODOS los fixes: clon humano v2 (BC con máscara
legal, val top-1 66.1%→**71.6%**; estocástico T=1.0), mesas 70% humanas
(exposición real ~47% de asientos), `--progress-fino` (sin trampa de fase 4),
LR 5e-5 plano, entropy 0.01, moon_realfull, rail de elite por `win_rate_vs_humano`.

Baseline re-calibrado: campeón vs clon-v2 = **0.480** (300 partidas). Gate: ≥0.54.

Resultado (2M pasos, 25.01M→27M): rail 0.393 → 0.453 → 0.440; snapshot final
**0.377** (300 partidas). **Reward de entrenamiento cayó 0.694→~0.40 al entrar en
la distribución humana y quedó PLANO los 2M pasos** — la optimización nunca
despegó. No es lentitud: no hay gradiente útil a LR conservador sobre el campeón
convergido. **La vía fine-tune queda refutada dos veces** (v10d por bug de
progress; v10e limpiamente).

## v11 — desde-cero humanizado (lanzado, en curso)

Conclusión operativa: reentrenar con política PLÁSTICA. No desde pesos
aleatorios: **init BC-PIMC** (`models/produccion/bc_base_con_pase.pkl`) + pool
humanizado desde progress 0. Config (wrapper `scripts/reanudar_finetune.sh`):
30M pasos, mesas 50% humanas (clon v2, T=1.0), prob-humano 0.5, pool-diverso +
ancla-experto, LR 1e-4→5e-5, entropy 0.03, moon_realfull, rail de elite por
win_rate_vs_humano cada ~1M pasos. Output: `models/v11_humano_scratch/`.
Duración estimada 4–8 h CPU (10 workers). **Gate de éxito: elite con
win_rate_vs_humano ≥ 0.54** (supera al campeón 0.480 por >2 SE) y luego
validación en partidas reales held-out.

## Veredicto v11 (2026-07-21): tampoco supera al campeón — ambas vías agotadas

v11 completó 30M pasos (5h48, sin crashes). Curva `win_rate_vs_humano`: 0.18 →
pico **0.407** (25.6M) → meseta ~0.37 los últimos 8M, mientras `vs_experto`
seguía subiendo (0.32→0.46): la misma firma de "mejora general sin mejora
humana", pese a 50 % de mesas humanas. Evaluación sólida del mejor elite
(25.6M, 300 partidas): **0.337 vs clon** (campeón 0.480) y **0.150 vs 3
campeones head-to-head** (paridad 0.25) — el v11 es sencillamente un jugador
más débil; 30M desde init BC no alcanzan el linaje del campeón (~25M + BC +
varios fine-tunes).

**Conclusión de la campaña RL (honesta):**
1. El campeón v10c sigue siendo el mejor modelo. Producción intacta.
2. Dos vías refutadas limpiamente: fine-tune humanizado (sin gradiente útil
   sobre política convergida) y desde-cero humanizado (no alcanza la fuerza
   general; el aprendizaje extraíble del clon se satura en ~0.37-0.40).
3. Lección central: entrenar contra el clon (71.6 % fidelidad) desarrolla
   fuerza general, no ventaja específica anti-humana. El techo es la fidelidad
   del clon, no el algoritmo de RL.
4. Camino recomendado: **volante de datos** — seguir capturando partidas
   reales (el bridge), reentrenar el clon al crecer el dataset (~10k+ manos),
   y re-evaluar RL entonces. ~~El pase sigue sin auditarse (palanca virgen)~~
   (auditado después, mismo día: §Auditoría del PASE y §Auditoría de OFENSIVA).
5. Subproducto útil: el elite v11 (estilo distinto) puede servir como oponente
   de diversidad en pools futuros.

## Auditoría del PASE y del regret (2026-07-21, offline, cero cómputo RL)

**El pase NO es una fuga — descartado con datos:**
- Pasar "peligro" (Q♠ o corazón alto) al receptor **no** aumenta que él haga
  luna: 4.7 % vs 4.9 % con pase seguro. Y nuestros puntos son mejores al pasar
  peligro (6.36 vs 7.13). La hipótesis "alimentamos la luna por el pase" es falsa.
- La política de Q♠ del modelo es **de libro**, monotónica en protección de
  picas: pasa Q♠ el 100 % con 1-2 picas, 93.9 % con 3, 68.8 % con 4, 29.5 % con
  5, 9.7 % con 6. Quedársela sin protección es catastrófico (12.64 pts medios,
  64 % de manos ≥13) pero **solo ocurre 14 veces de 344** (4 %).
- Conclusión: construir un "modelo de pase" no rendiría. ~~Palanca cerrada.~~
  **⚠ CORREGIDO en §Auditoría de OFENSIVA**: esta sección hizo la pregunta
  DEFENSIVA ("¿nuestro pase alimenta la luna rival?" → no). La pregunta
  ofensiva ("¿nuestro pase mata NUESTRAS manos de luna?") da SÍ — la palanca
  del pase sigue abierta, pero por el lado ofensivo.

**El regret contra oráculo info-completa ya NO discrimina (hallazgo metodológico):**
- Concentración: 5.2 % de decisiones acumulan ~la mitad del regret (1907 pts de
  4000 decisiones); el regret pico está en bazas 2-8 y crece monótono con la
  libertad de elección (2 legales → 0.60; 7-8+ → 1.42-1.53). O sea: descartes y
  liderazgos discrecionales de midgame.
- PERO el campeón **gana al BotExperto en todos esos segmentos** (brecha −0.15 a
  −0.23): no es defecto suyo, es dificultad estructural del segmento.
- El aparente "+3.38 peor que el experto en sus peores decisiones" era **sesgo de
  selección**: invertida la selección sale simétrico (experto 9.31 / campeón
  4.75). Comparación pareada limpia sobre las 4000: campeón mejor 18.7 %,
  experto mejor 19.1 %, empate 62.2 % → **indistinguibles por decisión**.
- Implicación: el regret vs oráculo omnisciente está saturado como métrica entre
  jugadores competentes. Explica por qué toda intervención guiada por regret
  falló. No usarlo más como guía de mejora.

**⚠ RETRACTADO — sesgo de muestra en `data/partidas_bridge_full.jsonl`.**
Una primera lectura concluyó que las derrotas eran por "desastres puntuales" y
que comíamos MENOS puntos que los rivales (6.53 vs 8.04/mano, 27.4 % vs 30.8 %
de manos ≥13). **Es un artefacto**: el importador descarta las manos NO
reconstruibles, y esas están sesgadas a nuestro favor. (Mecanismo CORREGIDO
2026-07-21: NO es el remate — el remate reconstruye al 100 % vía `handRevealed`.
Lo que falla son las manos SIN remate terminadas TEMPRANO, bazas 5–11: cuando
los 26 puntos ya cayeron, el servidor corta la mano y las cartas restantes de
los rivales nunca cruzan el cable — 739 manos, imposibles de reconstruir en
general. Las manos donde los puntos caen rápido sobre nosotros —y NUESTRAS
lunas, 12/13— son exactamente las que cortan temprano.)
Contraste decisivo sobre las MISMAS 556 partidas:

| Fuente | win-rate | score final nuestro | rivales |
|---|---|---|---|
| Servidor (`hearts.db`, verdad) | 23.9 % | **66.7** | 65.5 |
| JSONL importado (sesgado) | 36.4 % | 48.7 | 60.0 |

**Verdad**: contra estos humanos el modelo es **exactamente promedio** (66.7 vs
65.5, marginalmente peor). No es propenso al desastre ni sistemáticamente malo:
es del montón. Por eso se descartó la idea de un reward con aversión al riesgo:
atacaría un problema inexistente, y además R_terminal-por-puesto YA es el
objetivo correcto y el PBRS es policy-invariante por construcción (no puede ser
la causa).

**Regla para futuros análisis:** cualquier métrica de RESULTADO se calcula desde
`hearts.db` (servidor). El JSONL importado sirve para re-jugar decisiones
(regret, BC), no para medir outcomes. Las conclusiones pareadas dentro del
subset (campeón vs experto en las MISMAS decisiones) siguen siendo válidas; las
de nivel-partida no. Nota: el clon humano-BC se entrenó sobre este subset
sesgado — otra razón por la que su fidelidad (71.6 %) tiene techo.

## Auditoría de OFENSIVA de luna (2026-07-21, sobre `hearts.db`, cero cómputo RL)

Re-encuadre. Toda la campaña anterior midió la luna como **daño recibido**
(defensa) y la cerró como brecha estructural. Correcto pero es la palanca
equivocada: cuando un rival lunea, los 3 perdedores comen 26 por igual, así que
defender solo te empata con los otros dos. La palanca **asimétrica** —la que
nadie midió— es ser TÚ el que lunea.

**Descomposición del resultado (549 partidas, verdad del servidor).** La
aritmética cierra exacta (67.6 vs 66.4):

| Componente | n | pts relativos a media rival |
|---|---|---|
| Manos normales | 4489 | **−3.97 / partida (a tu favor)** |
| Lunas rivales | 368 | **+5.81 / partida (en contra)** |
| Lunas tuyas | 13 | −0.62 |
| Neto | | +1.22 |

**El campeón juega MEJOR que los humanos en manos normales.** Todo el déficit y
más viene de un único diferencial: luneas 0.27 % de las manos, cada rival 2.52 %
(**9.4× por jugador**). ⚠ Corrección de revisión: la primera versión de este
apartado decía "en simulación sí lunea 2.5–5.5 %, contra humanos no entra" —
**falso, era comparar unidades distintas**. El `moon_rate` de `eval_bots` es
por PARTIDA (`shooting_moon` es un flag de episodio); por mano son ~0.26–0.56 %,
lo MISMO que contra humanos (0.27 %). El campeón casi no lunea en NINGÚN
contexto — la ofensiva de luna nunca se desarrolló (sus rivales de entrenamiento
casi no luneaban, no hubo presión selectiva). Firma confirmada por fuerza de
mano post-pase (heurística de fuerza, tendencia > magnitud): con cartas flojas
le ganas al rival (7.45 vs 7.66), con cartas fuertes pierdes (10.85 vs 6.47;
descontando las lunas rivales de ese bucket sigue ~8.7 vs 6.47). Es el patrón
exacto de una política sin modo ofensivo.

**Causa aislada = el PASE (la pregunta que la §Auditoría del PASE cerró mal).**
Ese apartado probó "¿nuestro pase alimenta la luna rival?" (no). La pregunta
correcta era "¿nuestro pase mata nuestras propias manos de luna?" — y la
respuesta es SÍ. Validado con `models/moon_realfull/propio.pt` (AUC **0.958**
sobre las 372 lunas reales de `partidas_bridge_full`, mismo modelo sobre los 4
asientos → sin sesgo de calibración):

| Situación | P(luna) nuestra | P(luna) rival |
|---|---|---|
| SIN pase | 1.6 % manos P≥0.10 | 3.2 % |
| CON pase | 1.6 % | **10.3 %** |

Los rivales usan el pase **ofensivamente** (triplican su P de luna, 3.2→10.3 %);
nosotros lo usamos puramente defensivo. Medido pre/post sobre nuestro propio
asiento (n=3291): nuestro pase **destruye 57 manos de luna potenciales** (P≥0.10
→ <0.10) y solo crea 34 (z=2.41, p≈0.016), delta medio −0.0033. Es el pase
100 %-defensivo de Q♠ que la propia auditoría documentó ("de libro, monotónico
en protección de picas") — óptimo para no comer, pésimo para atacar.

**Confirmación decisiva desde el SERVIDOR (sin modelo, sin sesgo de muestra)** —
lunas completadas por tipo de mano en `hearts.db` (4870 manos):

| | lunas rival (por jugador) | lunas nuestras |
|---|---|---|
| CON pase (n=3850) | **2.97 %** | 0.26 % |
| SIN pase / hold (n=1020) | 0.82 % | 0.29 % |

El pase **multiplica ×3.6** la tasa de luna de los rivales; a nosotros nos da
×0.9 (nada). Y la brecha tiene DOS componentes, no una: incluso en manos hold
(cartas iid, sin pase de por medio) los rivales convierten 2.8× más que nosotros
(0.82 vs 0.29 %) → además del pase falta la **persecución en-mano** (el modo
luna durante la baza). Las dos palancas de la recomendación (#1 y #2) atacan
una componente cada una; ninguna basta sola.

Caveats de la parte con modelo (la parte servidor no los tiene):

- **Sesgo de reconstruibilidad, ahora cuantificado**: las manos de luna rival
  son reconstruibles al **100 %** (368/368), las nuestras al **8 %** (1/13),
  las normales al 84 %. El residuo sin-pase del muestreo JSONL (1.6 vs 3.2 %)
  viene de ahí, y las tasas de conversión "top-5 % nuestras → 2.2 %" son un
  SUELO: 12 de nuestras 13 lunas reales están excluidas de la muestra.
- El AUC 0.958 de la validación es parcialmente in-sample (`propio.pt` se
  entrenó sobre este mismo JSONL, 4 perspectivas → las 372 lunas rivales SON
  sus positivos de entrenamiento); el número honesto es el **0.945 de val**
  del entrenamiento. Sigue siendo señal real, no ruido.

**Recomendación (orden de coste, sin RL primero):**

1. **Pase ofensivo condicional** (barato, en inferencia): si P(luna) pre-pase
   supera un umbral → pase CONSTRUCTIVO (conservar controles altos, descartar
   bajas para vaciar un palo), no solo "no destruir": conservar solo recupera
   ~2.3 % de oportunidad; los rivales llegan a ~10 % porque el pase CREA la
   mano (3.2→10.3 %). Piezas: `propio.pt` (gate) + `pase_heuristico` estilo
   lunático.
2. **Modo luna compuesto durante la mano**: si el pozo sigue vivo → política
   `BotLunatico`; si se rompe (gate duro `_alguien_mas_tiene_puntos`) → vuelve
   al campeón. Ataca la componente de persecución (0.82 vs 0.29 % en hold).
3. **Solo si 1–2 mueven la aguja**, RL con `BotLunatico` en el pool y un
   currículum que no castigue acumular puntos intra-mano.

⚠ **#1 y #2 se prueban JUNTOS, nunca #1 solo**: conservar cartas altas sin modo
de persecución empeora — la evidencia de buckets ya muestra que el campeón come
10.85 pts/mano con manos fuertes. Y el gate de evaluación tiene un límite
conocido: el clon humano casi nunca ENFRENTÓ lunas nuestras (13 en todo el
dataset), así que su defensa anti-luna es no-representativa → una eval optimista
vs clon NO basta; el greenlight final son partidas reales del bridge
(lunas-a-favor y win-rate). También medir el coste de intentos fallidos
(pts comidos cuando P≥umbral y no se corona).

NO hacer: más fine-tune / desde-cero vs el clon (refutado 2×, techo = fidelidad
del clon). Este agujero es de POLÍTICA sobre información completamente observable
(tu propia mano), no de información — no lo necesita.

Scripts de esta auditoría: `scratchpad/audit2.py` (descomposición), `paso1.py`
(validación + control 4-asientos), `paso2.py` (control sin-pase + efecto pase).

## ModoLunar — implementación y veredicto vs clon (2026-07-21, rama feature/modo-lunar)

Se implementó la recomendación #1+#2 como composición (`src/agentes/modo_lunar.py`):
gate aprendido (`propio.pt`) → pase constructivo de BotLunatico → persecución
`_jugar_moon` → abort duro (gate) + abort blando (P mid-mano con historial
sintético). A/B contra 3 clones humanos: `scripts/evaluar_modo_lunar.py`.

**Lección metodológica (la más valiosa de la campaña):** las primeras 3000
partidas de A/B fueron RUIDO. Los clones muestrean de `torch.multinomial`
(RNG global, nunca resembrado por partida) → los brazos jamás estuvieron
pareados y el piso de ruido (~±3 pp) era del orden de todos los efectos
"medidos" — el +8.7 pp de un barrido y los −2/−4 pp de sus "confirmaciones"
por igual. Lo destapó un diagnóstico imposible (9 intervenciones "moviendo"
−3.1 pp). Fix: `torch.manual_seed` por partida + estadística pareada → las
partidas sin intervención son idénticas bit a bit y el SE del delta cae a
±0.7 pp. **Regla para futuros A/B con clones estocásticos: resembrar TODOS
los RNG por partida y reportar el delta pareado, nunca comparar brazos
sueltos.**

**Veredicto (1000 partidas pareadas, semillas vírgenes, p.10/j.30/abort.10):**

- Δwin_rate **+0.002 ± 0.007** (IC95 ≈ [−1.2, +1.6 pp]) — EV-NEUTRO exacto.
- El mecanismo funciona: lunas/partida 0.020→0.067 (×3.4), conversión 35.7 %,
  112 manos comprometidas. Pero 72 fallos × 18.4 pts anulan 40 lunas × 26.
- La composición es SEGURA (no degrada al campeón) pero no paga con la
  política de persecución cruda de BotLunatico a estos umbrales.

**Por qué esto NO cierra la palanca:** el clon no defiende lunas (casi no las
vio en su dataset) — no puede medir la transferencia a humanos reales. Contra
defensores reales la conversión baja pero los fallos se abaratan (el defensor
corta TEMPRANO → abort barato; hoy los fallos son caros porque mueren tarde
"solos"). El signo real es incierto y solo lo responde el bridge.

**Siguiente paso recomendado:** cablear `ModoLunar` al recomendador/autoplay
tras un flag (es EV-neutro vs clon → riesgo bajo) y medir el gate REAL del
doc: lunas-a-favor y win-rate en partidas del bridge. Mejoras de política si
el real da señal: persecución guiada por PIMC (lenta pero válida en el
copiloto) y compromiso sensible al marcador (lunear más al ir perdiendo).

## Recuperación de las manos no-reconstruibles (2026-07-21) — sesgo de muestra ELIMINADO

El corte-temprano NO perdía los datos: el wire manda el reveal (`subtype=10`
con los `w[]` de los 4 asientos) en TODO fin de mano, y los eventos `rawOpcode`
del jsonl conservan el payload. `session-runner` lo descartaba por un bug doble
(solo lo grababa en la rama de remate + off-by-N en el cross-check cuando una
carta se jugaba entre el reveal y el cierre). Ticket con el fix en vivo:
`hearts-sfs-bridge/docs/ISSUE-reveal-descartado.md`.

Mitigación offline aplicada (`export-reconstructed-from-jsonl.js` →
`recuperarRevealDeRaw`, re-parsea rawOpcode con el mismo `GameStateTracker` +
validación estructural; el importador Python valida puntos recomputados ==
servidor):

| | antes | después |
|---|---|---|
| manos reconstruibles | 4168 (85 %) | **4910 (100 % de las importadas)** |
| lunas propias | 1 | **14** |
| lunas totales | 372 | 385 |

`data/partidas_bridge_full.jsonl` regenerado 2026-07-21. Implicaciones:
- La regla "outcomes solo desde hearts.db" SIGUE vigente (la DB es la verdad),
  pero el sesgo de reconstruibilidad del JSONL quedó prácticamente eliminado.
- **El clon humano-BC se entrenó sobre el subset sesgado (85 %)** → reentrenarlo
  sobre el dataset completo es ahora el paso más barato del volante de datos
  (era su techo declarado de fidelidad).
- El residuo "sin-pase 1.6 vs 3.2 %" de §OFENSIVA debería re-medirse sobre el
  dataset completo (era artefacto del sesgo).
- ⚠ Nota: 17/589 sesiones sin `rawOpcode` siguen sin recuperarse; y
  `hearts.db` mantiene el flag `reconstructable` viejo (re-ingestar si se
  necesita alineado).
- **Moon models re-entrenados sobre el dataset completo: SIN ganancia**
  (propio 0.947 vs 0.945, rival 0.703 vs 0.707 — dentro del ruido; los
  positivos solo subieron +13). Los pesos vivos `models/moon_realfull` se
  mantienen sin cambios (no tocar la distribución de entrada del campeón por
  nada); el dir del experimento se borró. La ganancia del dataset completo va
  al CLON humano-BC, no a los moon models.
- **Clon humano-BC v3 reentrenado y PROMOVIDO** (2026-07-21,
  `models/humano_bc/pesos.npz`; el v2 quedó como `pesos_v2_sesgado.npz`):
  val top-1 **0.706 sobre la distribución completa** (v2: 0.716 sobre el
  subset sesgado — números NO comparables, distinta dificultad de val). El
  intento de comparación head-to-head sobre el mismo val dio v2 0.743 pero es
  **fuga de entrenamiento** (v2 entrenó sobre esas mismas partidas; el archivo
  viejo ya no existe para reproducir su split). Promoción por principio: v3 es
  el único entrenado sobre la distribución sin sesgo (incluye las manos de
  puntos-rápidos que v2 jamás vio). ⚠ Todos los baselines históricos "vs clon"
  (0.46–0.52) son contra v2.
- **Baseline de referencia vs clon v3 (A/B pareado, 600 partidas, semillas
  40000+): campeón 0.468** — v3 es un oponente más duro que v2 (0.499 pareado
  vs v2), consistente con que v3 sí conoce las manos de puntos-rápidos.
- **ModoLunar re-testeado vs v3: sigue EV-neutro** (Δwin −0.7 ± 1.0 pp;
  lunas/partida 0.030→0.102, conversión 51.8 %, fallos 18.8 pts). El veredicto
  de la campaña se sostiene contra el defensor mejor informado: seguro,
  triplica las lunas, no mueve el win-rate — la decisión de desplegarlo al
  bridge (gate real) sigue siendo del operador, con riesgo acotado.

## Recalibración sin sesgo + compromiso dinámico (2026-07-21/22)

Con el dataset completo (4910 manos) se recalibró la ofensiva:

- **Residuo sin-pase: desaparecido** (yo 2.6 % = rival 2.6 % en manos hold) —
  la tesis del pase ofensivo queda sobre datos limpios (con pase: rival 9.1 %
  vs yo 2.1 % de manos con P≥0.10).
- **Conversión rival real por bucket de P**: 11 % en [0.10–0.15), 16 % en
  [0.15–0.30), **37 % en [0.30–0.50), 62 % en 0.50+** → umbral de compromiso
  0.30 validado; por debajo es EV-negativo claro.
- **Hallazgo clave: 13/14 lunas reales del campeón nacieron con P~0.01 al
  inicio de mano** (mediana 0.008; el gate estático habría comprometido 1/14).
  Las lunas reales son OPORTUNISTAS de mitad de mano.

**Compromiso dinámico** (implementado): la señal del abort (P mid-mano) también
ENTRA si P sube sobre el umbral; nunca tras un abort. A/B pareado vs clon v3:

| Lote (semillas) | n | Δwin pareado | mid-mano | conversión |
|---|---|---|---|---|
| estático (40000) | 600 | −0.7 ± 1.0 pp | — | 51.8 % |
| dinámico (50000) | 600 | **+1.5 ± 1.2 pp** | 71/146 | 47.3 % |
| dinámico (60000) | 1000 | **+1.1 ± 0.9 pp** | 94/231 | 45.9 % |

| dinámico (70000) | 1400 | +0.5 ± 0.8 pp | 121/333 | 44.4 % |

**VEREDICTO (regla pre-declarada, pool de 3 lotes, n=3000): Δwin = +0.90 ±
0.53 pp (1.7σ, p≈0.09).** Tres lotes positivos (+1.5/+1.1/+0.5). Lectura
honesta: el compromiso dinámico es probablemente un efecto real pero PEQUEÑO
(~+1 pp) — mejor que el estático (−0.7), nunca negativo en 3000 partidas,
no concluyente al estándar 2σ. lunas/partida ~0.11–0.13 vs baseline
0.027–0.038 (×3–4) con conversión estable 44–47 %.

Implicación operativa: la config completa (pase ofensivo + compromiso
dinámico + aborts) es como mínimo neutra y probablemente +1 pp vs clon —
**riesgo acotado para el gate real del bridge** (que además ya captura data
completa tras el fix en vivo del ISSUE). El coste de fallos sigue en ~18 pts:
la palanca restante con efecto potencialmente mayor es la persecución-BC de
los 385 luneadores humanos.

**Persecución-BC (probada 2026-07-22): EMPATA con la heurística.** Se entrenó
`models/luna_bc` con las jugadas de los 385 luneadores humanos (replay
`seats_de="luna"`, val top-1 0.478 con sobreajuste esperado en ~5k ejemplos) y
se cableó como `politica_persecucion` opcional de ModoLunar. A/B pareado (600
partidas, semillas 100000): **Δwin +1.5 ± 1.2 pp — igual que la heurística**,
con el trade: conversión 37.8 % (vs 44-47 %) pero fallos más baratos (15.4 vs
~18 pts). Neto indistinguible. **El programa offline convergió en ~+1 pp**: el
cuello ya no es la política de persecución sino la economía fundamental vs
este clon. Config recomendada para el bridge: dinámico + heurística (la de
3000 partidas de evidencia y cero dependencias extra); el BC queda disponible
tras `--luna-bc`.

## Despliegue al gate real (2026-07-22) — LISTO

- `ModoLunar` cableado a producción: `Recomendador(modo_lunar=True)` intercepta
  pase (nunca en manos hold) y jugada; `None` → campeón como siempre. Flag:
  **`python scripts/servidor_inferencia.py --modelo models/produccion/v10c_campeon --modo-lunar`**
  (OFF por default). Stats del modo en `/estado`. Config desplegada: la de
  3000 partidas de evidencia (pase .10 / juego .30 dinámico / abort .10,
  persecución heurística).
- Fix en vivo del reveal aplicado y commiteado en el bridge (ISSUE resuelto):
  la data del periodo nuevo llega 100 % reconstruible de fábrica.
- **Periodo de medición limpio**: archivado pre-lanzamiento —
  `logs/archivo_2026-07-22/` (servidor), bridge
  `logs/archived-2026-07-22-pre-modolunar/` (sesiones + reconstructed) y
  `data/hearts-archived-2026-07-22-pre-modolunar.db` (556 partidas, WAL
  checkpointeado). `hearts.db` nuevo se regenera al re-ingestar las sesiones
  del periodo ModoLunar.
- **Métricas del gate** (leer de la DB nueva tras ~150–300 partidas):
  lunas-a-favor/mano (baseline 0.27 %), pts en manos comprometidas
  (`[modo-lunar]` en el log del servidor marca compromisos), win-rate
  (baseline 23.9 %). Éxito = economía por mano positiva:
  `lunas×26 > pts extra comidos en fallos`.

## VEREDICTO DEL GATE REAL (2026-07-25, 286 partidas / 2459 manos): REPROBADO

ModoLunar corrió en producción 3 días (`--modo-lunar`, periodo limpio post-
archivado). Atribución por replay exacto (misma clase/pesos/umbrales sobre las
manos reconstruidas; fidelidad verificada: 49/53 pases ofensivos ejecutados
tal cual por la app):

| Grupo | n | lunas | rel pts/mano |
|---|---|---|---|
| Comprometidas | 33 (1.3 %) | 3 (**conversión 9 %**) | +10.9 (fallos +14.6) |
| Huérfanas (pase ofensivo sin compromiso) | 25 | 0 | **+13.8** |
| Normales limpias | 2207 | 10 naturales (≈6.5 esperadas) | **−0.478 = baseline −0.49** ✓ |

**Economía real: −2.5 pts/partida EN CONTRA** (ganancia 3×26=78 vs coste
437+345=782 pts relativos). Win-rate 20.5 % vs baseline 23.0 % (±3 pp, el neto
de −2.5 pts/partida es consistente con ese drift).

**⚠ CORRECCIÓN (2026-07-25, bug de instrumento):** el replay de atribución no
llamaba `motor.resolver_baza()` → solo veía la baza 1 (por eso "0 mid-mano" y
"0 rupturas": artefactos). Números corregidos con el loop arreglado:
**comprometidas 72 (39 mid-mano), 8 lunas, conversión 11 %**, fallos 64 ×
+12.2 rel → neto **−2.0 pts/partida en contra**. Lunas naturales 13−8=5 (≈6.5
esperadas ✓). Las rupturas humanas ocurren a lo largo de toda la mano (mediana
baza 7, spread 0–12 desde el compromiso) — la defensa humana es captura
ordinaria de bazas con puntos, no solo stoppers finales. **El veredicto
REPROBADO se mantiene** (misma dirección, magnitud −2.0 vs −2.5). Regla para
futuros replays: tras `jugar_carta`, con 4 cartas en mesa SIEMPRE
`resolver_baza()` (ver `atribucion_v2.py`).

**La causa: los humanos SÍ defienden la luna.** La conversión colapsa de
44–52 % (clon, que no defiende) a **9 %** (reales). Con 9 % de conversión el
breakeven exigiría fallos de <3 pts; cuestan ~14.6 porque la mano ya quedó
moldeada sin escapes. La incertidumbre que el clon no podía resolver quedó
resuelta: **en contra**. (Mi hipótesis "contra defensores los fallos se
abaratan" era falsa: el defensor niega el payoff, no abarata el intento.)

**Acción: APAGAR `--modo-lunar`** (reiniciar el servidor sin el flag). El
campeón puro sigue siendo la mejor política en producción. Palanca de ofensiva
de luna por composición: CERRADA con datos reales — reabrir solo con un
enfoque cualitativamente distinto (p.ej. compromiso consciente del oponente),
no con ajuste de umbrales (33 compromisos no dan señal para re-calibrar).

**Lo que el periodo SÍ dejó:**
- Fix del reveal verificado en producción: **100 % reconstruible** de fábrica.
- +2459 manos limpias para el volante de datos (clon/moon models), incl. 191
  manos de luna nuevas.
- El instrumento de gate (flag + periodo limpio + atribución por replay)
  queda listo para cualquier experimento futuro.
- El diagnóstico original SIGUE en pie: los rivales nos lunean 2.56 %/mano
  (igual que antes) y en manos normales somos mejores (−0.48). El déficit es
  real pero su explotación requiere más que composición.

## Artefactos

- `hearts-sfs-bridge/src/export-reconstructed-from-jsonl.js` (nuevo, en el bridge;
  incluye el fix de manos fantasma)
- `data/partidas_bridge_full.jsonl` (556 partidas / 4910 manos tras la
  recuperación de reveals 2026-07-21; validado puntos==servidor mano a mano)
- `models/moon_realfull/{propio,rival}.pt` (AUC 0.945 / 0.707)
- `data/regret_v10c_clean.jsonl` (campeón) + `data/regret_experto_clean.jsonl` (referencia)
  + sus `.log`. Los `*_full.*` son la versión previa al fix (contaminada).
