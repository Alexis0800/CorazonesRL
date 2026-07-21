# Auditoría de datos reales + moon-prob (2026-07-20)

Auditoría senior sobre las partidas reales capturadas por el bridge SFS
(`hearts-sfs-bridge`) para entender por qué v10c **no le gana de forma
consistente a los humanos** (win-rate 23.9 % ≈ azar) y qué palanca lo mejora.

## TL;DR (revisado tras verificación adversarial)

El diagnóstico previo *"no measurable improvement por escasez de datos de pozo"*
era correcto en la causa pero se midió sobre una foto vieja de datos. La DB del
bridge ya tenía **~30× más datos reconstruibles sin importar**. Al desbloquearlos:

- Los modelos moon pasaron de **azar** (AUC 0.44–0.58) a **señal real**
  (propio **0.945**, rival **0.707**) — sobre datos LIMPIOS (ver §Verificación).
- Las manos de luna rival cargan **~2× el regret** de las normales (1.93 vs 0.92)…
  **PERO el BotExperto está igual de elevado ahí (2.07)**. Es decir: NO es un
  defecto de política del campeón, es una **brecha estructural de información**
  (el oráculo ve las manos ocultas; ninguna política pública puede igualarlo en
  defensa de luna). El campeón ya juega las manos de luna a nivel heurístico-fuerte.
- **Implicación para Fase 3**: el regret NO es el yardstick correcto (no puede
  mostrar mejora en una brecha estructural). Fase 3 es un experimento razonable
  —hay una señal pública nueva que el entrenamiento nunca tuvo— pero con riesgo
  real de resultado nulo, y debe juzgarse por **tasa de lunas-en-contra /
  win-rate en evaluación**, no por regret.

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

Guardados en `models/moon_realfull/` (no pisa los vivos `models/moon/`).

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
  más real por eso.

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

## Artefactos

- `hearts-sfs-bridge/src/export-reconstructed-from-jsonl.js` (nuevo, en el bridge;
  incluye el fix de manos fantasma)
- `data/partidas_bridge_full.jsonl` (558 partidas / 4168 manos, LIMPIO)
- `models/moon_realfull/{propio,rival}.pt` (AUC 0.945 / 0.707)
- `data/regret_v10c_clean.jsonl` (campeón) + `data/regret_experto_clean.jsonl` (referencia)
  + sus `.log`. Los `*_full.*` son la versión previa al fix (contaminada).
