# Veredicto: desde-cero v12 — GO-CONDICIONAL

## 1. Decisión y probabilidad honesta

**GO-CONDICIONAL.** Correr un desde-cero de 40M esta noche, **solo** tras un diff de 2 líneas en `opponent_pool.py` (condición bloqueante, ver 2.1) y con los kill-gates pre-declarados de 2.5. Si falla cualquier gate, se mata sin apelación: el costo hundido máximo es ~50 min (gate de 5M).

**Por qué GO y no NO-GO**: la lección que justificaba el NO-GO ("30M desde BC no alcanzan el linaje del campeón") quedó **refutada con datos primarios** (CONFIRMADO): el campeón ES un desde-cero de 20M+5M desde *exactamente el mismo archivo BC* que usó v11 (md5 idéntico entre `models/bc/bc_v10b.pkl` y `models/produccion/bc_base_con_pase.pkl`). v11 tuvo 20% más pasos que todo el linaje del campeón y aun así quedó a −16pp vs experto. El fracaso de v11 fue de **mezcla** (entropy 0.03, mesa_humana 0.5 desde paso 0, snapshot 200k), no de presupuesto. La receta v10c es reproducible y nunca se re-intentó con los ingredientes nuevos bien dosificados.

**Probabilidad honesta contra la vara** (pareado vs clon v3 ≥ 0.47 + muerte-4º <11% + conversión lunero <1.6%/mano, y luego bridge):

| Resultado | P estimada | Base |
|---|---|---|
| Igualar fuerza del campeón (~0.45-0.47 vs clon) | ~40-50% | v10c lo hizo una vez con esta receta; varianza de semilla desconocida (n=1), y toda modificación previa restó fuerza |
| **Superar la vara compuesta** (≥0.47 + gates de defensa) | **~25-30%** | Requiere que Φ_rank + lunero sumen sin restar; escala histórica de ganancias por ingrediente ~1pp; 5 intentos RL de este ciclo murieron |
| Que eso se traduzca en mejora vs **humanos reales** | desconocida, no medible offline | win-vs-clon ya demostró NO predecir el bridge (0.45-0.47 vs clon ↛ 23% vs humanos). Solo el bridge decide |

Traducido: es más probable que este run muera en un gate a que supere al campeón. Se corre igual porque (a) el costo con gates es de minutos-a-horas, no de días; (b) es la única vía RL no refutada; y (c) incluso un empate en fuerza con mejor defensa de luna es promovible (la firma de Φ_rank ya demostró muertes-4º baratas en el Run A).

---

## 2. SPEC completa del run

### 2.1 Pre-trabajo bloqueante: diff de 2 líneas en `src/rllib/opponent_pool.py` (CONFIRMADO como bug)

Sin esto, `--lunero-garantizado` **borra el ancla experto y mata `prob-humano`** en fases 2-4 (pisa `fns[opp_indices[0]]`, el slot del oponente duro — exactamente lo que dejó al Run A sin ancla), y `mesa_humana` diluye el bootstrap desde el paso 0 (lo que le pasó a v11).

```python
# línea 406 — gate de fase para mesas humanas:
if humano_pesos is not None and progress >= 0.40 and random.random() < mesa_humana:

# línea 456 — lunero a un slot de snapshot, no al slot del duro:
fns[opp_indices[1]] = OponenteLunar()
```

Seguro por construcción: en todo branch donde el guard del lunero puede dispararse (progress≥0.15, ≥1 snapshot), `opp_indices[1]` contiene un snapshot. Dejar UN check: test que con `progress=0.5, anclar=True, lunero=True` la mesa contenga {duro, OponenteLunar, snapshot}.

**Init elegido**: `models/produccion/bc_base_con_pase.pkl` (BC-PIMC, el mismo del campeón). **NO destilación** — su tooling no existe (medio día realista: generador on-policy con logits+values, trainer soft-label, GATE-0 pareado) y bloquearía la noche; queda como plan B (sección 3).

### 2.2 Comando exacto (tras el diff)

```bash
python scripts/train_rllib.py \
  --total-steps 40000000 --workers 10 \
  --output-dir models/v12_scratch \
  --con-pase --obs-dim 228 \
  --bc-weights models/produccion/bc_base_con_pase.pkl \
  --ancla-experto --pool-diverso \
  --humano-bc models/humano_bc/pesos.npz \
  --prob-humano 0.3 --mesa-humana 0.2 --temp-humano 1.0 \
  --lunero-garantizado --phi-rank \
  --moon-dir models/moon_realfull \
  --lr 1e-4 --lr-end 5e-5 --entropy-coeff 0.02 \
  --batch-size 8192 --gamma 0.999 --phi-lambda 0.5 \
  --snapshot-interval 100000 --eval-partidas 150
```

Receta base = v10c exacta (lr 1e-4→5e-5, entropy 0.02 — 0.01 mesetó a ~0.45, 0.03 fue el v11 plano; snapshot 100k, NO 200k). Ingredientes nuevos en dosis baja. **Nunca lr 3e-4 desde BC** (solo se usó con init aleatoria; destruiría el prior que es toda la ventaja temprana). `--eval-partidas` en 150 (CI ±0.078; con 75 los gates pierden poder).

### 2.3 Pool por fases (real, post-diff; límites = progress × 40M)

| Fase | Pasos | Mesa (80% en F3-F4) |
|---|---|---|
| F0 | 0–2M | 3 bots simples (sanity del init BC — ya sin dilución de mesa humana) |
| F1 | 2–6M | 2 simples + 1 duro (70% arquetipo diverso / 30% clon v3) |
| F2 | 6–16M | 1 duro + 1 OponenteLunar + 1 snapshot |
| F3 | 16–28M | 1 duro(ancla/clon 30%) + 1 lunero + 1 snapshot · **20% de mesas**: 2 clones v3 + 1 ancla heurística |
| F4 | 28–40M | igual con snapshots recientes |

Exposición efectiva al clon en F3-F4: ~0.64 asientos/mesa (v11 tuvo ~1.25 y no le compró nada — la fuerza-vs-clon llega con la fuerza general, no con la exposición).

### 2.4 Presupuesto y cadencias

- **40M ≈ 6.5-7h** a 5.9-6.2 M/h medidos (mesa_humana 0.2 interpolada; mesa 0.5 costaba 22-26%). Cabe en una noche.
- Extender a **60M por tramos** (`--resume` + `--total-steps` acumulativo, mecánica probada) SOLO si a 35M la mediana de 3 evals vs experto sigue subiendo. No presupuestar 100M: ningún run del repo ganó después de una meseta, y la extrapolación de v11 cruzaba el nivel campeón a ~200-760M.
- Reservar un posible **remate de +5M a lr 3e-5→1e-5** tras los 40M: el 0.60 estable del campeón lo aportó esa fase, no los 20M frescos.

### 2.5 KILL-GATES pre-declarados (mediana de 3 evals @150 partidas, vs curva de v11 al mismo paso)

| Paso (~hora) | Decisor | Umbral kill | v11 aquí | v10c aquí |
|---|---|---|---|---|
| **5M** (~0.85h) | `win_rate_vs_humano` | **< 0.25 → matar** | 0.19 (kill limpio) | n/a |
| 5M | vs experto (alarma, no kill) | < 0.30 preocupante | 0.287-0.293 (ruido) | 0.34-0.40 |
| **10M** (~1.7h) | vs experto | **< 0.40 → matar** (mediana-3 obligatoria: el punto suelto de v11 dio 0.407) | 0.347 | 0.44 |
| 10M | vs humano | < 0.35 → matar | 0.293 | n/a |
| **20M** (~3.4h) | **`comparar_snapshots.py` pareado 300 vs mesa mixta** | **puesto medio > 1.75 → matar** (campeón ~1.70) | — | pasa |
| 20M | vs experto (alarma) | < 0.45-0.50 preocupante (NO 0.55: falso-kill ~50% sobre un run fuerza-campeón) | 0.38 | ~0.54 ventaneado |

**Nunca promover ni continuar por bot_eval solo** — ha elegido mal dos veces (README de producción + Run A: bot_eval 0.37-0.60 mientras el pareado daba 0.388 vs 0.453). Riesgo declarado: ~15% de falso-kill a 10M para un run exactamente fuerza-v10c.

**Varas finales antes del bridge**: pareado vs clon ≥ 0.47 (2000 partidas), muerte-4º < 11.0%, conversión lunero < 1.6%/mano (`scripts/metricas_gate.py`), y **FiltroQS encima al final** (ortogonal, +0.55pp gratis sobre cualquier modelo).

---

## 3. Alternativas descartadas

- **5º fine-tune del campeón**: refutado 4/4 — degrada más rápido de lo que enseña.
- **Resume/alargar v11 a 40-60M**: cola plana (slope −0.0005/M) y empeorando en la meta humana; cruce del nivel campeón a ~200-760M (30-115h).
- **Desde-cero receta v11 sin cambios**: sería re-correr un fracaso caracterizado (mezcla tóxica, no presupuesto).
- **Init aleatoria**: dominada por BC init en toda ventana disponible (0.12 vs 0.38 al primer eval, gap nunca se cierra).
- **100M de entrada**: retorno marginal medido ~+0.06 por duplicación de pasos; por tramos con gates o nada.
- **Copia de pesos del campeón como "init"**: es el fine-tune de siempre con Adam fresco y logits saturados (entropía 0.32).
- **Sonda destilada 8M (soft labels T~2 + value destilado)**: el único experimento genuinamente no corrido (optimizador vs cuenca), pero exige ~medio día de tooling inexistente + GATE-0 pareado del init; **plan B si el scratch muere en los gates**, no bloqueo de esta noche. Ojo si se hace: el value head del campeón NO es copiable (lineal sobre encoder compartido) — hay que destilarlo por regresión.

## 4. Qué corre EN PARALELO con el bridge del usuario

El run de 10 workers **satura la CPU: no lanzarlo durante sesiones de bridge** (degradaría la latencia de `servidor_inferencia` y sesgaría la medición en curso). Programarlo nocturno (6.5-7h caben). Sin conflicto de artefactos: escribe solo en `models/v12_scratch/`, no toca `models/produccion/` ni el flag `--modo-lunar` del bridge.

Compatible con el bridge en marcha (costo CPU ~nulo):
1. El **diff + test de `opponent_pool.py`** (2.1) — solo código.
2. Escribir el **tooling de destilación** (plan B): generador on-policy con logits/values + soft-label trainer — solo código; el cómputo del dataset son ~minutos y se corre después.
3. La propia **medición del bridge con `--modo-lunar`** sigue siendo el dato más valioso del ciclo y no depende de nada de esto.