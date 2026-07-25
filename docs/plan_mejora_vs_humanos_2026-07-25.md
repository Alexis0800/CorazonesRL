# PLAN PRIORIZADO — superar a los humanos reales (bridge)

**Situación consolidada:** win-rate 23% (paridad 25%). El déficit total es ~5.7 pts rel/partida de diferencial de luna + ~0.7 de gestión del borde (endgame como 4º) + un residuo pequeño de "mano limpia" (+0.30±0.15 rel/mano tras controlar reparto). En manos normales el bot ya es mejor. La interceptación de lunas realizadas está muerta como palanca (techo 0.41 pts, señal pública llega tarde, regla bolt-on EV-negativa): la defensa de luna solo puede venir de **entrenamiento temprano**, no de reacción.

---

## Ranking (impacto esperado × confianza / coste)

| # | Palanca | Impacto esperado (pts rel/partida) | Confianza | Coste | Reentrena |
|---|---|---|---|---|---|
| 0 | Correcciones de doc + cierre de líneas | 0 directo (evita gasto futuro) | alta | ~1h | No |
| 1 | Filtro Q♠ evitable en inferencia | +0.8–2.0 (techo 2.5; central ~1.3) | media | ~2 días | No |
| 2 | Medición: elites/snapshots vs clon | prob. ~0; opción de +1 si hay spread | alta (en que es barato) | ~1h cómputo | No |
| 3 | Pool con lunero oportunista (defensa) | +1–2 conjetura (techo duro 5.7) | media-baja | 1 run fine-tune + gates | Sí |
| 4 | Φ_rank en PBRS (borde 100 + puesto) | +0.5–1 especulativo | media | trivial de código; va dentro del run | Sí |
| 5 | Ruido ε en fracción del pool | +0.5–1.5 (techo 1.8, sensible a controles) | media-baja | 2º run o ablación | Sí |

---

## FASE 0 — Hoy, coste ~0 (hacer ya, en paralelo con todo)

**Toca:** `docs/auditoria_moon_2026-07-20.md`.

Registrar con números corregidos para no re-derivar:

- **Corrección del diagnóstico:** las lunas rivales realizadas casi no son rompibles desde nuestro asiento (5.9% de manos con ventana, techo de interceptación perfecta ~0.41 pts rel/partida = 7% del déficit). La "brecha estructural" no es que el bot no rompa lunas.
- **NO implementar** la regla bolt-on "captura si coste ≤6": medida EV **−0.14 a −0.46** pts rel/partida (precisión 0–6.6%; la captura es garantizada, la luna no).
- **Cerradas con evidencia:** [188] como defensa reactiva in-hand (intersección alarma∩ventana 2/559); "enseñar a rematar errores" (ya capitaliza: **−0.49±0.18**, no −0.86; no "en las 11 bazas"); alimentar al líder en endgame (bot igual o mejor que humanos: 0.332→0.364 vs humanos bajo su baseline); bazas tardías y goteo de corazones (paridad/ventaja); "cerrar liderando" (artefacto de composición: con 4º fijo, 5.1% vs 5.2%).
- **Hechos nuevos citables:** el hallazgo real de endgame es unidireccional (supervivencia como 4º: muerte 12.0% vs 5.3%, z=3.8, robusto a lunas y a "cargar al bot"); firma "no descarga altas TEMPRANO" (bot 33.8% duck vs humano 19.6% con Q♠ fuera; se igualan tarde); bug de datos del bridge: última mano de cada partida tiene `points=[0,0,0,0]` — reconstruir desde `tricks`.

---

## FASE 1 — Sin reentrenar (paralelo, esta semana)

### Palanca 1: Filtro Q♠ evitable en `servidor_inferencia.py` (la mejor relación valor/coste del plan)

**Qué se toca:** filtro de composición estilo `ModoLunar` detrás de flag `--filtro-qs` (OFF por defecto). Reglas, con las correcciones del verificador incorporadas:
- Si la Q♠ está en mesa y existe legal que NO gana la baza → vetar ganadoras. **Excepto** cuando no hay alternativa (22/53 lideradas eran única legal) y **excepto** cuando ganar es correcto (endgame donde comer 13 nos mata menos que al rival cerca de 100 — condición simple: no aplicar si capturar termina la partida a nuestro favor o evita nuestra muerte).
- No liderar la Q♠ salvo K/A♠ ya jugadas.
- Casos "deferral" (todo duck deja ganar a nuestra propia Q): el filtro solo pospone; aceptarlo — posponer sigue teniendo valor opcional, pero **no contarlo como ganancia** en la proyección.

**Validación antes de gastar (3 escalones, el clon NO valida reacciones humanas):**
1. **Offline (gratis):** replay contrafactual sobre los 155–192 casos evitables estrictos — verificar que el filtro dispara en ellos y NO dispara en los 656 dump_forced / 354 self_forced.
2. **Pareado vs clon** (`scripts/evaluar_modo_lunar.py` como plantilla, ≥3000 partidas, semillas pareadas): gate de **no-regresión** (≥0 con IC). Esto solo descarta que el filtro rompa algo; no prueba la ganancia con humanos.
3. **Bridge real** con flag ON, gate pre-declarado ANTES de encender: en ~300 partidas (~3 días al ritmo actual de ~95/día), (a) tasa de Q♠ comida en la clase evitable (medible por replay del JSONL nuevo) baja vs baseline 10–12.6%, y (b) pts rel/mano no empeora. Outcomes SIEMPRE de la DB sqlite.

**Impacto honesto:** techo 2.5 rel/partida; con 40–60% realizable y descontando deferrals, esperar +0.8–2.0. Es el único ítem del plan con mecánica mayormente determinista (sustitución de carta), lo que reduce la dependencia de reacciones humanas.

### Palanca 2: Re-evaluar candidatos vs clon (medición, no entrenamiento)

**Qué se toca:** nada de producción. Script sobre `src/rllib/eval_bots.py` (`_eval_model_vs_factory`) contra 3× `SnapshotPolicy.from_weights(models/humano_bc/pesos.npz)`.

**Correcciones del verificador incorporadas:** candidatos = `models/v10c_finetune_pozo/elite/*` (9) + últimos snapshots de ese run, con el **campeón actual como baseline** (vara existente: 0.480 vs clon-v2, 300 partidas). Semillas pareadas; el ganador aparente se confirma con semillas frescas (winner's curse: a n=300, elegir el máximo de ~15 infla +4–6pp); si supera al campeón tras confirmación, validación held-out en partidas reales antes de promover.

**Gate:** solo promover si gana pareado + confirmación fresca + held-out real. **Prior débil** (todo lo medido en la campaña quedó bajo 0.480): el valor principal es cerrar o abrir la palanca con evidencia por ~1h de cómputo.

### Prerequisito de Fase 2 (hacer ahora, en paralelo): métricas de gate offline

**Qué se toca:** `scripts/comparar_snapshots.py` / `src/rllib/eval_bots.py`. Añadir por-snapshot, medibles por replay barato:
- **Tasa de duck innecesario TEMPRANO** (Q♠ sin jugar): bot 33.8% vs humano 19.6% — la señal vive entera ahí, no en la tasa global.
- **Pts E3 propios** (0.151/mano).
- **Muerte como 4º en endgame** (12.0% vs 5.3% humano).
- **Conversión de luna del rival** en partidas vs lunero-oportunista en el env.

Sin estas varas, la Fase 2 no tiene forma barata de saber si un run movió lo que debía antes de pagar el gate real.

---

## FASE 2 — Un reentrenamiento gated (después de Fase 1; prior en contra: 3 fine-tunes refutados)

Los tres cambios de entrenamiento se ordenan así porque el pot de la defensa de luna (5.7 pts) es 3–5× mayor que los demás. **Disciplina de atribución:** un cambio de distribución de oponentes por run; Φ_rank puede ir como flag A/B dentro del mismo run porque su ablación es barata (dos colas del mismo pool).

### Palanca 3 (Run A, principal): lunero oportunista garantizado en el pool

**Qué se toca:** `src/rllib/opponent_pool.py`:
- (a) garantizar ≥1 arquetipo lunero por mesa en fases 2–4 (hoy: P=0.4 de mesa, 1 solo slot, 13% de asientos — vs realidad 3/3 asientos al 2.5%/mano).
- (b) lunero **oportunista**: reusar la lógica dynamic-commit de `src/agentes/modo_lunar.py` como OPONENTE (13/14 lunas humanas son mid-hand; `BotLunatico` se compromete desde el pase — estilo equivocado). Esto no reabre la ofensiva cerrada: es rival de entrenamiento, no política propia.
- **No tocar el pase del agente** (hecho cerrado: el pase defensivo no es facilitador).

### Palanca 4 (dentro del Run A, flag A/B): Φ_rank en `src/entorno/recompensas_partida.py`

**Qué se toca:** `potencial()` → Φ_rank = λ·(R_terminal interpolado del puesto actual, **interpolación por gap al rival inmediato obligatoria** — la versión escalón salta λ·0.6 por 1 punto), con empate 4-way promediado para que Φ(0,0,0,0)=0 y gamma consistente con PPO. Flag en `RewardConfigPartida` para A/B contra el Φ actual. Mecanismo CONFIRMADO (9.7% de fronteras con potencial contradiciendo el orden terminal; clip nunca satura). Objetivo del gradiente: **sobrevivir como 4º al borde** y ordenar por puesto — NO "cerrar liderando" (artefacto refutado).

### Palanca 5 (Run B, solo si A paga o en ablación): ruido en fracción del pool

**Qué se toca:** `opponent_pool.py` — envolver una fracción (5–15%) de `SnapshotPolicy` con ε-random o softmax con temperatura; opcionalmente regenerar dataset BC con mundos PIMC de arquetipos mixtos + jugadas aleatorias-legales (`src/mcts/pimc.py`, `src/mcts/dataset.py`). Distinto de la palanca cerrada (clon como rival único). Techo honesto 1–1.8 y el residuo base es marginal (z~2): es la de menor prioridad de las tres.

### Gates de Fase 2 (pre-declarados ANTES de lanzar el run)

1. **Offline por snapshot** (las métricas del prerequisito): Run A debe bajar conversión de luna rival vs lunero-oportunista y (si Φ_rank ON) bajar muerte-como-4º; Run B debe mover duck temprano 34%→hacia 22%. Si la métrica objetivo no se mueve en el env, el run muere ahí, sin gastar más.
2. **Pareado vs clon:** no-regresión sobre 0.480 (semillas pareadas + confirmación fresca). Solo sanidad — el clon no defiende lunas como humano ni valida reacciones.
3. **Bridge real** (el único gate que cuenta): flag de despliegue, ~300–500 partidas (3–5 días), métrica primaria pre-declarada: **conversión de luna rival 2.5%→<2.0%/mano** (Run A) medida en la DB sqlite; secundarias: pts rel/mano, win-rate. Con ~4400 manos, un drop de 0.8pp/mano/rival es detectable (SE~0.4pp sobre la tasa agregada 7.5%). Recordar: en el periodo ModoLunar las lunas rivales NO bajaron (2.52→2.56) — esa es la vara nula.

---

## Dependencias y paralelismo

```
HOY (paralelo):     Fase 0 (docs) ─┐
                    Palanca 1 (filtro Q♠: código + offline + pareado) ─┐
                    Palanca 2 (medición elites, 1h) ─┤                 │
                    Métricas de gate offline ────────┤                 │
                                                     │                 │
SEMANA 1-2:         Palanca 1 → bridge real (flag, 300 partidas) ◄─────┘
                    Fase 2 Run A (lunero + Φ_rank A/B) ◄── requiere métricas de gate
                                                     │
DESPUÉS:            Run A → pareado → bridge real    │
                    Run B (ruido) solo si A pagó o como ablación separada
```

- Palanca 1 y Fase 2 son independientes: el filtro Q♠ puede estar en producción mientras se entrena (pero el periodo de medición del bridge para el gate de Fase 2 debe registrar qué flags estaban ON — no confundir efectos; idealmente estabilizar el filtro ANTES de abrir la ventana de medición del Run A).
- Palanca 2 no bloquea nada; si sorprende con un ganador, se promueve por su propio gate.

## Incertidumbre, sin inflar

- **Suma de techos ≈ 4–6 pts rel/partida** si todo paga — comparable al déficit total, pero los techos no se suman limpiamente (la defensa de luna y el residuo limpio se solapan poco; Q♠ y ruido algo más).
- **Escenario realista:** Palanca 1 +~1, Fase 2 +0 a +2 (el prior de fine-tunes es malo; la diferencia esta vez es que el pool nunca ejerció presión de luna realista — es un cambio de distribución de datos, no de rival-clon — pero eso es un argumento, no evidencia).
- **Lo único que decide es el bridge real con gate pre-declarado.** El clon valida no-regresión, nada más; los humanos defendieron la luna cuando el clon no lo hizo, y pueden reaccionar al filtro Q♠ de formas que el pareado no muestra (p.ej., dejar de tirarnos la Q si dejamos de comerla).

## No hacer (cerrado con evidencia, no re-proponer)

Regla bolt-on de romper luna (EV −0.14 a −0.46) · invertir en [188] como defensa reactiva · enseñar a rematar errores · `puedo_alimentar`/ataque al líder · heurísticas de conversión post-mitad · bazas tardías / goteo de corazones · ofensiva de luna por composición · fine-tune/desde-cero vs clon como rival único · regret PIMC como métrica · pase defensivo anti-luna · re-entrenar "para que use [188]".
---

# ANEXO: Plan detallado de ejecución + auditoría de contradicciones (2026-07-25)

## Auditoría de contradicciones (verificación independiente previa a ejecutar)

Antes de comprometer cómputo se re-verificaron los supuestos que cargan el plan:

1. **Q♠ evitable (Fase 1a) — CONFIRMADO con método independiente.** Replay
   propio sobre los 2 corpus (528 Q♠ comidas): 33 sustituibles + 22 jugamos
   nuestra Q teniendo alternativa + 25 la lideramos = **80/528 = 15.2 %**
   (workflow: 14 %). Dos instrumentos distintos, mismo número. Desglose no
   evitable: 230 forzadas, 123 Q propia forzada, 95 líder-ambiguo.
2. **Bug de instrumento encontrado y corregido** (los replays post-gate no
   llamaban `resolver_baza()` → solo veían la baza 1). El veredicto del gate
   se re-midió: comprometidas 72 (39 mid-mano), conversión 11 %, neto
   **−2.0 pts/partida** — REPROBADO se mantiene, con números corregidos en
   la auditoría. Regla: todo replay manual debe resolver bazas.
3. **Tensión "solo 6 % rompible" vs "los humanos nos rompen al 89 %" —
   RESUELTA, no es contradicción.** El 6 % está sesgado por superviviente
   (solo lunas REALIZADAS). Las rupturas de nuestros 64 intentos fallidos
   ocurren a lo largo de toda la mano (mediana baza 7, spread 0–12 desde el
   compromiso): la defensa humana es captura ordinaria de bazas con puntos —
   comportamiento APRENDIBLE, lo que sostiene la premisa del Run A.
4. **Riesgo v10e (fine-tune sin gradiente, refutado 2×) vs Run A — abierto y
   mitigado.** Diferencia argumentable: la presión de luna crea eventos de
   reward nítidos (swings ±26 rel) vs el shift difuso de estilo del clon que
   no movió gradiente. Es un argumento, no evidencia → el gate offline
   (¿baja la conversión del lunero en el env?) mata el run barato si no hay
   gradiente. Punto de decisión explícito abajo.
5. **Φ_rank no puede cambiar el óptimo** (PBRS es policy-invariante con
   cualquier Φ): el beneficio esperado es de ASIGNACIÓN DE CRÉDITO
   (supervivencia como 4º al borde), no de objetivo. Por eso va como A/B
   dentro del Run A, no como run propio.

## FASE 1 (sin reentrenar — esta semana)

### 1a. Filtro Q♠ (`--filtro-qs`, OFF por default)

- **Código** (~medio día): clase `FiltroQS` en `src/agentes/` (patrón ModoLunar:
  devuelve `None` = no aplica). Reglas: (R1) Q♠ en mesa y existe legal que
  pierde la baza → vetar ganadoras; (R2) nunca liderar la propia Q♠ salvo
  K/A♠ ya jugadas; (R3) siguiendo picas con la Q en mano y pudiendo jugar
  otra → no jugar la Q salvo forzado. Excepción endgame: no aplicar si
  capturar nos da la partida o evita nuestra muerte. Cablear en
  `Recomendador` tras `ModoLunar` (misma cadena de intercepción).
- **Validación offline (gratis)**: replay sobre los 80 casos evitables
  identificados → el filtro debe disparar en ≥70 y en <2 % de las 448
  no-evitables (falsos positivos).
- **Pareado vs clon** (no-regresión): 3000 partidas, gate Δwin ≥ −0.5 pp.
- **Bridge real** (gate pre-declarado ANTES de encender): ~300 partidas,
  métrica primaria = tasa de Q♠-comida-en-clase-evitable 15.2 % → <8 %
  (medible por replay del periodo); secundaria pts rel/mano no peor.
- Registrar en el log del servidor cada intervención (`[filtro-qs]`).

### 1b. Re-evaluación de elites archivados (1 h de cómputo)

Candidatos: `models/v10c_finetune_pozo/elite/*` + últimos snapshots. Baseline:
campeón 0.468 vs clon v3 (pareado). Protocolo winner's curse: ganador aparente
→ confirmación con semillas frescas → held-out real solo si supera.

### 1c. Métricas de gate offline (prerequisito de Fase 2)

En `src/rllib/eval_bots.py`/`comparar_snapshots.py`: duck-innecesario-temprano
(vara: bot 36 %, humano 22 %), pts propios en endgame como 4º (muerte 12 % vs
5.3 %), conversión de luna rival vs lunero-oportunista en el env.

## FASE 2 (UN run gated — después de 1c)

### Pre-check de calibración (antes de lanzar)

Simular 200 partidas con el pool propuesto y medir tasa de luna rival
lograda contra el campeón congelado: objetivo ≈ 2–3 %/mano (realidad humana
2.5 %). Si el lunero-oportunista no corona en el env, ajustar ANTES del run.

### Run A: presión de luna realista + Φ_rank (A/B)

- `opponent_pool.py`: ≥1 lunero por mesa en fases 2–4; lunero = wrapper de
  oponente con la lógica dynamic-commit de `ModoLunar` (oportunista mid-mano,
  como los humanos) sobre un arquetipo base.
- `recompensas_partida.py`: `phi_rank: bool` en `RewardConfigPartida` —
  Φ_rank = λ·R_terminal_interpolado(puesto, gap al rival inmediato), empates
  promediados, Φ(0,0,0,0)=0. Test unitario de telescopaje.
- Fine-tune desde el campeón, LR conservador, ~5–10 M pasos, dos colas A/B
  (Φ actual vs Φ_rank) del mismo pool.
- **Gates en orden (cada uno mata el run barato):**
  1. Gradiente: reward de entrenamiento se mueve en 1–2 M pasos (si plano →
     PUNTO DE DECISIÓN: aceptar cierre de la vía fine-tune con esta
     distribución, no insistir).
  2. Offline: conversión del lunero en env baja ≥25 % relativo; muerte-como-4º
     baja (cola Φ_rank); sin regresión en duck/QS.
  3. Pareado vs clon: no-regresión sobre 0.468.
  4. Bridge real 300–500 partidas: **lunas rivales 2.5 → <2.0 %/mano** (SE
     ~0.4 pp con ~4400 manos) y muerte-como-4º 12 → <8 %. Vara nula conocida:
     en el periodo ModoLunar las lunas rivales NO se movieron (2.52→2.56).

### Run B (condicional): ε-ruido en pool

Solo si Run A paga o como ablación separada. Gate offline propio:
duck-temprano 36 % → hacia 22 %.

## Cronograma y dependencias

- Día 1–2: 1a código + validación offline; 1b y 1c en paralelo.
- Día 2–3: 1a pareado vs clon; pre-check de calibración del pool.
- Día 3+: 1a al bridge (300 partidas ≈ 3 días al ritmo actual); Run A se lanza
  cuando 1c esté y SIN solapar su ventana de medición con el switch del
  filtro (estabilizar flags antes de abrir la ventana del Run A).
