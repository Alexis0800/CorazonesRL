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