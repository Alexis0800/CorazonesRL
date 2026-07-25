# Plan — Siguiente iteracion de entrenamiento (post-v12_scratch)

Base: 11 hallazgos verificados (0 refutados) sobre telemetria del run vivo, medicion empirica de Φ_rank contra 844 partidas reales, auditoria de obs 228 sobre 637 decisiones reales, y auditoria de config PPO contra el codigo instalado de Ray 2.55.1. Nada de lo aqui propuesto toca el run actual.

---

## 1. Vigilancia del run actual: 3 numeros para los gates de 10M y 20M

| # | Metrica | Fuente | Gate 10M | Gate 20M | Accion si falla |
|---|---------|--------|----------|----------|-----------------|
| 1 | **Entropia** (media de ventana 1M en `models/v12_scratch/logs/eval_log.jsonl`) | eval_log | Sano >= 0.42; vigilancia 0.38-0.42; preocupacion < 0.38 sostenido 1M (v10c NUNCA lo hizo); colapso < 0.32. Punto de decision temprano: ventana 7-8M (tras el cambio de fase a 6M). | Igual: ninguna ventana 1M < 0.38 sostenida | Si < 0.38 sostenido sin despegue: preparar relanzamiento con `entropy_coeff` 0.03 o schedule 0.03->0.015 para la SIGUIENTE iteracion. NO matar el run solo por esto. |
| 2 | **win_rate vs experto** (bot_eval_log) | bot_eval | Despegue esperado ~8.5M (progress-matched: v10c desperto a 21% = 4.26M/20M con salto a 0.62). A 10M: si sigue en el plateau 0.20-0.29 **Y** entropia < 0.40 → interpretar como colapso temprano, no pre-despegue. | A 20M (50% progress) debe estar claramente post-despegue (>= 0.5); v10c a 50% ya llevaba media partida despegado. | Solo la COMBINACION (sin despegue + entropia baja) justifica matar; cualquiera de las dos sola es esperable. |
| 3 | **moon_rate** (bot_eval) | bot_eval | v10c vivia en 0.02-0.06; v12 ultimo punto 0.016. Si < 0.015 a 10M con `lunero_garantizado=true` → anotar auditoria "el pool ensena defensa pero no ofensiva de luna" para la siguiente iteracion. | Recuperado a >= 0.02 | No es criterio de kill; es diagnostico para el diseño del proximo run. |

Contexto honesto sobre el numero 1: la entropia de v12 cruzo 0.4 al 12.3% de progress (v10c al 57.9%) — cae mas rapido que la referencia, posible efecto del clon dosificado `prob_humano=0.3` que v10c no tenia. Pero las alertas se estan ESPACIANDO (2/60 evals recientes, no 8%), el run esta a 2x del regimen del Run A muerto (media 0.218), y el resto de curvas (vs_experto, manos_por_partida, reward) van clavadas a la trayectoria v10c progress-matched. El "rebote de fase" de v10c es un prior debil (+0.013 en ventanas de 200k): si a 8M no rebota, eso NO es anomalo.

Secundaria (sin gate): `win_rate_vs_humano` 0.11-0.20 vs campeon 0.45-0.47 — normal pre-despegue, solo vigilar tendencia.

---

## 2. Cambios propuestos, rankeados por (impacto × confianza / costo)

### Grupo A — Requieren codigo (barato, aplicar en frio antes del proximo lanzamiento)

**A1. Telemetria de learner en callbacks.py** — rank #1 absoluto (costo ~0, confianza alta, desbloquea A2 y los condicionales).
`src/rllib/callbacks.py:68-70`: extraer de `learner_stats` (ademas de `entropy`): `kl`, `cur_kl_coeff`, `vf_loss`, `vf_explained_var`, `policy_loss`, `grad_gnorm`, `cur_lr` (opcionales: `total_loss`, `entropy_coeff`). ~8 lineas, `.get(..., nan)` seguro. Verificado que el old-stack de Ray 2.55.1 ya expone todas esas keys gratis (`ppo_torch_policy.py` stats_fn + `apply_grad_clipping`; `grad_clip=0.5` garantiza `grad_gnorm`).
*A/B*: ninguno — es instrumentacion. Alerta nueva: `kl` sostenidamente >> `kl_target` del run (NO un 0.02 absoluto: el gatillo real del controlador es `2*kl_target` y sube x1.5, no x2) o `vf_explained_var < 0.3` tras fase 2.

**A2. Rediseño Φ_rank (GAP=26 + mix α=0.5)** — rank #2. Spec completa en la seccion 3. ~6 lineas + 1 test. Es el unico cambio con base empirica de 844 partidas reales detras.

**A3. λ_GAE 0.98 (A/B)** — rank #3. Una linea en `src/rllib/config.py:42` (0.95 → 0.98; horizonte 20 → ~47 pasos ≈ 3.3-3.8 manos, en episodios de ~140). Motivacion verificada: con 0.95 el credito del R_terminal que llega por retorno observado a la mano 1 es 1.1e-3 — la señal del PUESTO FINAL depende 100% de V(s) + shaping (la recompensa densa por baza si llega; no confundir). Gate del brazo: `vf_explained_var` (sale de A1) + curva vs-experto; abortar si vf_ev cae o entropy colapsa antes de 3M.
**Restriccion: un factor por run.** Secuencia de runs cortos de 5M: primero A2 (Φ mix vs Φ media), despues A3 con el Φ ganador. No montar ambos en el mismo A/B.

**A4. Condicionales (solo si la telemetria lo pide, decision con datos, no ahora):**
- `entropy_coeff` 0.02 → 0.03 o schedule: SOLO si el gate 8M/10M del run actual falla por entropia (numero 1 de la seccion 1).
- `use_kl_loss=False` (PPO clip-only, compensar con `num_epochs` 10→5): SOLO si la curva `cur_kl_coeff` del proximo run (habilitada por A1) muestra crecimiento sostenido >> 0.2. Hoy el KL adaptativo esta activo y CIEGO (defaults True/0.2/0.01, 160 pasos de gradiente por update) — pero "el KL tipicamente supera el target" es especulacion sin telemetria; primero medir.

### Grupo B — Aplicables por flag a un run nuevo (sin codigo nuevo)

**B1. `--con-pase --obs-dim 332` (memoria del pase v13)** — ya implementado y determinista. Es el sustituto barato del LSTM (misma necesidad: memoria entre steps) sin el costo de un BC-pretrain recurrente que no existe. Candidato principal para el proximo desde-cero grande. Costo real: regenerar dataset BC con dim 332 + run 20-40M — por eso va empaquetado con C1/C2, un solo salto de dim.

### Grupo C — Requieren re-entrenar con obs nueva (empaquetar TODO en el salto a 332, un solo dataset BC, un solo run)

**C1. Poda de dims muertas en `observacion.py`** (medidas en 637 decisiones reales): eliminar `[227]` (hardcoded 0), `[220]` (std=0 estructural), `[191:193]` (all_void nunca dispara salvo treboles); unificar `[196]`/`[182]` (redundancia ESTRUCTURAL garantizada por codigo, r=-1.000). **Cuidado con dos**: `[189]`/`[195]` divergen cuando solo el agente esta >=85 — medir en muestra mayor antes de unificar o conservar ambas; `[224:226]` estan muertas solo por construccion del replay — NO podar en un modelo con pase. Mientras tanto: anotar en el docstring de `observacion.py` cuales dims estan medidas como muertas, para que nadie las "arregle" por separado.

**C2. 3 escalares de riesgo Q♠**: `puntos_en_mesa/26`, `rank_max_palo_salida/14`, `altas_picas_que_cubren_Q_vivas/2`. El hecho estructural esta confirmado (`obs[205]` mezcla J/Q/K/A de picas, la propia Q y el J irrelevante juntos; y el conteo ni siquiera resta la mesa actual). **Expectativa recalibrada, no la del titulo original**: la fuga causal real es ~6-10% de las Q comidas (no 15.2%, que incluia hindsight) y la brecha de duck es +7.6pp (no +14.2pp) — payoff moderado, no dramatico. El corto plazo lo cubre la Palanca 1 ya planificada (filtro Q♠ en `servidor_inferencia`, sin retrain).

### Grupo D — Documentacion (costo cero, hacer hoy)

- Registrar en docs/ el veredicto cuantitativo de la saturacion Φ_rank (seccion 3) — hipotesis abierta → resuelta.
- Registrar que la "refutacion" LSTM esta CONFUNDIDA (v10_lstm corrio sin BC, lr 3e-4, sin pase/ancla/pool; comparacion fue LSTM-frio vs MLP-con-BC), cerrada por COSTO no por evidencia. Matices: el max 0.48 fue un pico de una eval, y los runs LSTM si tuvieron entropia (default implicito), no cero.
- Docstring en `build_ppo_config`: ver seccion 4 (vf_clip).

---

## 3. Rediseño Φ_rank listo para usar

**Veredicto empirico previo (registrar antes del A/B, no re-derivar):** la hipotesis de saturacion es PARCIALMENTE cierta. Version fuerte ("gaps tipicos > 10") falsa para gaps adyacentes del marcador ordenado (mediana 8); PERO cierta para gaps agente-vs-rival (mediana 14, 59% de sigmoides individuales saturadas con GAP=10). Neto: con GAP=10, ~19-21% de las manos reales producen ΔΦ exactamente 0 y ademas cero gradiente por alimentar a un lider a >10 pts. Cifras canonicas (reproduccion del verificador): ΔΦ==0 con g10 = 2306/8062 = 28.6% bruto / 21.4% de las 7321 manos reales netas de no-ops del bridge.

**Cambio** (en `src/entorno/recompensas_partida.py`, ~6 lineas + 1 test):
```
PHI_RANK_GAP:   10.0 -> 26.0     # una mano completa de puntos, escala natural
PHI_RANK_ALPHA: 0.5              # nuevo
potencial() con PHI_RANK=True -> alpha * _potencial_rank(GAP=26) + (1-alpha) * (rama Phi_media actual)
```
Test unitario: `mix([0,0,0,0]) == 0` y monotonia en score propio.

**Por que estos numeros:** GAP=26 → cobertura de rival cercano ~93-95%, solo 5-7% de estados totalmente planos (vs 34% con g10). El mix elimina las manos muertas por completo: ΔΦ==0 = **0.0%** en el dataset canonico (mas fuerte de lo declarado originalmente — el "piso 9.2%" era un artefacto de pipeline). mean|ΔΦ| = 0.081 (vs 0.154 g10, 0.043 media pura): señal intermedia, sin zona muerta. GAP-schedule creciente: descartado con datos (GAP=26 fijo ya cubre 88-97% en todas las manos). PBRS garantiza invarianza de politica para cualquier Φ — cero riesgo de correccion.

**Honestidad sobre la frontera:** la pendiente real del mix en gap~0 es ~0.0037/pt/rival, **~4x mas suave** que g10 puro (no 2.6x como decia el borrador). Mismo signo y ordenamiento — la correccion de frontera 2/3 sobrevive cualitativamente pero atenuada. Si esa correccion era el motivo principal de Φ_rank, considerar α=0.6-0.7 como variante; reportar la pendiente efectiva real en el doc del cambio.

**Spec del A/B** (el re-test que Φ_rank tiene pendiente desde su kill injusto):
- Run corto 5M, receta v12 identica salvo el potencial. Brazo A: Φ_media (actual). Brazo B: mix α=0.5/GAP=26.
- Gates: curva vs-experto contra referencias v10b/v10c/v12 a pasos iguales; `vf_explained_var` y entropy (requiere A1 aplicado ANTES).
- Abort del brazo: entropy < 0.32 antes de 3M o vf_ev en caida.
- Gates de kill calibrados con las referencias reales de la seccion 1 — no repetir el error que mato al Run A.
- Un factor por vez: este run NO lleva λ=0.98 ni use_kl_loss=False.

---

## 4. Que NO tocar (la evidencia dice que ya esta bien)

1. **El run v12_scratch vivo**: va EN trayectoria v10c progress-matched en vs_experto, manos_por_partida y estructura de reward. No matar, no editar callbacks/config en caliente. Solo los gates de la seccion 1.
2. **`vf_clip_param=10.0` (default)**: matematicamente inerte a nuestra escala — el clip es sobre el error CUADRATICO (umbral efectivo |δ| > √10 ≈ 3.2) y con targets exactos en [-1.5, +1.5] el peor error posible es 9 < 10. NO "arreglarlo" bajandolo a ~1: eso SI recortaria el VF justo en los episodios decididos al final. Documentar en el docstring de `build_ppo_config` con la semantica correcta (clip sobre (V-target)²).
3. **`baza_reward_weight=0.15`**: obsoleto v9, ningun codigo lo consume — es metadato del argparse. No "reconectarlo".
4. **LSTM**: cerrado por costo (re-test justo = BC recurrente inexistente + 40M). El sustituto es B1 (obs 332). No relanzar, no citar como refutado.
5. **`fcnet_hiddens [512,512,256]`**: sin explorar ≠ refutado. Solo tocar si el run 332 se estanca — features primero, capacidad despues.
6. **KL adaptativo y entropy_coeff 0.02**: no tocar a ciegas; ambos son condicionales de A4, decision con la telemetria de A1.
7. **Refutados que siguen refutados**: fine-tunes del campeon (4x), desde-cero estilo v11 (mezcla mala). No reaparecen en este plan.

**Orden de ejecucion sugerido:** hoy → A1 + D (codigo frio + docs). Al terminar/evaluar v12_scratch → run A/B de 5M para A2 (Φ mix), luego run de 5M para A3 (λ) con el Φ ganador. El proximo desde-cero grande → B1+C1+C2 en un solo salto a dim 332, con las decisiones A4 ya informadas por la telemetria nueva.