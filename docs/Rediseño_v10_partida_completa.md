# Rediseño v10 — Partida completa + recompensa basada en potencial (PBRS)

> Estado: **v10a implementado y testeado** (partida completa sin pase). v10b (pase) pendiente.
> Reemplaza el MDP de "una mano aislada" de v5–v9.
> Decisiones del usuario: objetivo = *ganar > top-2*; alcance = *partida completa primero (v10a), pase después (v10b)*.

## Estado de implementación (v10a)

| Componente | Archivo | Estado |
|---|---|---|
| Bucle de partida + ranking | `src/dominio/motor.py` (`nueva_partida`, `partida_terminada`, `ranking_partida`, `puntuaciones_historicas`) | ✅ + tests |
| Recompensa R_terminal + PBRS | `src/entorno/recompensas_partida.py` | ✅ + tests (telescopaje) |
| Env partida completa | `src/entorno/corazones_rllib.py` | ✅ + tests |
| Observación (224 dims, features de partida ahora VIVAS) | `src/entorno/observacion.py` (sin cambios; revividas) | ✅ |
| Eval vs bots (partidas completas) | `src/rllib/eval_bots.py` | ✅ |
| Config (γ=0.999, reward_config) | `src/rllib/config.py` | ✅ |
| Pipeline + callbacks anti-farming | `train_rllib.py`, `src/rllib/callbacks.py` | ✅ |
| Monitor de avance | `monitorear.py` | ✅ |
| Evaluación final + recomendaciones | `evaluar_final.py` | ✅ |

Comandos:
```bash
# Entrenar v10a (MLP; LSTM opcional con --use-lstm)
python scripts/train_rllib.py --total-steps 20000000 --workers 8 --output-dir models/v10

# Ver avance en vivo (otra terminal)
python scripts/monitorear.py --dir models/v10 --watch

# Evaluación final con recomendaciones
python scripts/evaluar_final.py --dir models/v10 --partidas 200
```

> Nota de diseño: la observación se mantiene en 224 dims (DIM_V11). Las features de
> marcador (scores, líder, cerca de 100, mano terminal) que en v9 estaban muertas
> (siempre 0) **cobran vida** al persistir el marcador entre manos — no se inventan
> dims nuevas para evitar churn de arquitectura sin ganancia de señal.

---

## 0. Por qué este rediseño

v9 se estancó temprano (plano en `bot_eval` durante 10M de steps; pierde contra `BotExperto`). Causa raíz: **el episodio era una sola mano con el marcador reiniciado a `[0,0,0,0]`**. Consecuencias:

- El objetivo real optimizado era "minimizar mis puntos en una mano aislada" → techo bajo que un heurístico ya alcanza.
- Features de marcador (scores históricos, líder, cerca de 100, `puedo_alimentar`) **muertas** (siempre 0).
- `recompensa_fin_partida` (±500) y las ~20 funciones tácticas de `CalculadoraRecompensas` → **código muerto** (sin callers).
- LSTM desperdiciada (13 steps/episodio, sin estado entre manos).
- Toda la estrategia de [Estrategias Avanzadas.md](Estrategias%20Avanzadas.md) (atacar al líder, acuerdo no hablado, kingmaker, timing del pozo) es **inexpresable** en ese MDP.

El rediseño cambia el episodio a **partida completa a 100** y reconstruye la recompensa para que el objetivo sea ganar la partida, sin que el agente pueda "farmear" señales intermedias.

---

## 1. MDP

| Elemento | Definición |
|---|---|
| **Episodio** | Una **partida completa**: se juegan manos hasta que, al cerrar una mano, algún jugador ≥ 100 pts. Ganador = menor puntuación acumulada. |
| **Step del agente** | Jugar **una carta** (igual que hoy). Una partida ≈ 8–13 manos × 13 bazas ≈ **100–170 steps**. |
| **Asiento** | `random_position` por partida (el agente aprende las 4 posiciones). |
| **Oponentes** | 3 jugadores del pool de self-play / bots según fase del curriculum, jugando también la partida completa. |
| **γ (descuento)** | **0.999** a nivel de carta (γ^130 ≈ 0.88 → el puesto final propaga bien). PBRS reduce la dependencia de γ. |
| **Longitud variable** | Sí; PPO lo maneja sin problema. |

El estado del marcador **persiste entre manos dentro del episodio** y solo se reinicia al empezar una nueva partida (`reset()`).

---

## 2. Recompensa (núcleo del rediseño)

Tres capas. Solo la primera es objetivo "real"; la segunda es shaping garantizado-no-farmeable; la tercera no existe.

### 2.1 R_terminal — objetivo primario (una sola vez, al fin de partida)

Según puesto final (decisión del usuario: *ganar > top-2*):

| Puesto | Reward |
|---|---|
| 1º | **+1.0** |
| 2º | **+0.3** |
| 3º | **−0.3** |
| 4º | **−1.0** |

Es la **única recompensa que define el óptimo**. Empates de score: desempatar por regla del juego (a definir; p.ej. menor en la última mano, o reparto del valor entre empatados).

### 2.2 Shaping basado en potencial (PBRS) — densifica sin sesgar

Teorema (Ng, Harada & Russell, 1999): si la señal de forma es

```
F(s → s') = γ · Φ(s') − Φ(s)
```

entonces **la política óptima NO cambia** respecto a usar solo R_terminal. La suma de F sobre el episodio telescopia a `−Φ(s₀) + γᵀ·Φ(s_T)`; con `Φ(terminal)=0` y `Φ(s₀)≈0`, **el shaping neto del episodio ≈ 0** → es imposible farmearlo. Solo redistribuye el crédito en el tiempo y acelera el aprendizaje.

**Potencial Φ (sobre el marcador acumulado):**

```
Φ(s) = λ · clip( (media_scores_rivales − mi_score) / 100 , −1, +1 )
Φ(terminal) = 0
```

- `λ` = peso del shaping (arrancar en **0.5**, tunear). Magnitud comparable a R_terminal pero sin dominarlo.
- Mejora tu **margen relativo** en el marcador ⇒ Φ sube.

**Implementación rigurosa (clave):** mantener `Φ` calculable en **cada** step (es función del marcador actual, siempre conocido) y añadir `γ·Φ(s_{t+1}) − Φ(s_t)` en **todos** los steps. Como el marcador solo cambia al cerrar mano, intra-mano el término es ≈0 y el salto real ocurre en el borde de mano. Esto garantiza el telescopaje exacto. (Equivalente práctico: aplicar `γ_mano·Φ' − Φ` solo en bordes de mano; más simple, rompe el telescopaje de forma despreciable.)

**Por qué se auto-alinea con la estrategia humana** (todo emerge de Φ, sin codificar tácticas):

| Situación | Efecto en scores | ΔΦ | Lección aprendida |
|---|---|---|---|
| Tomar pocos puntos en la mano | mi_score sube poco | + | Jugar bien la ronda |
| Defender el pozo de un rival | rival no cierra, sube su score | + | Romper el pozo |
| Hacer yo el pozo | los 3 rivales +26 | ++ | Intentar el pozo cuando conviene |
| Alimentar al líder (cerca de 100) | su score sube | + | Acuerdo no hablado |

### 2.3 Sin recompensa por baza

No hay reward denso por baza/truco. **GAE** propaga el reward de la mano hacia atrás por los 13 trucos. Esto **elimina** las ~20 funciones tácticas hechas a mano (fuente de reward hacking y de la complejidad que querías simplificar). Si tras entrenar persisten errores tácticos concretos, se añaden **como micro-shaping potencial**, nunca como bonus crudo.

### 2.4 Lo que se BORRA

- `recompensa_fin_partida` (±500) y todas las funciones tácticas de `CalculadoraRecompensas` (código muerto).
- La señal por baza `baza_reward_weight` y las señales v9 inline (Q♠ −5, moon hearts, K♠ discard).
- `recompensas_minimal.py` (queda obsoleto).
- Dimensiones de observación muertas (ver §3).

---

## 3. Observación (reescritura limpia, SSOT real)

Se **conserva** lo táctico de la mano (útil para jugar bien la ronda) y se **revive/añade** lo de partida (ahora con señal real). Layout exacto a fijar en implementación; estructura:

**Conservar (táctico de mano):** mano (52), mesa (52), cementerio (52), voids (16), Q♠ tracker, corazones rotos, posición en baza, moon_prob agente/rival, palo de salida, cartas/altas restantes por palo, palo salida, quién jugó en la baza.

**Revivir/añadir (nivel partida — ahora VIVOS):**
- Scores acumulados de los 4 (relativos al agente, /100).
- Distancia a 100 de cada jugador.
- Soy líder / soy colista / mi posición en el ranking (1–4).
- Nº de manos jugadas en la partida / fase de partida.
- Nº de jugadores en zona de peligro (cerca de 100).
- `puedo_alimentar` (rival cerca de 100) — ahora se dispara de verdad.

**Borrar:** cualquier dim constante o redundante. Definir `DIM_V12` en `dimensiones.py` y migrar todo a esa constante (nada hardcodeado).

---

## 4. Cambios en el dominio (`src/dominio/motor.py`)

El motor implementa correctamente las reglas de **una mano**. Faltan:

1. **Bucle de partida.** Tras `aplicar_puntuacion()`, comprobar fin de partida: `max(puntuacion) >= 100`. Si no, repartir nueva mano **preservando el acumulado** (`repartir()` ya conserva `puntuacion`, solo limpia `bazas_ganadas` — verificar). El env conduce mano a mano dentro del mismo episodio.
2. **Ganador / ranking** de partida (menor score; desempate definido).
3. **(v10b) Pase**: fase de selección de 3 cartas con rotación izq/der/enfrente/sin-pase, y **recálculo del portador del 2♣ DESPUÉS del intercambio** (hoy se fija en `repartir()`, hay que moverlo).

Recomendación de capas: añadir orquestación de partida testeable en `dominio` (helper de "siguiente mano" + detección de fin), y que el env la conduzca paso a paso (no un `jugar_partida` bloqueante, porque el env necesita control carta a carta).

---

## 5. Self-play, curriculum y evaluación

- **Curriculum** análogo a las 5 fases actuales pero con episodio = partida: bootstrap vs bots → mix bots+snapshots → self-play puro. Pool de snapshots (cap 50).
- **Evaluación primaria: Elo en torneos de partida completa** entre snapshots + golden baselines (`v7_golden`) + bots. (v9 no logueaba Elo; corregir.)
- **Métrica absoluta:** win-rate y **top-2-rate** en partidas completas vs bots fijos (reemplaza el `bot_eval` por-mano).
- **Chequeo anti-farming:** loguear por separado (a) retorno solo-R_terminal y (b) retorno con shaping; verificar que el ranking Elo correlaciona con R_terminal, no con el shaping. `avg_pts/mano` pasa a ser **diagnóstico**, no objetivo.
- **Métricas de pozo:** tasa de intentos y de defensas de pozo.

---

## 6. Plan de versiones

- **v10a** — partida completa, **sin pase**. Criterio de éxito: superar el Elo de `v7_golden` y ganar > top-2 de forma consistente a 3× `BotExperto` en partidas completas.
- **v10b** — añade la fase de **pase** (rotación direccional + recálculo del 2♣). Criterio: mejora incremental de Elo sobre v10a.

---

## 7. Orden de implementación sugerido

1. **Dominio**: bucle de partida + fin/ranking en `motor` + tests (`tests/dominio/`). Sin RL todavía.
2. **Recompensa**: nueva `RewardConfig`/calculadora minimalista (R_terminal + Φ) + tests del telescopaje PBRS (verificar suma ≈ −Φ(s₀)).
3. **Observación**: `DIM_V12`, builder con features de partida vivas, test de no-constancia.
4. **Env**: `CorazonesEnvRLlib` conduce la partida multi-mano; `reset` solo al fin de partida.
5. **Pipeline**: curriculum de partida, eval Elo de partida, logging anti-farming.
6. Entrenar v10a, validar contra criterios de §6.
7. v10b: pase.
