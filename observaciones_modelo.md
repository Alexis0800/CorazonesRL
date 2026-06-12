# Observaciones del Modelo RL — Partidas Interactivas

Registro de hallazgos durante partidas interactivas con `asesor_partida.py`.
Modelo evaluado: `modelos_historicos/v2/modelo_final.zip` (MaskablePPO, ~86% win rate, 8M steps).
VecNormalize: `vecnormalize/v2_vecnorm_final.pkl`.

---

## Caso 1: Q♠ recomendada con corazones rotos (⚠️ Error del modelo)

**Fecha:** 2026-06-11

### Estado de la partida

```
BAZA 3 de 13  │  💔 SÍ (corazones rotos)
Histórico: [VOS:10] [J1:13] [J2:28] [J3:1]
Palo salida: picas (9s liderada por J2)

TU MANO:
  d: 2d
  s: 2s  Js  Qs  Ks  As
  h: 6h  8h  Qh  Kh  Ah

Legales: 2s  Js  Qs  Ks  As
```

### Recomendación del modelo

**Qs** (Q♠ = 13 puntos).

### Por qué es incorrecta

1. **No se puede hacer pozo** — corazones ya rotos ($\text{💔} = \text{SÍ}$). El pleno es imposible.
2. En modo *"minimizar puntos"*, Q♠ es la peor carta para tirar (13 pts).
3. **Alternativas con 0 puntos y mismo control:**
   - `As` → gana la baza (14 > 9), 0 pts, lidera la siguiente.
   - `Ks` → gana la baza (13 > 9), 0 pts, lidera la siguiente.
   - `Js` → gana la baza (11 > 9), 0 pts, lidera la siguiente.
4. Tirar Q♠ en esta baza son **13 puntos gratis regalados** sin beneficio táctico alguno.

### Causa probable

El modelo aprendió el heurístico *"sacate la Q♠ cuando la mesa está limpia de corazones"* como estrategia general. Este heurístico es correcto cuando el pozo todavía es posible (corazones no rotos), pero el modelo **no aprendió a cambiar de estrategia** cuando `corazones_rotos=True`.

### Principio a corregir

> Cuando `corazones_rotos = True` y el pozo es imposible, el modelo debe priorizar **no llevarse puntos**, no "sacarse la Q♠". La Q♠ son 13 puntos y debe tratarse como una carta de alto riesgo, no como una carta para descartar rápido.

### Soluciones propuestas

#### A. Penalización por Q♠ con corazones rotos (recomendado)

```python
# En step() del entorno, después de resolver baza:
if corazones_rotos and carta_jugada == Q_SPADES_ID:
    reward -= 5.0  # Penalización por Q♠ innecesaria
```

#### B. Flag de pozo posible en la observación

```python
pozo_posible = not corazones_rotos or puntos_mano[agente] >= 13
```

#### C. Reward shaping por puntos en mano

```python
reward -= 0.1 * puntos_mano[agente]
```

### Prioridad

| # | Solución | Esfuerzo | Impacto |
|---|----------|----------|---------|
| A | Penalización Q♠ con 💔 | Bajo | Alto |
| B | Flag `pozo_posible` | Bajo | Medio |
| C | Reward por puntos en mano | Medio | Medio |

---

## Caso 2: Singleton de picas en cierre de mano (✅ Acierto del modelo)

**Fecha:** 2026-06-11

### Estado de la partida (bazas 11-13)

```
BAZA 11 de 13  │  💔 SÍ
Histórico: [VOS:81] [J1:30] [J2:53] [J3:19]
👑 Q♠: J2 (ya fue jugada, no hay puntos de Q♠ en juego)

Voids:
  VOS: d
  J1:  h
  J2:  c s            ← J2 no tiene tréboles NI picas
  J3:  c               ← J3 no tiene tréboles

TU MANO (3 cartas):
  c: 7c    Qc
  s: 2s
```

### Recomendación del modelo

**2s** ⭐ (liderar con el singleton de picas).

### Qué pasó

| Baza | Lideraste | Jugaron | Ganador | Pts |
|---|---|---|---|---|
| 11 | `2s` | J1:`8c` J2:`Jd` J3:`6d` | VOS | **0** |
| 12 | `7c` | J1:`10d` J2:`9d` J3:`Ad` | VOS | **0** |
| 13 | `Qc` | J1:`7d` J2:`8d` J3:`Kd` | VOS | **0** |

**Resultado: 0 puntos en las últimas 3 bazas.** Ganaste las 3 y retuviste el control absoluto del cierre.

### Por qué fue correcta

1. El modelo **trackeó los voids**: J2 y J3 eran void en tréboles. J2 también void en picas.
2. Liderar `2s` era seguro: si alguien tenía picas, vos fugabas trébol (basura de 0 pts). Si nadie tenía, ganabas con `2s` y retenías el liderazgo.
3. Con el liderazgo asegurado en bazas 12 y 13, tus tréboles altos (`Qc`, `7c`) dominaban — los demás solo tenían diamantes, donde eras void.
4. **0 puntos en todo el cierre.** Mano controlada de principio a fin.

### Qué habría pasado si liderabas `7c`

J1 juega trébol (probablemente más bajo que `7c`), J2 fuga porque es void en `c`, J3 juega trébol. **J3 gana con un trébol más alto** (ej. `Ac`, `Kc`, etc. que estaban en juego). Perdés el control de las bazas 12 y 13, y te pueden meter puntos.

### Matiz importante: si hubiera habido puntos en juego

El usuario señaló correctamente que **en este caso específico, era indiferente** quién ganara las bazas 11-13 porque:

- Ya no quedaban corazones en juego (todos jugados en bazas anteriores)
- La Q♠ ya había sido capturada por J2
- Las bazas restantes solo contenían cartas de 0 puntos (tréboles, diamantes, picas bajas)

**Pero si hubieran quedado corazones sin jugar o la Q♠ todavía en juego, el modelo debería priorizar NO llevarse esas bazas.** Ganar bazas cuando contienen puntos es malo. El modelo acertó aquí porque el cementerio le indicó que no quedaban puntos — fue una decisión informada, no ciega.

### Principio reforzado

> Cuando el cementerio indica que **no quedan puntos en juego**, es indiferente quién gane las bazas restantes. El modelo puede liderar agresivamente para retener el control. Pero si **quedan corazones o Q♠ sin jugar**, debe priorizar perder las bazas para no acumular puntos.

---

## Caso 3: Q♠ recomendada sin corazones rotos, pero pozo imposible por mano insuficiente (⚠️ Error del modelo)

**Fecha:** 2026-06-11

### Estado de la partida

```
BAZA 3 de 13  │  💔 NO (corazones NO rotos)
Histórico: [VOS:87] [J1:34] [J2:67] [J3:20]

Mesa: 10s (J1)  7s (J2)  9s (J3)
Palo salida: picas (J1 lideró con 10s)

TU MANO (11 cartas):
  d: 2d   5d
  s: 2s   4s   8s   Js   Qs
  h: 3h   4h   Jh   Ah          ← solo 4 corazones, 2 bajos (3h, 4h)

Legales (obligado a seguir picas): 2s  4s  8s  Js  Qs
```

### Recomendación del modelo

**Qs** ⭐ (Q♠ = 13 puntos).

### Por qué es incorrecta

1. **Pozo imposible.** Aunque 💔 = NO, el pozo requiere ganar **los 13 corazones + Q♠**. El agente solo tiene **4 corazones**, de los cuales dos (3h, 4h) son tan bajos que no pueden ganar ninguna baza. Los otros 9 corazones están repartidos entre 3 rivales. Es matemáticamente imposible capturarlos todos.

2. **Puntuación en zona de peligro.** VOS ya tiene 87 puntos históricos. Q♠ = 13 puntos → 87 + 13 = 100, justo en el umbral de fin de partida. Cualquier punto adicional en esta mano te deja fuera.

3. **Hay alternativas con 0 puntos:**
   - `2s` → pierde contra 10s, 0 pts, Q♠ se guarda para tirarla en baza de otro.
   - `4s` → ídem.
   - `8s` → ídem.

4. **Rivales inteligentes frenan el pozo.** Si jugás contra humanos o modelos decentes, apenas detectan que te llevaste la Q♠ en baza 3, van a hacer todo lo posible por evitar que captures corazones — tirando corazones altos cuando no puedas ganarlos, por ejemplo.

### Patrón sistemático detectado

Los **Casos 1 y 3** comparten la misma raíz:

> El modelo aprendió el heurístico *"mesa limpia de corazones → tirar Q♠"* como regla general, **sin evaluar si el pozo es realmente alcanzable**.

Disparadores de pozo que el modelo NO está evaluando:

| Condición | Caso 1 | Caso 3 |
|---|---|---|
| ¿💔 roto? | SÍ → pozo imposible | NO → pozo teóricamente posible |
| ¿Corazones altos suficientes? | No evalúa | 2 de 4 son bajos → pozo imposible |
| ¿Cantidad de corazones en mano? | No evalúa | Solo 4 de 13 → pozo imposible |
| ¿Puntuación actual? | No evalúa | 87 pts → Q♠ = 100, al borde |
| ¿Rivales pueden frenar? | No evalúa | Humanos/modelos lo harían |

### ¿Cuándo SÍ conviene tirar Q♠ en mesa limpia?

Solo cuando se cumplan **todas** estas condiciones:

1. 💔 = NO (corazones no rotos).
2. **Mayoría de corazones altos en mano** (al menos 6-7 corazones con J, Q, K, A).
3. Puntuación actual baja (Q♠ no te acerca a 100).
4. Rivales no tienen corazones altos suficientes para frenarte (difícil de saber).

### Solución propuesta adicional

Agregar en la observación un feature que resuma la **viabilidad del pozo**:

```python
# En construir_observacion():
corazones_en_mano = sum(1 for cid in mano if _PALOS[cid] == 3)
corazones_altos = sum(1 for cid in mano
                      if _PALOS[cid] == 3 and _VALORES[cid] >= 11)  # J, Q, K, A

pozo_viable = (
    not corazones_rotos
    and corazones_en_mano >= 6
    and corazones_altos >= 3
    and puntaje_historico[agente] < 80
)
```

---

## Resumen actualizado

| # | Caso | Recomendación | ¿Correcta? | Causa raíz |
|---|---|---|---|---|
| 1 | Q♠ con 💔 roto | Tirar Q♠ (13 pts) | ❌ | No distingue modo pozo vs minimizar |
| 2 | Singleton picas cierre | Liderar 2s (0 pts) | ✅ | Trackeo de voids + cementerio |
| 3 | Q♠ sin 💔, mano débil | Tirar Q♠ (13 pts) | ❌ | No evalúa viabilidad real del pozo |

---

**Archivo creado:** 2026-06-11
**Suite de tests:** 204/204 ✅ al momento del registro
