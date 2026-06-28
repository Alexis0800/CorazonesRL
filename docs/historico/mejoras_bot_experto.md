# Mejoras al BotExperto — Análisis y Cambios

## Metodología

Todas las mejoras fueron identificadas y validadas usando **PIMC** (Perfect Information Monte Carlo) como juez imparcial. Para cada decisión no trivial del bot (≥2 cartas legales), PIMC simula el fin de la mano en ~30 mundos posibles con diferentes distribuciones de cartas, calcula el puntaje esperado de cada carta posible, y marca **divergencia** cuando el bot eligió peor que el óptimo PIMC.

```
python scripts/analizar_errores_bot.py --partidas 200 --mundos 30 --seed 42
```

La métrica clave es **coste medio por mano** (pts de penalización extra esperados por elegir subóptimamente).

### Línea base

| Run | Partidas | Pts/mano | Divergencias |
|-----|----------|----------|--------------|
| Original (sin cambios) | 500 | ~9.5 pts | 35.7% decisiones |
| **Versión mejorada** | **500** | **~8.6 pts** | **~32.5%** |

Reducción: **≈10% de pts/mano en condiciones de juego real**.

---

## Cambios aplicados

### 1. `_liderar` — Threshold de quema de palos: baza 10 → baza 6

**Qué hace el código original:**  
El bot quemaba sus máximas de ♣/♦ (liderar su carta más alta cuando es la ganadora del palo) recién desde la baza 10.

**El problema:**  
Quedaban palos "bloqueados" en mano demasiado tiempo. En bazas 6–9 ya era habitual que algún rival estuviera void en ♣ o ♦, lo que significa que al liderar esas cartas en bazas tardías el bot acumulaba descartados de corazones.

**El cambio:**
```python
# Antes
if baza >= 10:
    ...quemar maximas de ♣/♦...

# Después
if baza >= 6:
    ...quemar maximas de ♣/♦...
```

**Por qué funciona:**  
Liderar la máxima de un palo seguro garantiza ganar una baza limpia (0 pts) en un momento donde todavía hay cartas de ese palo en manos rivales, antes de que se generen los voids que convierten esa misma jugada en un imán de corazones.

---

### 2. `_liderar` — Dump de Q♠ cuando K♠/A♠ siguen en circulación

**Qué hace el código original:**  
El bot nunca lidera con Q♠ si tiene alternativa.

**El problema:**  
En bazas tardías (≥7) con K♠ o A♠ todavía en manos rivales, liderar Q♠ es una jugada ganadora: el poseedor de K♠/A♠ **debe** seguir en ♠ y cubrirá Q♠, llevándose los 13 pts. Cargar Q♠ hasta el final obliga al bot a deshacerse de ella en condiciones mucho peores.

El PIMC detectó casos con coste 13–21 pts por no liderar Q♠ en el momento adecuado.

**El cambio:**
```python
# Nueva regla, ANTES del bloque "nunca liderar Q♠"
mi_q_en_mano = [c for c in legales if c.es_dama_de_picas]
if mi_q_en_mano and baza >= 7 and not self._tengo_maxima_del_palo(motor, idx, _PICA):
    alguno_puede_seguir_picas = any(
        _PICA not in self._vacios[i] for i in range(4) if i != idx
    )
    if alguno_puede_seguir_picas:
        return mi_q_en_mano[0]
```

**Condiciones requeridas (todas deben cumplirse):**

| Condición | Razón |
|-----------|-------|
| `baza >= 7` | En bazas tempranas conviene reservar Q♠ para un dump más controlado |
| `not _tengo_maxima_del_palo(♠)` | Si soy la maxima de ♠ (nadie puede cubrirme), liderarla me da los 13 pts |
| `alguno_puede_seguir_picas` | Si todos los rivales son void confirmados en ♠, Q♠ ganaría la baza y acumularíamos los 13 pts |

**Por qué funciona:**  
`_tengo_maxima_del_palo` comprueba si existe algún ♠ mayor que el mío en manos rivales. Si existe K♠ o A♠ en circulación, esa carta cubrirá Q♠ cuando la lideremos. La verificación de voids confirmados (`_vacios`) descarta el caso borde donde todos han desechado sus ♠.

---

### 3. `_seguir_palo` — Baza sin puntos: quemar la carta alta que no gana

**Qué hace el código original:**  
En bazas sin puntos en la mesa, el bot seguía el palo con la carta **más baja** posible.

**El problema:**  
Si la carta ganadora actual es, por ejemplo, K♦ (13) y el bot tiene [3♦, 9♦], el original jugaba 3♦. Pero 3♦ y 9♦ son igualmente inofensivos para **esta** baza. La diferencia está en el futuro: 9♦ puede ganar bazas de ♦ donde se descarten corazones, mientras que 3♦ raramente lo hará.

**El cambio:**
```python
# Antes: return min(mismo_palo, key=lambda c: c.valor)

# Después (baza sin puntos, ganadora conocida)
if ganadora is not None:
    no_gana = [c for c in mismo_palo if c.valor <= ganadora.valor]
    if no_gana:
        return max(no_gana, key=lambda c: c.valor)  # la más alta que pierde
return min(mismo_palo, key=lambda c: c.valor)
```

**Por qué funciona:**  
Si puedo perder la baza de todas formas (mis cartas ≤ la ganadora actual), el resultado de **esta** baza es el mismo sea cual sea la carta que juegue. Pero eliminar la carta más alta de las perdedoras reduce la probabilidad de ganar bazas futuras donde los rivales descarten corazones. El fallback a `min` se aplica cuando todas mis cartas ganarían la baza.

---

### 4. `_descartar` (baza sin puntos) — Q♠: descarte anticipado en baza ≥6

**Qué hace el código original:**  
En bazas sin puntos en la mesa, Q♠ nunca se descartaba (se guardaba para bazas con puntos).

**El problema:**  
En bazas 6–12, cada baza que pasa reduce las oportunidades de encontrar una baza con puntos donde descargar Q♠ al jugador objetivo. La herramienta PIMC detectó costes de 10–13 pts por no descartar Q♠ en bazas 6–7.

**El cambio:**
```python
# Baza sin puntos, nuevo bloque
if q and baza >= 6 and modo != "POZO":
    return q[0]
```

**Por qué funciona:**  
Si la baza no tiene puntos, descargar Q♠ da 13 pts al ganador de esa baza (malo para él), pero libera al bot de la mayor liability del juego. En modo POZO se omite porque Q♠ puede ser necesaria para completar el shooting the moon.

**El test que valida el límite:**  
`test_descarta_carta_sin_puntos_si_no_hay_urgencia` (baza 2) confirma que en bazas muy tempranas aún se prefiere no descargar Q♠.

---

### 5. `_descartar` (baza sin puntos) — K♠/A♠ con Q♠ activa

**Qué hace el código original:**  
K♠ y A♠ se trataban como cartas sin puntos normales (0 pts) y se descartaban solo al final, después de los corazones altos.

**El problema:**  
Si Q♠ todavía está en circulación y alguien lidera ♠, el bot con K♠ o A♠ **debe seguir** en ♠ y gana la baza. Al ganar esa baza, el bot convierte a K♠/A♠ en el imán de Q♠: si quien tiene Q♠ es void en ♠, la descartará en esa baza → el bot recibe 13 pts de golpe.

**El cambio:**
```python
if self._q_activa(motor):
    picas_altas = [c for c in legales
                   if c.palo == _PICA and c.valor > 12 and not c.es_dama_de_picas]
    if picas_altas:
        return max(picas_altas, key=lambda c: c.valor)
```

**Por qué funciona:**  
Descargar K♠/A♠ en una baza limpia (0 pts) elimina el riesgo de ganar bazas de ♠ donde Q♠ podría ser descartada. La condición `_q_activa` limita el bloque al período donde Q♠ sigue siendo un peligro real.

---

### 6. `_descartar` (baza sin puntos) — A♣/A♦ antes que corazones bajos

**Qué hace el código original:**  
Los ases de palos seguros (A♣, A♦) se descartaban como cualquier carta sin puntos, después de corazones altos pero con igual prioridad que otras cartas sin puntos.

**El problema:**  
A♣ y A♦ siempre ganan bazas de su palo. Si un rival es void en ♣ o ♦ y el bot lidera con A♣/A♦, ese rival descartará corazones en esa baza → el bot acumula los puntos. El PIMC detectó casos de coste ~8 pts por retener ases seguros sobre corazones bajos.

**El cambio:**
```python
ases_seguros = [c for c in sin_puntos if c.valor == 14]
if ases_seguros:
    return ases_seguros[0]
```

**Por qué funciona:**  
Un as de ♣ o ♦ solo vale 0 pts si lo pierde alguien más; en manos del bot, garantiza ganar bazas donde acumular descartados. Liberarlo en una baza limpia elimina esa amenaza sin coste inmediato.

---

### 7. `_descartar` (baza sin puntos) — Corazones bajos cuando `corazones_rotos`

**Qué hace el código original:**  
Los corazones bajos (2♥–10♥) no recibían tratamiento especial; se descartaban después de las cartas sin puntos solo como último recurso.

**El problema:**  
Una vez que los corazones están rotos, cualquier rival puede liderarlos. Un corazón bajo en mano podría convertirse en ganador de una baza de ♥ donde ya existen descartados de corazones más altos, o puede quedarse como liability que se acumula con el tiempo. El PIMC detectó costes de ~8–18 pts por retener corazones bajos sobre cartas sin puntos en bazas tardías con corazones rotos.

**El cambio:**
```python
if corazones_todos and motor.corazones_rotos:
    return corazones_todos[0]  # el corazón más alto disponible
```

**Por qué funciona:**  
Con corazones rotos, las bazas de ♥ son comunes. Dar un corazón bajo al ganador de una baza limpia cuesta 1 pt al ganador pero libera al bot de una carta que podría forzarlo a ganar bazas de ♥ con puntos en el futuro. Las cartas sin puntos (♣/♦ sin ser ases) raramente crean ese riesgo.

**Condición de guarda en test:**  
`test_descarta_carta_sin_puntos_si_no_hay_urgencia` (baza 2, `corazones_rotos=False`) confirma que la regla no aplica cuando los corazones no han sido rotos, situación donde conservar corazones bajos es preferible.

---

## Orden de prioridad final en `_descartar` (baza sin puntos)

```
1. Q♠          — si baza ≥6 y no modo POZO
2. K♠ / A♠     — si Q♠ sigue activa en el juego
3. J/Q/K/A♥    — corazones altos (≥J), siempre peligrosos
4. A♣ / A♦     — ases de palos seguros (ganan su palo)
5. 2–10♥       — corazones bajos, solo si corazones_rotos=True
6. carta más alta sin puntos  — liberar mano
7. cualquier corazón bajo     — si corazones_rotos=False
8. Q♠ (sin umbral)            — último recurso
```

---

## Errores residuales (no corregidos)

Después de los cambios, el PIMC sigue detectando dos patrones que no se pudieron corregir con reglas simples sin provocar regresiones:

### Liderar corazones en bazas tardías

El PIMC recomienda liderar corazones medios o altos (7♥–K♥) en bazas 9–12 cuando el rival tiene voids confirmados en los palos seguros. Intentar esta regla de forma general causó +71 errores nuevos en `_liderar` porque la elección del corazón exacto a liderar depende del estado completo de la partida.

### Ganar bazas sin puntos para controlar el lead

En bazas finales (11–13), el PIMC ocasionalmente recomienda ganar una baza sin puntos (jugando la carta más alta del palo) para controlar el lead de las últimas bazas. La regla `max(mismo_palo)` vs `min(mismo_palo)` depende del estado exacto de manos restantes y no puede generalizarse sin simulación.

---

## Archivos modificados

| Archivo | Líneas afectadas | Cambio |
|---------|-----------------|--------|
| `src/agentes/bot_experto.py` | `_liderar` ~308–343 | Q♠ dump + threshold baza 6 |
| `src/agentes/bot_experto.py` | `_seguir_palo` ~378–386 | Mayor no-ganadora en baza sin puntos |
| `src/agentes/bot_experto.py` | `_descartar` ~436–474 | Orden completo de prioridades |

Todos los cambios son compatibles con los **418 tests existentes**.

```bash
python -m pytest tests/ -q  # 418 passed
```
