# Diseño: modelos aprendidos de moon_prob (reemplazo de la heurística)

## Contexto y motivación

`moon_prob_agente`/`moon_prob_rival` son 2 features escalares del vector de
observación (`obs[187:189]`) calculadas hoy por una fórmula heurística de
coeficientes fijos, **duplicada en 3 lugares**:

- `src/entorno/corazones_rllib.py::_calcular_moon_prob` (entrenamiento, con
  manos rivales VERDADERAS — self-play tiene información perfecta)
- `scripts/recomendador.py::_moon_prob` (producción — desde la sesión
  anterior, ya corregido para promediar sobre mundos determinizados en vez
  de una mano rival ficticia fija)
- `scripts/pimc_regret.py::_moon_prob` (evaluación offline)

Un análisis de regret sobre partidas reales (`scripts/pimc_regret_real.py`,
4767 decisiones de 483 manos reconstruibles) mostró que el campeón v10c
comete errores de cola gorda en partidas reales (11.4% de decisiones con
regret ≥3 puntos, hasta 24 puntos en el peor caso) muy por encima de lo
medido en evaluación simulada (0.572 regret medio). Los pases (16.8% de
507/3009 manos reales terminan en pozo, repartido parejo entre asientos) son
un sospechoso plausible. Este spec reemplaza la fórmula por 2 modelos
pequeños aprendidos de partidas reales, con el objetivo de estimar
moon_prob con más precisión que un Monte Carlo ciego a patrones de
comportamiento.

## Datos y etiquetas

- Fuente: las 483 manos reconstruibles de `data/partidas_bridge.jsonl`
  (importadas con `scripts/importar_sesiones_bridge.py`, replay vía
  `src/captura/replay.py`).
- Etiqueta: el campo `lunaSeat` que el bridge ya registra por mano (verdad
  terreno — qué asiento, si alguno, completó el pozo).
- Cada mano reconstruible se re-juega generando un snapshot **por baza
  resuelta**, desde **las 4 perspectivas** (no solo la del agente real de
  esa sesión) — así cada mano aporta ~13 puntos de decisión × 4
  perspectivas, multiplicando el dataset efectivo a un estimado de
  15-20K ejemplos por modelo.
- Gate duro (no aprendido): si cualquier otro jugador ya capturó puntos en
  la mano, el pozo del objetivo es 0 exacto — es una regla del juego. Los
  snapshots donde el objetivo ya no está "vivo" para el pozo se excluyen del
  dataset de entrenamiento (no aportan señal).
- Split train/val: por **partida completa** (`partida_id`), no por mano ni
  snapshot, para evitar fuga de información entre snapshots de la misma
  mano/sesión. Además, reservar las sesiones **cronológicamente más
  recientes** como validación adicional, para detectar sobreajuste a
  patrones de oponentes de esos días específicos en vez de generalización.

## Modelo A — "mi propio pozo"

Evaluado siempre desde la perspectiva real del agente (mano exacta
conocida). Reutiliza el vector completo de `ObservacionBuilder(dim=332)`
desde la perspectiva propia (mano, mesa, cementerio, vacíos, corazones
rotos, baza actual, memoria de pase v13 — todo ya implementado), **con los
2 slots de moon_prob `[187:189]` puestos en cero** (son el objetivo a
predecir, no pueden ser también entrada — evita circularidad).

Además, un feature bonus explícito (no trivialmente recuperable del one-hot
crudo sin cómputo extra): razón bazas-con-puntos-que-gané / bazas-con-puntos
-jugadas-hasta-ahora (requiere historial baza-por-baza, ver más abajo).

Se descartan explícitamente como features separadas: "corazones altos",
"control Q♠", "cartas altas restantes por palo" — todas recuperables por la
red directamente de mano+cementerio; agregarlas aparte sería redundante con
el pedido explícito de que el modelo aprenda de las 13 cartas reales, no de
un resumen hecho a mano.

## Modelo B — "¿un rival específico está armando el pozo?"

Evaluado SOLO con información pública sobre un rival específico j (nunca su
mano real — cualquier feature que requiera conocerla filtraría información
imposible de tener en producción real, y se descarta explícitamente por esa
razón).

Features:

- Cartas capturadas por j hasta ahora (52 one-hot exacto — no solo conteo)
- Cementerio global (52 one-hot, compartido con el Modelo A)
- Mesa actual (52 one-hot, la baza en curso hasta el momento)
- Vacíos de j por palo (4 bits)
- Razón bazas-con-puntos-ganadas-por-j / bazas-con-puntos-jugadas (requiere
  historial baza-por-baza: quién ganó cada una y si tenía puntos; 0/0 → 0.0
  por convención, ninguna baza con puntos jugada aún no es señal de nada)
- **Veces que j lideró con corazón o Q♠ pudiendo evitarlo** (señal de
  intención más fuerte que solo "ganó la baza" — liderar con corazones sin
  necesidad es deliberado)
- Es líder/ganador provisional de la baza en curso (bool)
- Baza actual, corazones rotos, posición relativa de j (izq/frente/der)
- Memoria de pase relativa a j (ver siguiente sección) — cero si no aplica

## Integración de la información del pase

El pase da información EXACTA (no incertidumbre), así que no amerita un
tercer modelo — se integra como features adicionales, reutilizando la
lógica ya implementada en `ObservacionBuilder._construir_bloque_v13`
(nunca antes entrenada en v10c porque el campeón usa 228 dims, no 332):

- **Modelo A**: ya viene incluido al reutilizar el vector v13 completo
  (cartas dadas al receptor sin jugar aún, cartas recibidas del dador que
  aún tengo en mano).
- **Modelo B**, relativo al rival j específico evaluado:
  - si j es mi receptor: "cartas que le di y aún no se han jugado" (52
    one-hot) — sé exactamente que están/estuvieron en su mano.
  - si j es mi dador: "cartas que recibí de él" (52 one-hot) — sé
    exactamente que ya NO las tiene.
  - si j es el 4º jugador (sin relación de pase conmigo esa mano): ambos
    bloques en cero, sin fuga.

## Hallazgo de integración: falta un endpoint en el servidor de inferencia

Para la razón bazas-con-puntos y la memoria de pase en producción real, se
necesita: (a) un historial baza-por-baza dentro de la mano en curso, y (b)
las cartas REALES dadas/recibidas en el pase. Ninguno de los dos existe hoy
fuera de la sesión de replay:

- El historial baza-por-baza se agrega como estado interno nuevo en
  `Recomendador` (una lista por mano, actualizada en `registrar_baza`/
  `registrar_resto`) y en el entorno de entrenamiento (`corazones_rllib.py`).
- **`servidor_inferencia.py` no tiene forma de que el bridge reporte qué se
  pasó realmente** — `/recomendar_pase` solo sugiere, nunca se registra la
  ejecución real (a diferencia de `/registrar_baza` para las jugadas). Se
  necesita un nuevo endpoint `/registrar_pase` (`{"cartas_dadas": [...],
  "cartas_recibidas": [...]}`) para que la memoria de pase funcione en
  partidas reales, no solo en el replay de entrenamiento.

## Arquitectura del código

Nuevo módulo `src/entorno/moon_model.py` (mismo nivel que
`observacion.py`/`dimensiones.py`, mismo patrón SSOT):

- Extracción de features pura y determinista para A y B.
- Clase de red: MLP chico en PyTorch (dependencia ya existente en el
  proyecto, evita sumar scikit-learn), ~entrada(≈224-330) → 32 → 1 con
  sigmoid, uno por modelo.
- Carga/caché de pesos entrenados (`models/moon/propio.pt`,
  `models/moon/rival.pt`).
- Función pública única `moon_prob(motor, idx, agente_idx, vacios,
  historial_bazas, cartas_pasadas, cartas_recibidas, ...) -> float` que
  reemplaza las 3 implementaciones duplicadas de la heurística
  (`corazones_rllib.py`, `recomendador.py`, `pimc_regret.py`).

## Entrenamiento

Script `scripts/entrenar_moon_prob.py`: genera el dataset descrito arriba,
entrena ambas redes (BCE loss, class weighting opcional dado el ~16.8% de
positivos), early stopping por AUC/Brier en validación, imprime métricas.

## Validación (3 pasos)

1. **Offline**: AUC-ROC y Brier score (no solo accuracy — el valor entra
   como probabilidad continua a la red principal, la calibración importa
   más que el umbral) en el split de validación, incluyendo el corte
   cronológico.
2. **Backtest de regret**: reemplazar el Monte Carlo actual en
   `recomendador.py` por los modelos nuevos y re-correr
   `scripts/pimc_regret_real.py` sobre las mismas 483 manos, comparando
   regret medio y la cola (≥3/≥5/≥10 pts) antes/después.
3. **Re-evaluación simulada** (tras Fase 2): comparar contra el baseline
   documentado de v10c (win 0.76, top2 0.932, puesto 1.39-1.7 según la
   tabla de oponentes) con `scripts/comparar_snapshots.py` antes de
   promover un nuevo checkpoint a `models/produccion/`.

## Reentrenamiento (Fase 2)

Fine-tune (no desde cero) de `models/produccion/v10c_campeon`: se conecta
`moon_prob()` en `corazones_rllib.py._build_obs` en lugar de
`_calcular_moon_prob`, manteniendo `obs_dim=228` — el modelo aprendido solo
mejora el VALOR de los 2 escalares que ya existían en esa posición, no el
tamaño de entrada de la red principal (expandir a 332 para exponer memoria
de pase cruda a la política completa sería una reestructuración mayor,
fuera de alcance de este spec). LR bajo (siguiendo la convención del
proyecto, ~1e-4→5e-5), pocos pasos (mucho menos que el entrenamiento
original de 20M), preservando `elite/` y el pool diverso ya existentes.

## Fases

- **Fase 1** (este plan de implementación): extracción de features, las 2
  redes, script de entrenamiento, validación offline + backtest de regret,
  endpoint `/registrar_pase`, wiring en `recomendador.py`. Ya útil de
  inmediato en partidas reales, sin tocar el entrenamiento de v10c.
- **Fase 2** (plan separado, después de validar Fase 1): wiring en
  `corazones_rllib.py`, fine-tune de v10c, re-evaluación, promoción a
  producción.

## Testing

- Tests unitarios de extracción de features (determinismo: mismo
  motor+contexto → mismo vector), mismo estilo que `test_recomendador.py`.
- Test del nuevo endpoint `/registrar_pase`, mismo estilo que
  `test_servidor_inferencia.py`.
- Autocheck `--demo`/`__main__` en `entrenar_moon_prob.py` (convención del
  proyecto: toda lógica no trivial deja un check ejecutable).
