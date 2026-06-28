# ROADMAP — Próximos pasos

Estado: **v10c** es un agente fuerte y completo (partida completa + pase + self-play
consistente). Lo siguiente es **usarlo y validarlo en el mundo real**, y llevarlo
al móvil. Orden recomendado: de lo más simple/valioso a lo más complejo.

---

## Fase 1 — Validar como copiloto (AHORA, sin código nuevo)

Ya tienes `recomendador.py`. El objetivo es **medir si sus recomendaciones son buenas
en partidas reales** antes de invertir en app.

- [ ] Usar `recomendador.py --modelo models/produccion/v10c_campeon` en partidas reales
      (presencial/online), anotando los casos donde la recomendación te parezca rara.
- [ ] Llevar un registro simple: mano + situación + lo que recomendó + lo que tú
      habrías hecho + resultado. (Sirve como mini-dataset de validación y para depurar.)
- [ ] Si hay patrones de error → ajustar (reward, bots, más entrenamiento) y re-evaluar.

**Entregable:** confianza (o lista de fallos) en las recomendaciones del modelo.

---

## Fase 2 — Dataset de partidas reales vs humanos

Para medir **qué tan eficiente es contra humanos reales** (no solo bots).

- [ ] **Logger de partidas**: extender `recomendador.py` (o un modo nuevo) para que
      GUARDE cada partida jugada (manos, pases, jugadas, resultado) a un `.jsonl`.
      Esto convierte cada partida real en datos.
- [ ] **Métrica de eficiencia humana**: con N partidas reales, medir el puesto medio /
      win-rate del modelo (si jugaste siguiendo sus recomendaciones) vs humanos.
- [ ] **(Opcional) Fine-tune con datos humanos**: si los humanos juegan distinto a
      nuestros bots, añadir un arquetipo "humano" al pool (clonado de los logs) y/o
      hacer un fine-tune ligero. Cierra el gap de distribución sim→real.

**Entregable:** número honesto de "qué tan bueno es vs humanos" + datos para mejorar.

**Nota técnica:** el `BotExperto` y los arquetipos ya cubren mucho del juego humano,
pero los humanos tienen sesgos (errores, estilos). Un dataset real es la única forma
de medir y cerrar ese gap.

---

## Fase 3 — App móvil copiloto (Flutter o Kotlin)

Tu visión: una **burbuja overlay** que lee la pantalla y sugiere la acción. Conviene
construirla en **incrementos**, porque el "leer la pantalla" es la parte difícil.

### 3a. Copiloto manual (lo primero a construir)
La app replica `recomendador.py` pero con UI táctil: tocas las cartas de tu mano,
indicas las jugadas, y te muestra la recomendación. **Sin visión todavía.**

Decisión de arquitectura (importante):
- **Opción A — Servidor (rápido de montar):** la app envía el estado a una API
  (tu PC o un servidor cloud corriendo el modelo Python) y recibe la recomendación.
  Reusa TODO el código Python tal cual (`recomendador.py` → endpoint Flask/FastAPI).
  Requiere conexión. **Recomendado para validar la UX primero.**
- **Opción B — On-device (offline):** exportar el modelo y reimplementar la lógica
  en el móvil (ver Fase 4). Más trabajo; necesario para presencial sin internet.

> La parte que NO es trivial de portar no es el modelo (es un MLP pequeño, ~6 MB),
> sino la **reconstrucción de la observación de 228 dims y el tracking del estado
> de juego** (lo que hace `recomendador.py`: cementerio, voids, marcador, fase de
> pase…). Conviene aislar esa lógica en un "núcleo de inferencia" reutilizable.

### 3b. Overlay con burbuja (Android)
- Burbuja flotante: Android **Foreground Service + SYSTEM_ALERT_WINDOW** (overlay).
  En Flutter hay paquetes (`flutter_overlay_window`); en Kotlin nativo es directo.
- Mostrar la recomendación encima del juego online.

### 3c. Lectura de pantalla (online) — la parte difícil
- Capturar la pantalla del juego (MediaProjection / Accessibility Service) y **parsear
  las cartas/estado** con visión. Es **frágil y específico de cada app** de Hearts
  online (cada UI es distinta). Empezar por UNA app objetivo.

**Entregable incremental:** copiloto manual usable → overlay → (luego) lectura auto.

---

## Fase 4 — On-device 100% para presencial (con foto/visión)

Objetivo final: en partida presencial, **solo el celular** (sin PC, quizá sin internet).
El modelo debe correr 100% local y reconocer tus cartas.

### 4a. Modelo on-device
- **Exportar** el modelo a un formato móvil: PyTorch → **ONNX** → ONNX Runtime Mobile,
  o → **TFLite**. El MLP es pequeño → inferencia instantánea en cualquier móvil.
- **Reimplementar el núcleo de inferencia** (construcción de obs 228 + máscara +
  tracking de estado + lógica de pase) en Dart/Kotlin. Es el grueso del trabajo;
  conviene tener tests que comparen la obs del móvil vs la de Python (paridad exacta).

### 4b. Reconocimiento de cartas
Dos caminos, de menor a mayor esfuerzo:
- **Entrada rápida manual (recomendado para empezar):** tocar/teclear tus 13 cartas
  toma segundos y es 100% fiable. Es el camino más viable y robusto.
- **Foto + visión:** tomar una foto de tu mano y detectar las cartas con un modelo
  de visión (detección/clasificación de las 52 cartas). Es un proyecto de ML aparte:
  necesita dataset de fotos de cartas etiquetadas, un detector (YOLO/clasificador)
  exportado a TFLite. Útil y "mágico", pero **bastante más trabajo y menos fiable**
  que la entrada manual. Sugerencia: foto solo para tu MANO inicial (13 cartas de
  una vez), y entrada manual/tap para las jugadas de la mesa (que cambian rápido).

**Entregable:** app offline que, dada tu mano (tap o foto) y las jugadas, te dice
qué pasar y qué jugar — sin PC ni internet.

---

## Resumen de prioridades

1. **Validar como copiloto** (Fase 1) — gratis, ya disponible. *Hazlo primero.*
2. **Logger + dataset humano** (Fase 2) — mide eficiencia real, barato.
3. **App copiloto manual vía servidor** (Fase 3a-A) — UX rápida, reusa Python.
4. **Overlay** (Fase 3b) → **lectura de pantalla** (Fase 3c) — incremental.
5. **On-device + visión** (Fase 4) — el objetivo final, el más complejo.

> Consejo: la **entrada manual** (tap de cartas) es el 90% del valor con el 10% del
> esfuerzo. La visión por foto es lo último y opcional. Y aislar un "núcleo de
> inferencia" portable (obs + estado + modelo) desde ya facilitará todas las fases.

---

## Mejoras posibles del MODELO (paralelas, opcionales)
- **Dataset BC más grande / más mundos PIMC** → BC base más fuerte.
- **Más arquetipos / dataset humano en el pool** → mejor generalización a humanos.
- **Meta de marcador avanzada en BotExperto** (kingmaker, asesinato asistido,
  aplastar al colista) — baja frecuencia, PIMC ya compensa; solo si se busca
  fidelidad humana máxima.
- **Entropía / curriculum**: v10c quedó en entropía 0.41 (sana); experimentar con
  schedules si se quiere exprimir más.
