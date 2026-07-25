# Proyecto Corazones — Ecosistema de IA para el juego de cartas Hearts

> **Brief técnico para CV / portafolio.** Documento único y autocontenido: describe los tres
> componentes del ecosistema (agente de RL, bridge para partidas reales, app Angular de registro)
> con suficiente detalle para que cualquier persona —o IA— entienda qué se construyó, con qué
> tecnología y hacia dónde va. Todo está diseñado para converger en una sola plataforma.

---

## 0. Resumen ejecutivo (elevator pitch)

Ecosistema completo, construido de punta a punta por una sola persona, alrededor del juego de
cartas **Corazones (Hearts)**. Tiene tres piezas que hoy funcionan por separado y están pensadas
para unirse:

1. **Un agente de Reinforcement Learning** que aprende a jugar Hearts a nivel experto
   (Ray RLlib + PPO, Behavioral Cloning desde un oráculo PIMC, self-play diverso, evaluación por Elo).
2. **Un "bridge" de visión por computadora + copiloto** que conecta ese agente a **partidas reales**
   en una app de móvil: lee la pantalla, reconoce las cartas, recomienda la jugada y hasta puede
   ejecutar el pase automáticamente.
3. **Una aplicación web Angular** (arquitectura hexagonal, Firebase, tiempo real) para llevar el
   **registro de partidas humanas**: puntajes, pases, premios/pagos, salas e invitación de espectadores.

El hilo conductor: los tres componentes comparten el mismo dominio de juego y los mismos datos.
El agente genera y consume datos; el bridge captura partidas reales que alimentan al agente; la app
web registra partidas humanas que son, a la vez, dataset de entrenamiento y producto para usuarios.

---

## 1. Componente A — Agente de Reinforcement Learning (el modelo de ML)

**Stack:** Python 3.12 · Ray RLlib v2.55.1 · PPO (old API stack, `TorchModelV2`) · PyTorch · Gymnasium.

### 1.1 El problema y por qué es difícil

Hearts es un juego de **información imperfecta**, 4 jugadores, con mecánicas que rompen los enfoques
ingenuos de RL:

- No ves las manos de los rivales (información oculta).
- El objetivo es **minimizar** puntos… salvo cuando conviene "**dispararle a la luna**" (shooting the
  moon): si capturas *todos* los puntos, tu penalización se invierte y penalizas a los rivales. Esto
  crea una política no monótona muy difícil de aprender.
- Hay una **fase de pase** de 3 cartas antes de cada mano que cambia según una rotación
  (izquierda → frente → derecha → no pasa).
- La partida real es **a 100 puntos**, es decir, muchas manos encadenadas donde el marcador persiste:
  las decisiones dependen de quién va ganando/perdiendo globalmente.

### 1.2 Pipeline de entrenamiento (de cero a campeón)

```
Oráculo PIMC  ──►  Behavioral Cloning  ──►  PPO fine-tune con self-play diverso  ──►  Evaluación Elo
(maestro)          (imitar al maestro)       (superar al maestro jugando solo)        (medir de verdad)
```

1. **Oráculo PIMC (Perfect Information Monte Carlo).** Para cada carta legal, simula cientos de
   "mundos" posibles (repartos aleatorios de las cartas ocultas) con rollouts de arquetipos mixtos, y
   elige la carta que minimiza el puntaje esperado. Es un "maestro" caro pero fuerte que genera
   pares `(observación, acción)` etiquetados.

2. **Behavioral Cloning (BC).** Se pre-entrena una red para **imitar** al oráculo PIMC. Esto da un
   punto de partida muchísimo mejor que empezar de cero (evita millones de pasos de exploración inútil).

3. **PPO con self-play diverso.** El agente se afina jugando contra un **pool de oponentes**
   gestionado por fases: al inicio bots heurísticos, luego un bot experto, luego snapshots históricos
   de sí mismo, hasta terminar jugando puro self-play contra versiones previas. Esto evita el
   sobreajuste a un solo estilo de rival y produce una política robusta.

4. **Evaluación por Elo de mínimos cuadrados.** La métrica primaria **no** es el win-rate contra bots
   (señal débil), sino un **Elo sin sesgo de orden** calculado con mínimos cuadrados sobre torneos
   entre snapshots. El win-rate contra heurísticos es solo una señal secundaria.

**Campeón actual: `v10c`** — juega la partida completa a 100 puntos incluyendo el pase de cartas.

### 1.3 Diseño técnico destacable

- **Un episodio = una partida completa** a 100 puntos (varias manos, el marcador persiste). Esto fue
  un rediseño consciente: la versión previa modelaba una-sola-mano como MDP y se estancó, porque
  perdía toda la estrategia de "gestión del marcador" a lo largo de la partida.
- **Recompensa (SSOT única).** `R_terminal` por posición final (1º = +1, 2º = +0.3, 3º = −0.3,
  4º = −1) **más** *Potential-Based Reward Shaping (PBRS)* sobre el marcador. El PBRS acelera el
  aprendizaje sin cambiar la política óptima (garantía teórica), y su `gamma` debe coincidir con el de PPO.
- **Vector de observación versionado y con Single-Source-of-Truth.** 224 dims (sin pase), 228 (con
  pase), 332 (con memoria del pase). Codifica: mano propia, mesa, cementerio, voids conocidos por
  rival y palo, marcadores normalizados, corazones rotos, tracker de la Dama de Picas, probabilidad de
  luna propia y rival, alertas de luna, cartas altas restantes por palo, etc. **Todo relativo al
  agente**, y la memoria del pase se computa por-perspectiva (cada jugador solo conoce lo suyo), así
  que **transfiere a partidas reales**.
- **Action masking.** La red (`HeartsActionMaskModel`, MLP 512→512→256) enmascara jugadas ilegales
  clampeando sus logits a −1e9, garantizando que solo se elijan cartas legales.

### 1.4 Modelos de "luna" aprendidos y la ofensiva de luna (línea de trabajo actual)

La heurística vieja estimaba la probabilidad de que alguien dispare a la luna con coeficientes fijos.
Se reemplazó por **dos MLPs entrenados** (`EstimadorMoonProb`):

- `features_propio`: predice P(luna del agente) — **AUC 0.945** en validación.
- `features_rival`: predice P(luna de un rival) usando solo señales públicas (nunca su mano) — AUC 0.707.

Sobre eso se construyó **`ModoLunar`**: una *ofensiva de luna por composición* (sin RL nuevo) que
combina el gate aprendido + un pase constructivo + compromiso dinámico (entra a mitad de mano si la
probabilidad sube — el dato real muestra que 13 de 14 lunas humanas son oportunistas a media mano) +
persecución + abortos duros/suaves. Validado en **3000 partidas pareadas** contra un clon del campeón:
**~+1 punto porcentual y nunca negativo** (EV-neutro o positivo). El instrumento de A/B re-siembra
todos los RNG por partida para un pareo limpio.

### 1.5 Cómo se evalúa la seriedad de la ingeniería

- Tests con `pytest` (dominio, torneo/Elo, replay de observaciones, etc.).
- Arquitectura en capas estricta: `dominio/` (lógica pura del juego, sin dependencias de RL) →
  `entorno/` (envs Gymnasium) → `rllib/` (pipeline) → `torneo/` (Elo/evaluación) → `mcts/` (oráculo).
- Principios SSOT explícitos: dimensiones y recompensas viven en un solo lugar, nunca hardcodeadas.
- Verificación empírica y "pre-declaración de veredictos" antes de cada experimento (disciplina anti
  p-hacking): se fija la hipótesis y el criterio de éxito *antes* de correr las 3000 partidas.

---

## 2. Componente B — El "bridge": del modelo a las partidas reales

**Objetivo:** que el agente entrenado deje de vivir en un simulador y **juegue/asista partidas reales**
en una app de Hearts para Android, sin acceso al código del juego, solo mirando la pantalla.

### 2.1 Copiloto (`recomendador.py`)

Un asistente que, dada la situación de una partida real, **recomienda la mejor jugada** usando el
modelo campeón (y, opcionalmente, `ModoLunar`). Es el puente de inferencia: mismo estimador de
probabilidad de luna en entrenamiento y en producción, para que no haya *train/serve skew*.

### 2.2 Visión por computadora (leer la pantalla del juego)

Pipeline propio de visión, calibrado para la app real (1080×1728), **sin OCR de sistema**:

- **Lectura del banner de texto** por *template matching* → fase del juego, dirección del pase, de
  quién es el turno, ganador de la baza. Generaliza entre dispositivos.
- **Reconocimiento de cartas** con dos motores: uno híbrido para las cartas de la **mesa** (detección
  + color rojo/negro + rango por glifo + palo por forma del pip: lóbulos ♥/♦, solidez ♣/♠); y un
  `ReconocedorPlantilla` para la **mano** del jugador basado en *matchTemplate multiescala* contra los
  naipes completos extraídos del propio sprite de la APK (arte idéntico al render). Resultado:
  **lectura de la mano inicial 13/13 perfecta**. Ambos motores devuelven "no sé" ante baja confianza —
  **nunca emiten una carta equivocada**.
- **Máquina de estados** (`MaquinaCaptura`) que convierte la secuencia de frames en un stream de
  eventos de juego, con **confirmación temporal** (una carta debe verse estable varios frames) para
  filtrar *misreads* de animación, y bufferiza las 4 cartas de cada baza para emitirlas en orden de turno.

### 2.3 Arquitectura desacoplada (misma "cabeza", muchas fuentes)

Se aplicó **inversión de dependencias** con un puerto `AdaptadorJuego` que emite un stream de eventos
(`InicioMano`, `PaseAgente`, `JugadaObservada`, `RemateResto`, `FinMano`, `FinPartida`). El colector
solo conoce eventos, así que la fuente es intercambiable:

- `AdaptadorManual` — narras la partida por consola (usable hoy, sin necesidad de dispositivo).
- `AdaptadorSimulado` — el motor juega solo (valida todo el pipeline sin hardware).
- `AdaptadorVisual` — conduce la máquina de estados desde una carpeta de frames, un video o polling por ADB.

El **mismo "cerebro"** valida offline y captura en vivo. Cada partida capturada se re-juega
(*replay* puro) a través del motor + el constructor de observaciones para emitir pares `(obs, acción)`
idénticos a los del entrenamiento — cerrando el ciclo: **las partidas reales se vuelven dataset**.

### 2.4 Auto-pase / auto-juego (actuación)

`ControladorPase` cierra el lazo actuando sobre el dispositivo por ADB: lee la mano con sus puntos de
toque en píxeles, pide 3 cartas al copiloto, **relee la mano antes de cada toque** (porque las cartas
seleccionadas se re-acomodan visualmente), confirma y deduce las cartas recibidas por diferencia.
Solo toca cuando localiza positivamente la carta objetivo. *(El auto-juego de las bazas queda como
siguiente hito.)*

> ⚠️ Nota de responsabilidad: automatizar taps sobre una app de terceros puede violar sus términos de
> servicio; por eso el sistema tiene modos de solo-observación y "seco" (dry-run) y está pensado como
> investigación/copiloto, no como bot desatendido.

---

## 3. Componente C — App Angular de registro de partidas (`E:\Angular\CorazonesApp`)

**Qué es:** una aplicación web para **llevar el registro de partidas de Hearts entre humanos** —
puntajes ronda a ronda, dirección de pases, cálculo de premios y pagos, historial, y **salas con
invitación de espectadores en tiempo real**. Es el producto de cara al usuario y, a la vez, la fuente
natural de datos de partidas humanas.

### 3.1 Stack

- **Angular 21** (standalone components, sin NgModules), **Signals** + `computed()` para estado reactivo,
  `ChangeDetectionStrategy.OnPush` en todos los componentes, rutas con *lazy loading*.
- **PrimeNG 21** + `@primeuix/themes` para la UI; `primeicons`.
- **Firebase** (`@angular/fire` 21): Auth, Firestore (tiempo real) y Hosting.
- **TypeScript strict** (sin `any`), `readonly` en modelos de dominio, Prettier.
- **Vitest** como test runner.

### 3.2 Arquitectura (hexagonal / puertos y adaptadores)

Separación en capas limpia — el punto más fuerte del proyecto según su propia auditoría interna:

```
core/          → dominio puro: models, ports, services, utils (sin dependencias de infraestructura)
features/      → páginas por caso de uso (scoreboard, history, summary, room, viewer, profile, auth)
infrastructure/→ implementaciones concretas de los puertos (Firestore, LocalStorage, Hybrid)
shared/        → componentes reutilizables (p. ej. teclado numérico)
```

**Patrón puerto/adaptador (DIP):** el token `GAMES_REPOSITORY` abstrae la persistencia. Hay tres
implementaciones **intercambiables** que cumplen el mismo contrato `GamesRepository`:

- `FirestoreGamesRepository` — nube, para usuarios autenticados.
- `LocalStorageGamesRepository` — offline, sin cuenta.
- `HybridGamesRepository` — *smart routing*: delega en Firestore si hay sesión, en LocalStorage si no.

Los servicios (p. ej. `GameSessionService`) dependen del **puerto**, no de la implementación: agregar
una nueva fuente de datos no toca la lógica existente (Open/Closed real).

### 3.3 Funcionalidades

- **Scoreboard en vivo:** captura de puntajes por ronda con un teclado numérico propio; totales,
  puntaje de victoria configurable, migración automática de partidas antiguas al esquema nuevo.
- **Reglas de dominio modeladas explícitamente:** rotación de la dirección de pase (`standard`/`reverse`),
  *shooting the moon* (redistribución/inversión), "castigo" y "pozo", valor configurable de la Dama de
  Picas (13 o 5 puntos), y **sustitución de jugadores a media partida** (modelo de `slots` con
  entradas por rango de rondas).
- **Cálculo de premios y liquidación de pagos:** algoritmo direccional deudor→acreedor (el mayor
  deudor paga al mayor acreedor primero) que minimiza el número de transferencias. Manejo de empates
  con orden de desempate persistido.
- **Salas y espectadores:** crear una sala, invitar espectadores por código; los espectadores ven la
  tabla de posiciones (`standings`) en **tiempo real** vía rutas `/r/:gameId` y `/sala/:roomCode`.
- **Historial:** lista y detalle de partidas terminadas; **exportar el resumen como imagen**
  (`html2canvas`, lazy-loaded).
- **Auth y perfil:** Firebase Auth, guard de rutas protegidas, reglas de Firestore que restringen el
  acceso de cada usuario a sus datos.

### 3.4 Calidad

Auditoría interna (SOLID / seguridad / performance) con veredicto "bien estructurado, sin deuda
técnica crítica": SRP por servicio, DIP modelo, sin `any`, sin memory leaks (uso de `toSignal`/`effect`
con cleanup automático, sin `.subscribe()` manual en componentes), sin `innerHTML` inseguro, queries de
Firestore indexadas sin *full-collection scans*. Build con *code splitting* por feature.

---

## 4. La visión: converger en una sola plataforma

Hoy los tres componentes están **desacoplados a propósito** pero comparten dominio y datos, y están
pensados para unirse:

- La **app Angular** deja de ser solo un marcador y se convierte en el front-end del ecosistema: un
  usuario registra su partida humana y, con un toque, **pide la recomendación del agente de RL**
  (Componente A) para su mano — el copiloto entra en la web.
- El **bridge de visión** (Componente B) y la app comparten el mismo modelo de dominio y formato de
  eventos, así que una partida —capturada por visión *o* registrada a mano en la web— alimenta el
  mismo dataset que **reentrena al agente**. El ciclo se cierra: humanos generan datos → el agente
  mejora → el agente asiste a los humanos.
- Objetivo final: **on-device / móvil** — que el copiloto y (donde sea lícito) la asistencia de juego
  vivan junto al registro de partidas, en una sola experiencia.

En una frase para el CV: *un ecosistema vertical completo —investigación de RL, visión por computadora
para el mundo real, y una app web de producción— unificado por un mismo dominio de juego y diseñado
para converger.*

---

## 5. Resumen de tecnologías y competencias demostradas

| Área | Tecnologías / conceptos |
|------|--------------------------|
| **Reinforcement Learning** | Ray RLlib, PPO, self-play diverso, PBRS reward shaping, action masking, Elo por mínimos cuadrados |
| **Imitation / búsqueda** | Behavioral Cloning, PIMC (Perfect Information Monte Carlo), regret |
| **ML aplicado** | MLPs en PyTorch, ingeniería de features, evaluación (AUC), A/B pareado, disciplina experimental |
| **Visión por computadora** | template matching multiescala, detección/clasificación de cartas sin OCR, máquina de estados de captura, integración ADB |
| **Arquitectura de software** | capas/hexagonal, puertos y adaptadores, DIP, SSOT, Strategy pattern (en Python y en Angular) |
| **Frontend** | Angular 21, Signals, standalone components, OnPush, PrimeNG, TypeScript strict |
| **Backend / cloud** | Firebase Auth + Firestore (tiempo real) + Hosting, reglas de seguridad, sincronización offline/online |
| **Ingeniería** | pytest / Vitest, diseño para determinismo y reproducibilidad, code splitting, documentación de diseño |
| **Lenguajes** | Python, TypeScript |

---

*Nota: este documento es un brief técnico exhaustivo pensado para dárselo a otra IA (o a un
reclutador) que lo reescriba en el tono y extensión adecuados para el CV. Todos los detalles de
métricas y arquitectura son reales y verificables en los dos repositorios.*
