# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Reinforcement learning agent for the card game Hearts (Corazones) on **Ray RLlib v2.55.1 + PPO** (old API stack, TorchModelV2). Pipeline: **Behavioral Cloning from a PIMC oracle** → **PPO fine-tune with diverse self-play** against a pool of historical snapshots + heuristic archetypes, evaluated with a least-squares Elo system. The champion **v10c** plays full games to 100 pts including the card pass. Includes a copilot (`recomendador.py`) for real games.

> **Branch `feature/v5`** (current). Legacy SB3 code removed; entry point is `scripts/train_rllib.py`. Champion model in `models/produccion/`. Design doc: `docs/Rediseño_v10_partida_completa.md`. Roadmap (copilot, mobile app, human dataset, on-device): `docs/ROADMAP.md`. Obsolete docs archived under `docs/historico/`.

## Commands

All commands assume the virtual environment is activated:

```powershell
.venv\Scripts\Activate.ps1
```

### Tests

```bash
# Run all tests
python -m pytest tests/ -q

# Run a single test file
python -m pytest tests/dominio/test_modulo1.py -q

# Run a single test
python -m pytest tests/torneo/test_elo.py::TestEloConvergente::test_elo_inicial -q
```

### Training (RLlib — nuevo pipeline)

**Requires Python 3.12** — Ray 2.55.1 has no wheels for Python 3.13.

```bash
# Train from scratch (RLlib PPO)
python scripts/train_rllib.py --total-steps 20000000 --output-dir models/v_rllib

# With more workers and GPU
python scripts/train_rllib.py --total-steps 20000000 --workers 4 --gpus 1 --output-dir models/v_rllib
```

### Evaluation

```bash
# Elo tournament between snapshots
python -m src.torneo.elo --directorio models/v8/elite --partidas 50 --elo-puro --incluir-bots
```

## Architecture

### Layer Structure

**`src/dominio/`** — Pure game logic, no RL dependencies.

- `carta.py`: Immutable `Carta` value object with `id`, `palo`, `valor`, `puntos`, `es_corazon`, `es_dama_de_picas`.
- `baraja.py`: French deck, shuffling, dealing.
- `jugador.py`: Player state (hand, `bazas_ganadas`, accumulated score).
- `motor.py`: `MotorCorazones` — full game loop (tricks, scoring, `obtener_jugadas_legales()` with 4-filter cascade, shooting the moon, broken hearts, Q♠).

**`src/entorno/`** — Gymnasium RL environments.

- `corazones_rllib.py`: `CorazonesEnvRLlib(gym.Env)` — **primary RLlib training env**. **One episode = one FULL GAME to 100 pts** (multiple hands, scoreboard persists). Obs space `Dict({"obs": Box(obs_dim,), "action_mask": Box(52,)})` where `obs_dim` defaults to `DIM_ENTORNO=224`; with `con_pase=True` (the pass phase) it requires `obs_dim>=228` (v10c uses 228). Reward = R_terminal by final placement (1st=+1, 2nd=+0.3, 3rd=−0.3, 4th=−1) + PBRS potential shaping on the scoreboard (`reward_config`/`gamma` in env_config; gamma must match PPO). Opponents managed inside via `opponent_factory`. Passes `gymnasium.check_env`. See `docs/Rediseño_v10_partida_completa.md`.
- `observacion.py`: `ObservacionBuilder` — SSOT for the observation vector. Supports 224 (v11, no pass) and 228 (v12, with pass) via the `dim` constructor arg. Scoreboard features (`[172:176]`, near-100, leader, terminal-hand) are LIVE because score persists across hands.
- `recompensas_partida.py`: **SSOT for rewards** — `RewardConfigPartida` + `CalculadoraRecompensasPartida` (R_terminal + PBRS Φ). This is what `CorazonesEnvRLlib` uses.
- `dimensiones.py`: SSOT for observation dimensions (`DIM_V11=224` default sin pase, `DIM_V12=228` con pase, `DIM_V13=332` con pase + memoria del pase; `DIM_ENTORNO=224`). Always import from here — never hardcode `224`/`228`/`332`.
- `moon_model.py`: 2 modelos aprendidos (MLP chico en PyTorch) que reemplazan la heurística de coeficientes fijos para `moon_prob_agente`/`moon_prob_rival` (obs `[187:189]`) — `features_propio` (reutiliza `ObservacionBuilder(dim=332)` desde la perspectiva real del agente) + `features_rival` (solo señales públicas de un rival específico, nunca su mano) + `EstimadorMoonProb` (con gate duro y fallback seguro a 0.0 sin pesos entrenados). Usado hoy en `scripts/recomendador.py` (producción); `src/entorno/corazones_rllib.py` sigue usando la heurística vieja durante el entrenamiento de v10c (por diseño, ver spec en `docs/superpowers/specs/2026-07-06-moon-prob-modelo-aprendido-design.md`). Entrenar con `scripts/entrenar_moon_prob.py` (genera dataset re-jugando manos reales de `data/partidas_bridge.jsonl`).

**`src/agentes/`** — Agent strategies (Strategy pattern: `(motor, idx, legales) → Carta`).

- `heuristicos.py`: Three stateless bots — `bot_conservador`, `bot_agresivo`, `bot_evasivo`.
- `bot_experto.py`: `BotExperto` — stronger heuristic using void-tracking and suit-lead logic.
- `bot_castigador.py`: `BotCastigador` — stateful per-hand bot, resets automatically each mano.
- `bot_lunatico.py`: `BotLunatico` — moon-shooter (hunts the pozo).
- `bot_atacante_lider.py`: `BotAtacanteLider` — loads points onto the scoreboard leader.
- `pase.py`: `pase_heuristico` — per-archetype card-pass strategy.

**`src/rllib/`** — RLlib pipeline components (new).

- `model.py`: `HeartsActionMaskModel(TorchModelV2)` — MLP `input → 512 → 512 → 256` with action masking via logit clamping. Registered as `"hearts_model"` in `ModelCatalog`.
- `config.py`: `build_ppo_config()` — builds `PPOConfig` (old API stack). Key Ray 2.55.1 params: `minibatch_size` (was `sgd_minibatch_size`), `num_epochs` (was `num_sgd_iter`), `env_runners()` (was `rollouts()`), `preprocessor_pref=None` for Dict obs.
- `opponent_pool.py`: `OpponentPool` — phase-based factory (0–5%: bots, 5–15%: expert, 15–40%: expert+snapshots, 40–70%: snapshots, 70–100%: pure snapshots). `SnapshotPolicy` wraps a Ray policy into the `(motor, idx, legales) → Carta` signature (with neural pass).
- `eval_bots.py`: full-game evaluation against bots, via the env.
- `callbacks.py`: `HeartsCallbacks(DefaultCallbacks)` — logs episode metrics to `eval_log.jsonl`.
- `utils.py`: snapshot save/load/prune utilities.

**`src/torneo/`** — Evaluation infrastructure.

- `elo.py`: Least-squares Elo (no order bias). Invokable as `python -m src.torneo.elo`.
- `evaluacion.py`: Win-rate evaluation against heuristic bots (includes its own VecNormalize handling for the SB3 golden baselines).

**`src/mcts/`** — PIMC (Perfect Information Monte Carlo) oracle and dataset tools.

- `pimc.py`: `pimc_mejor_jugada()` — for each legal card, simulates `num_mundos` random completions, returns the card minimizing expected score (mixed archetype rollouts).
- `pimc_recursivo.py`: recursive PIMC scoring used by the dataset generator.
- `dataset.py`: Generates `(obs, action)` pairs for BC pretraining (with/without pass).
- `analisis.py`: error-analysis helpers (used by `analizar_errores.py`).

### Scripts (`scripts/`)

All CLI entry points live in `scripts/` and are run from the repo root as `python scripts/<name>.py` (each has a `sys.path` bootstrap so it resolves `src` regardless of cwd).

`train_rllib.py` (PPO + self-play; flags `--con-pase`, `--bc-weights`, `--pool-diverso`, `--ancla-experto`), `generar_dataset_bc.py` (PIMC-labeled BC dataset), `entrenar_bc.py` (BC training), `evaluar_final.py`, `monitorear.py` (live training progress), `jugar_modelo.py` (watch/play vs model), `recomendador.py` (**copilot** for real games), `analizar_errores.py`, `comparar_snapshots.py`, `pimc_regret.py`.

**`src/captura/`** — Human-game dataset collector (Roadmap Fase 2–3). Reuses `dominio`/`entorno`; only `adb.py` needs optional deps (`requirements-captura.txt`).

- `puerto.py`: `AdaptadorJuego` (ABC) — emits an **event stream** (`InicioMano`, `PaseAgente`, `JugadaObservada`, `RemateResto`, `FinMano`, `FinPartida`). The DIP boundary: the collector knows only events, so the source (ADB / manual / sim) is swappable. `RemateResto` models the app's "se llevará el resto" (a player claims all remaining tricks): it records the conceding seat + the revealed remaining hands instead of fabricating plays.
- `manual.py`: `AdaptadorManual` — console source, **usable today without ADB** (you narrate the game; it computes the scoreboard, incl. concession when you type `resto`).
- `simulado.py`: `AdaptadorSimulado` — motor-played games (random-legal, with pass) to validate the whole pipeline without a device. Backs `capturar.py --fuente demo`.
- `adb.py`: `ClienteADB` (real `adb` subprocess: screencap/tap), `ParserPantalla` (ABC) + `ParserPlantillas` (legacy template-matching skeleton), `AdaptadorADB` (old polling skeleton). Lazy-imports `cv2`.
- **Visión por captura de pantalla (app Hearts es, 1080×1728)** — pipeline nuevo, calibrado en `calibracion/hearts_app/` (ver su README):
  - `vision_hearts.py`: `Regiones` (cajas en fracciones), `BannerClasificador` (lee el banner de texto por plantillas → fase/dirección de pase/turno/ganador de baza, sin OCR de sistema), `leer_mesa`/`leer_estado` → `EstadoVisual`, y **`leer_mano`** (mano del agente, 13 cartas — **100% en la captura real**).
  - `vision_cartas.py`: dos reconocedores de carta. (1) `Reconocedor` híbrido para la **mesa** — `detectar_cartas` + color (rojo/negro) + rango (glifo por plantilla) + palo (forma del pip: lóbulos ♥/♦, *solidez* ♣/♠). (2) **`ReconocedorPlantilla`** para la **mano** — `matchTemplate` (correlación normalizada **multiescala**, `_ESCALAS`) contra los **naipes completos del sprite de la APK** (arte idéntico al renderizado); la ventana de búsqueda absorbe el desajuste de pocos píxeles que rompe la firma de esquina. `leer_mano` localiza las 13 cartas solapadas (`localizar_cartas`: franja fina + interpolación de **posiciones uniformes**, ya que el paso del abanico es constante), saca el palo por bloque (la app agrupa la mano por palo) y el rango por carta. Ambos devuelven `None`/carta omitida si la confianza es baja (no emiten carta errónea). Las plantillas se generan del sprite con `cartas_desde_sprite.py` → `calibracion/hearts_app/cartas_completas/` (versionadas).
  - `maquina.py`: `MaquinaCaptura` — pura; convierte la secuencia de `EstadoVisual` en eventos del puerto. Bufferiza las 4 cartas de cada baza y las emite en **orden de turno** (líder = quien juega el 2♣ en la baza 1; ganador del banner "X recoge la baza" en las siguientes). **Confirmación temporal** (una carta debe verse estable ≥`min_confirmaciones` frames) para filtrar misreads de animación; lee la mesa solo en estados asentados (`turno`/`baza`).
  - `adaptador_visual.py`: `AdaptadorVisual(AdaptadorJuego)` — conduce la máquina desde una **fuente de fotogramas** (carpeta/video o poll ADB). El mismo cerebro valida offline y captura en vivo.
- `recolector.py`: `RecolectorPartidas` — aggregates events → `RegistroPartida`.
- `escritor.py` / `modelos.py`: append-only JSONL I/O + DTOs (cards stored as `carta.id`).
- `replay.py`: **pure** offline encoder — replays a `RegistroPartida` through `MotorCorazones` + `ObservacionBuilder` to emit `(obs, accion)`. Hands are reconstructed from the recorded plays plus `manos_restantes` (each seat totals 13 cards), so opponents' hands are never needed and conceded rounds still replay. ⚠ Currently emits the **minimal** obs (`construir_desde_motor`): it does NOT yet rebuild strategic features nor v13 pass-memory, so human-data BC isn't yet aligned with a v13-trained model. Aligning the encoder to full v13 obs (incl. `pase_dado`/`direccion` → di/recibí planes) is a TODO before fine-tuning on human data.

Scripts: `scripts/capturar.py` (`--fuente manual|demo|adb`), `scripts/inspeccionar_captura.py` (decode a `.jsonl` to readable form + integrity check), `scripts/calibrar_captura.py` (grab a screenshot via ADB), `scripts/jsonl_a_dataset.py` (JSONL → `(obs, accion)` `.npz`). ⚠ Auto-clicking a game app may violate its ToS.

Visión por captura (app Hearts, ver `calibracion/hearts_app/README.md`): `scripts/capturar_visual.py` (`--fuente carpeta|adb`, captura por visión → JSONL; `--solo-validas` filtra por replay → dataset limpio; `--validar` re-juega), `scripts/diagnostico_captura.py` (un frame → imprime banner + mesa + mano legibles; `--overlay` dibuja las regiones), `scripts/monitor_vivo.py` (poll ADB → consola en vivo, **solo observa**; `--frame` para probar sin ADB), `scripts/cartas_desde_sprite.py` (genera `cartas_completas/` + esquinas desde el sprite de la APK), `scripts/calibrar_regiones.py` (dibuja las regiones sobre un frame para verificarlas), `scripts/agrupar_banners.py` / `scripts/agrupar_cartas.py` (descubren plantillas nuevas agrupando recortes por similitud → etiquetar a mano), `scripts/mapear_frames.py` (mapea cada frame → estado de banner), `scripts/auto_pase.py` (**auto-pase**: tap por ADB con el modelo — `--frame X --seco` para probar sin tocar). Estado: **mano inicial 13/13 perfecto** (matchTemplate multiescala del sprite); banner OK (generaliza entre dispositivos); mesa rango+color+palo OK en el video. **Auto-pase implementado** (ver `src/captura/auto_pase.py`), pendiente de calibrar el botón círculo-check (`regiones.confirmar` o plantilla `confirmar.png`). Falta: aplicar el mismo `ReconocedorPlantilla` a la **mesa** (validar con frames en-juego), y **auto-juego de las bazas**. ⚠ ADB para visión necesita `adb` en PATH (binarios en `herramientas/`, gitignored); auto-tap puede violar la ToS de la app.

- `auto_pase.py`: `ControladorPase` — actuador del **pase**. Lee la mano con `leer_mano_posiciones` (id + punto de toque en píxeles), pide 3 cartas a un callback `recomendar` (envuelve `Recomendador`), las toca **releyendo la mano antes de cada toque** (las seleccionadas suben sobre el banner y la mano se re-fluye por bloques ♣♦♠♥ → no se cachean posiciones), confirma (`_confirmar`) y deduce las recibidas por diferencia (`mano_nueva − (mano_vieja − pasadas)`, sin región especial). Solo toca cuando localiza positivamente la carta objetivo (nunca toca una carta equivocada).

### Observation Vector (224 = v11 no-pass; 228 = v12 with pass; 332 = v13 + pass memory)

| Range | Content |
| ------- | --------- |
| `[0:52]` | Agent's hand (one-hot) |
| `[52:104]` | Current trick / mesa (one-hot) |
| `[104:156]` | Played cards / cementerio (one-hot) |
| `[156:172]` | Known voids: 4 players × 4 suits |
| `[172:176]` | Historical scores normalized /100 |
| `[176:180]` | Current hand points normalized /26 |
| `[180]` | Hearts broken (0/1) |
| `[181]` | Position in trick (0.0, 0.33, 0.66, 1.0) |
| `[182:187]` | Q♠ tracker (one-hot, 5 states) |
| `[187]` | `moon_prob_agente`: P(Moon del agente) continuo [0, 1] — heurística fija en entrenamiento (`corazones_rllib.py`), modelo aprendido en producción (`moon_model.py`, ver arriba) |
| `[188]` | `moon_prob_rival`: max P(Moon) entre los 3 rivales [0, 1] — misma nota que `[187]` |
| `[189]` | `puedo_alimentar`: puede dar puntos a un rival (0/1) |
| `[190:194]` | `all_void_X`: all 3 opponents are void in suit X |
| `[194]` | Baza number / 13.0 |
| `[195]` | Players near 100 / 3.0 |
| `[196]` | Q♠ already captured |
| `[197]` | Agent is score leader |
| `[198]` | Terminal hand possible (score ≥74) |
| `[199:203]` | Cards remaining by suit / 13.0 |
| `[203:207]` | High cards (J/Q/K/A) remaining by suit / 4.0 |
| `[207:211]` | Probability Q♠ by relative player |
| `[211:215]` | Hearts captured this hand / 13.0 |
| `[215:219]` | Moon alert by player (≥6 hearts) |
| `[219]` | Led suit (`palo_salida`): 0.0 if None, else suit/3.0 |
| `[220:224]` | Who played in current trick (4 bits, relative positions) |
| `[224:228]` | **v12 pass phase** (`con_pase`): `fase_pase`, `direccion`, `n_seleccionadas`, reserved |
| `[228:280]` | **v13 pass memory**: cards I gave to the receiver, still unplayed (one-hot) |
| `[280:332]` | **v13 pass memory**: cards I received from the giver, still in hand (one-hot) |

All positions are **relative to the agent** (`(player_idx - agent_idx) % 4`). v13 pass-memory is computed **per perspective** (each player knows only its own give/receive) and is deterministic, legal info — so it transfers to real games. Receiver/giver positions derive from `direccion` (`[225]`). Train it with `--con-pase --obs-dim 332`.

### Model Storage Layout

```
models/{version}/
├── config.json          # Version metadata
├── snapshots/           # Training checkpoints (snapshot_NNNNNNNNN.zip)
├── vecnorm/             # VecNormalize stats per snapshot (*_vecnorm.pkl)
├── elite/               # Best N snapshots after Elo tournament
└── torneos/             # Elo results (elo_paso_*.txt, eval_log.jsonl)
```

Golden (frozen) baselines: `models/v5_golden/` (190-dim, ~1500 Elo) and `models/v7_golden/` (194-dim, 1723 Elo) — these are SB3 and use the `vecnorm/` stats. The current champion lives in `models/produccion/` (RLlib, no VecNormalize).

### Key Design Constraints

- **SSOT**: Reward values → `RewardConfigPartida` (recompensas_partida.py). Observation dims → `dimensiones.py`. Never hardcode `224` or `228`.
- **Action masking**: `CorazonesEnvRLlib` returns `{"obs": ..., "action_mask": ...}`. `HeartsActionMaskModel` clamps illegal actions to `-1e9`. `preprocessor_pref=None` is required in `env_runners()` to prevent RLlib from flattening the Dict obs.
- **Ray 2.55.1 API changes** (vs older docs): `rollouts()` → `env_runners()`, `sgd_minibatch_size` → `minibatch_size`, `num_sgd_iter` → `num_epochs`, `.build()` → `.build_algo()`.
- **Self-play pool**: phase-based opponent mixing in `OpponentPool`. Pool capped at 50 most-recent snapshots.
- **`motor.repartir()` clears `bazas_ganadas`**: critical for determinism across episodes (bug fix applied to `src/dominio/motor.py`).
- **Elo is the primary metric**; win rate against bots is a weaker signal.
- **Diagnostic alerts** written to `eval_log.jsonl` with `"tipo": "alerta"` when metrics cross thresholds.
