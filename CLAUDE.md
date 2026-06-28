# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Reinforcement learning agent for the card game Hearts (Corazones) on **Ray RLlib v2.55.1 + PPO** (old API stack, TorchModelV2). Pipeline: **Behavioral Cloning from a PIMC oracle** → **PPO fine-tune with diverse self-play** against a pool of historical snapshots + heuristic archetypes, evaluated with a least-squares Elo system. The champion **v10c** plays full games to 100 pts including the card pass. Includes a copilot (`recomendador.py`) for real games.

> **Branch `feature/v5`** (current). Legacy SB3 code removed; entry point is `train_rllib.py`. Champion model in `models/produccion/`. Design doc: `docs/Rediseño_v10_partida_completa.md`. Roadmap (copilot, mobile app, human dataset, on-device): `docs/ROADMAP.md`. Obsolete docs archived under `docs/historico/`.

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
python train_rllib.py --total-steps 20000000 --output-dir models/v_rllib

# With more workers and GPU
python train_rllib.py --total-steps 20000000 --workers 4 --gpus 1 --output-dir models/v_rllib
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
- `dimensiones.py`: SSOT for observation dimensions (`DIM_V11=224` default sin pase, `DIM_V12=228` con pase; `DIM_ENTORNO=224`). Always import from here — never hardcode `224`/`228`.

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

### Top-level scripts

`train_rllib.py` (PPO + self-play; flags `--con-pase`, `--bc-weights`, `--pool-diverso`, `--ancla-experto`), `generar_dataset_bc.py` (PIMC-labeled BC dataset), `entrenar_bc.py` (BC training), `evaluar_final.py`, `monitorear.py` (live training progress), `jugar_modelo.py` (watch/play vs model), `recomendador.py` (**copilot** for real games), `analizar_errores.py`, `comparar_snapshots.py`, `pimc_regret.py`.

> **Planned (Roadmap Fase 2–3):** `src/captura/` — ADB-driven data collector for human games. Will reuse `dominio`/`entorno`, hide ADB+vision behind an `AdaptadorJuego` interface (DIP), and ship optional deps in `requirements-captura.txt`.

### Observation Vector (224 dims = v11 no-pass; 228 = v12 with pass)

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
| `[187]` | `moon_prob_agente`: P(Moon del agente) continuo [0, 1] |
| `[188]` | `moon_prob_rival`: max P(Moon) entre los 3 rivales [0, 1] |
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

All positions are **relative to the agent** (`(player_idx - agent_idx) % 4`).

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
