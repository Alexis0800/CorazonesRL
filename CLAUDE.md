# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Reinforcement learning agent for the card game Hearts (Corazones), migrating from MaskablePPO (SB3) to **Ray RLlib v2.55.1 + PPO** (old API stack, TorchModelV2). The agent trains via Fictitious Self-Play against a pool of its own historical snapshots and heuristic bots, evaluated with a least-squares Elo system.

> **Branch `feature/refactorizacion`**: active RLlib migration. Legacy SB3 code has been removed. The new entry point is `train_rllib.py`.

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

- `corazones_rllib.py`: `CorazonesEnvRLlib(gym.Env)` — **primary RLlib training env**. Observation space `Dict({"obs": Box(224,), "action_mask": Box(52,)})`. One episode = one mano (13 agent steps). Opponents managed inside the env via `opponent_factory`. Passes `gymnasium.check_env`.
- `observacion.py`: `ObservacionBuilder` — SSOT for the **224-dim** observation vector (v11). Supports 190/194/220/224 via constructor parameter. Call `construir()` for full observation; `construir_desde_motor()` for minimal (used by opponent snapshots in self-play).
- `recompensas.py`: `RewardConfig` (frozen dataclass, **v12 SSOT**) and `CalculadoraRecompensas` — ALL reward logic centralized here (SRP).
- `recompensas_minimal.py`: `RewardConfigMinimal` — alternative minimal reward system (3 signals only) for A/B experimentation.
- `dimensiones.py`: SSOT for observation dimensions (`DIM_V5=190`, `DIM_V6=194`, `DIM_V10=220`, `DIM_V11=224`, `DIM_ENTORNO=224`). Always import from here — never hardcode.

**`src/agentes/`** — Agent strategies (Strategy pattern: `(motor, idx, legales) → Carta`).

- `heuristicos.py`: Three stateless bots — `bot_conservador`, `bot_agresivo`, `bot_evasivo`.
- `bot_experto.py`: `BotExperto` — stronger heuristic using void-tracking and suit-lead logic.
- `bot_castigador.py`: `BotCastigador` — stateful per-hand bot, resets automatically each mano.

**`src/rllib/`** — RLlib pipeline components (new).

- `model.py`: `HeartsActionMaskModel(TorchModelV2)` — MLP `input → 512 → 512 → 256` with action masking via logit clamping. Registered as `"hearts_model"` in `ModelCatalog`.
- `config.py`: `build_ppo_config()` — builds `PPOConfig` (old API stack). Key Ray 2.55.1 params: `minibatch_size` (was `sgd_minibatch_size`), `num_epochs` (was `num_sgd_iter`), `env_runners()` (was `rollouts()`), `preprocessor_pref=None` for Dict obs.
- `opponent_pool.py`: `OpponentPool` — phase-based factory (0–5%: bots, 5–15%: expert, 15–40%: expert+snapshots, 40–70%: snapshots, 70–100%: pure snapshots). `SnapshotPolicy` wraps a Ray policy into the `(motor, idx, legales) → Carta` signature.
- `callbacks.py`: `HeartsCallbacks(DefaultCallbacks)` — logs episode metrics to `eval_log.jsonl`.
- `utils.py`: snapshot save/load/prune utilities.

**`src/torneo/`** — Evaluation infrastructure.

- `elo.py`: Least-squares Elo (no order bias). Invokable as `python -m src.torneo.elo`.
- `evaluacion.py`: Win-rate evaluation against the three heuristic bots.
- `normalizacion.py`: VecNormalize utilities (legacy SB3, kept for golden baselines).

**`src/entrenamiento/`** — Legacy SB3 training config (kept for reference).

- `config.py`: `Hiperparametros` dataclass and path helpers.

**`src/mcts/`** — PIMC (Perfect Information Monte Carlo) oracle and dataset tools.

- `pimc.py`: `pimc_mejor_jugada()` — for each legal card, simulates `num_mundos` random completions using `bot_evasivo`, returns the card minimizing expected score.
- `dataset.py`: Generates `(obs, action)` pairs for BC pretraining.

**`train_rllib.py`** — Main RLlib training pipeline. Phase progression, snapshot management, self-play pool updates via `algo.env_runners`.

**`train_self_play.py`** — Legacy constants kept for reference only (not used by RLlib pipeline).

### Observation Vector (224 dims, v11 standard = `DIM_ENTORNO`)

| Range | Content |
|-------|---------
| `[0:52]` | Agent's hand (one-hot) |
| `[52:104]` | Current trick / mesa (one-hot) |
| `[104:156]` | Played cards / cementerio (one-hot) |
| `[156:172]` | Known voids: 4 players × 4 suits |
| `[172:176]` | Historical scores normalized /100 |
| `[176:180]` | Current hand points normalized /26 |
| `[180]` | Hearts broken (0/1) |
| `[181]` | Position in trick (0.0, 0.33, 0.66, 1.0) |
| `[182:187]` | Q♠ tracker (one-hot, 5 states) |
| `[187:190]` | Strategic flags: `pozo_viable`, `debo_arriesgar`, `puedo_alimentar` |
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

Golden (frozen) baselines: `models/v5_golden/` (190-dim, ~1500 Elo) and `models/v7_golden/` (194-dim, 1723 Elo).

### Key Design Constraints

- **SSOT**: Reward values → `RewardConfig` (recompensas.py). Observation dims → `dimensiones.py`. Never hardcode `220` or `224`.
- **Action masking**: `CorazonesEnvRLlib` returns `{"obs": ..., "action_mask": ...}`. `HeartsActionMaskModel` clamps illegal actions to `-1e9`. `preprocessor_pref=None` is required in `env_runners()` to prevent RLlib from flattening the Dict obs.
- **Ray 2.55.1 API changes** (vs older docs): `rollouts()` → `env_runners()`, `sgd_minibatch_size` → `minibatch_size`, `num_sgd_iter` → `num_epochs`, `.build()` → `.build_algo()`.
- **Self-play pool**: phase-based opponent mixing in `OpponentPool`. Pool capped at 50 most-recent snapshots.
- **`motor.repartir()` clears `bazas_ganadas`**: critical for determinism across episodes (bug fix applied to `src/dominio/motor.py`).
- **Elo is the primary metric**; win rate against bots is a weaker signal.
- **Diagnostic alerts** written to `eval_log.jsonl` with `"tipo": "alerta"` when metrics cross thresholds.
