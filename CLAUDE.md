# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Reinforcement learning agent for the card game Hearts (Corazones), using MaskablePPO from `sb3-contrib`. The agent trains via Fictitious Self-Play against a pool of its own historical snapshots and three heuristic bots, evaluated with a least-squares Elo system.

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

### Training

```bash
# Train from scratch
python train.py --total-steps 20000000 --output-dir models/v9

# Resume from a checkpoint
python train.py --resume models/v7_golden/snapshots/snapshot_0014900000 --total-steps 25000000 --output-dir models/v7_cont

# With Intel Arc GPU
python train.py --total-steps 20000000 --device dml --output-dir models/v9
```

### Behavioral Cloning (BC) Pretraining

```bash
# Generate PIMC dataset (oracle-quality (obs, action) pairs)
python scripts/generar_dataset_bc.py --partidas 5000 --output datasets/mcts_50k.npz

# Supervised pretraining on the dataset (warm-start for RL)
python train_bc.py --dataset datasets/mcts_50k.npz --output models/bc_pretrain --epochs 30 --lr 1e-3 --batch 512 --obs-dim 220

# Resume RL fine-tuning from the BC checkpoint
python train.py --resume models/bc_pretrain --total-steps 20000000 --output-dir models/v9
```

### Evaluation & Play

```bash
# Elo tournament between snapshots
python -m src.torneo.elo --directorio models/v8/elite --partidas 50 --elo-puro --incluir-bots

# Evaluate win rate against bots
python scripts/evaluar.py --modelo models/v8/elite/snapshot_0015000000.zip

# Play interactively against the model
python scripts/jugar.py --modelo models/v8/elite/snapshot_0015000000.zip
```

## Architecture

### Layer Structure

**`src/dominio/`** — Pure game logic, no RL dependencies.

- `carta.py`: Immutable `Carta` value object with `id`, `palo`, `valor`, `puntos`, `es_corazon`, `es_dama_de_picas`.
- `baraja.py`: French deck, shuffling, dealing.
- `jugador.py`: Player state (hand, `bazas_ganadas`, accumulated score).
- `motor.py`: `MotorCorazones` — full game loop (tricks, scoring, `obtener_jugadas_legales()` with 4-filter cascade, shooting the moon, broken hearts, Q♠).

**`src/entorno/`** — Gymnasium RL environment.

- `single_agent.py`: `CorazonesEnv(gym.Env)` — the primary training environment. Wraps `MotorCorazones`, handles action masking, calls opponent policies. Uses `CalculadoraRecompensas` for all reward logic (delegated, not inline).
- `multi_agent.py`: `CorazonesAEC` — PettingZoo AEC environment for multi-agent experiments.
- `observacion.py`: `ObservacionBuilder` — SSOT for the **220-dim** observation vector (standard). Supports 190/194/220 via constructor parameter. Call `construir()` for full observation; `construir_desde_motor()` for minimal (used by opponent snapshots in self-play).
- `recompensas.py`: `RewardConfig` (frozen dataclass, **v12 SSOT**) and `CalculadoraRecompensas` — ALL reward logic centralized here (SRP). `CorazonesEnv` delegates every reward calculation to this module.
- `recompensas_minimal.py`: `RewardConfigMinimal` and `CalculadoraRecompensasMinimal` — alternative v13_minimal reward system with only 3 signals (captured points, distance reward, shooting moon). Kept separate from `recompensas.py` for A/B experimentation without cross-contamination.
- `dimensiones.py`: SSOT for observation dimensions (`DIM_V5=190`, `DIM_V6=194`, `DIM_V10=220`, `DIM_ENTORNO=220`, `DIM_ENTRENAMIENTO=220`, `DIMS_VALIDAS`). All modules import from here — no hardcoded `194` or `220` anywhere.

**`src/agentes/`** — Agent strategies (Strategy pattern: `(motor, idx, legales) → Carta`).

- `heuristicos.py`: Three stateless bots — `bot_conservador` (play lowest), `bot_agresivo` (play highest), `bot_evasivo`.
- `bot_experto.py`: `BotExperto` — stronger heuristic that uses void-tracking and suit-lead logic.
- `bot_castigador.py`: `BotCastigador` — stateful per-hand bot that aggressively leads/follows ♠ to punish the Q♠ holder. Resets automatically at each new hand.
- `politica_rl.py`: `PoliticaSB3` — adapts a `MaskablePPO` model to the policy callable signature.

**`src/red.py`** — `CorazonesFeatureExtractor`: MLP `input → 256 → 256 → 128` (ReLU), compatible with `MaskablePPO`. `obtener_policy_kwargs()` returns `policy_kwargs` for SB3.

**`src/torneo/`** — Evaluation infrastructure.

- `elo.py`: Least-squares Elo (no order bias). Runs round-robin tournaments; snapshots and bots both receive ratings. Invokable as `python -m src.torneo.elo`.
- `evaluacion.py`: Win-rate evaluation against the three heuristic bots.
- `normalizacion.py`: VecNormalize loading/saving utilities.

**`src/entrenamiento/`** — Training plumbing.

- `config.py`: SSOT for all paths and hyperparameters (`Hiperparametros` dataclass, `HP_DEFAULT`). Key v12 defaults: `vf_coef=0.25`, `max_grad_norm=0.3`, `dim_observacion=220`. Also provides `directorio_*_version(version)` path helpers.
- `self_play.py`: `crear_entorno_self_play()` — builds a `CorazonesEnv` with mixed opponents (bots + historical snapshots loaded from `models/{version}/snapshots/`).

**`src/mcts/`** — PIMC (Perfect Information Monte Carlo) oracle and dataset tools.

- `pimc.py`: `pimc_mejor_jugada()` — main oracle entry point. For each legal card, simulates `num_mundos` random completions of the hand using `bot_evasivo` as rollout policy, returns the card minimizing expected score.
- `pimc_recursivo.py`: Recursive determinization variant.
- `dataset.py`: Generates `(obs, action)` pairs by running PIMC on game states — the source data for BC pretraining.
- `analisis.py`: Post-hoc analysis of PIMC decisions.

**`src/cli/`** — Actual implementations for interactive play and evaluation. `scripts/jugar.py` and `scripts/evaluar.py` are thin entry-point wrappers around these.

- `jugar.py`: Full human-vs-model interactive loop with card visualization and legal-move prompting.
- `evaluar.py`: Win-rate evaluation runner.

**`train.py`** — Main autonomous training pipeline. Manages the full loop: snapshot saving, VecNormalize, cosine decay of `prob_bot` (50%→20%), LR schedule (3 phases), async Elo tournaments, elite snapshot pruning.

**`train_bc.py`** — Supervised BC pretraining: trains an actor head with CrossEntropyLoss on PIMC datasets. The output `.zip` can be loaded by `MaskablePPO.load()` as a warm-start for RL.

**`train_self_play.py`** — Legacy constants and helpers still imported by `train.py`.

**`scripts/`** — Analysis, diagnostic, and dataset-generation tools. Prefixed with `_` if not intended as direct entry points. Key scripts: `generar_dataset_bc.py`, `entrenar_bc.py`, `analizar_errores_bot.py`, `diagnosticar_modelo.py`.

**Legacy versioned modules** (`src/v2_1/`, `src/v2_ronda/`, `src/v3/`, `src/v3_1/`, `src/v4/`, `src/v5/`) — Superseded experiment branches kept for reference. Each contains its own `entorno.py`, `train.py`, `recompensas.py`, and sometimes `elo.py`. Do not modify; the canonical system is `src/entorno/` + `train.py`.

### Observation Vector (220 dims, v12 standard)

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

- **Single Source of Truth (SSOT)**: Reward values live ONLY in `RewardConfig` (recompensas.py). Observation dimensions live ONLY in `dimensiones.py`. No magic numbers `194` or `220` anywhere in the codebase — always import `DIM_ENTORNO` or `DIM_ENTRENAMIENTO`.
- **VecNormalize coupling**: every `MaskablePPO` snapshot has a paired `_vecnorm.pkl`. Loading a model without its VecNormalize degrades play quality. The `_Modelo190Wrapper` in `elo.py` handles cross-gen compatibility for 190-dim vs 220-dim models.
- **Action masking**: `CorazonesEnv` always provides an action mask via `action_masks()`. Use `MaskablePPO` (not `PPO`) and `MaskableEvalCallback`.
- **Self-play pool**: snapshots older than `min_snapshot_steps` (500k) are excluded from the opponent pool. Pool is capped at `max_snapshots` (50) most-recent entries.
- **Elo is the primary metric**; win rate against bots is a weaker signal because the bots are simple.
- **Diagnostic alerts** are written to `eval_log.jsonl` with `"tipo": "alerta"` when metrics cross thresholds (entropy, KL, value_loss, explained_variance).
