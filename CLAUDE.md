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
python train_auto_v6.py --total-steps 20000000 --output-dir modelos/v9

# Resume from a checkpoint
python train_auto_v6.py --resume modelos/v7_golden/snapshots/snapshot_0014900000 --total-steps 25000000 --output-dir modelos/v7_cont

# With Intel Arc GPU
python train_auto_v6.py --total-steps 20000000 --device dml --output-dir modelos/v9
```

### Evaluation & Play
```bash
# Elo tournament between snapshots
python -m src.torneo.elo --directorio modelos/v8/elite --partidas 50 --elo-puro --incluir-bots

# Evaluate win rate against bots
python scripts/evaluar.py --modelo modelos/v8/elite/snapshot_0015000000.zip

# Play interactively against the model
python scripts/jugar.py --modelo modelos/v8/elite/snapshot_0015000000.zip
```

## Architecture

### Layer Structure

**`src/dominio/`** — Pure game logic, no RL dependencies.
- `carta.py`: Immutable `Carta` value object with `id`, `palo`, `valor`, `puntos`, `es_corazon`, `es_dama_de_picas`.
- `baraja.py`: French deck, shuffling, dealing.
- `jugador.py`: Player state (hand, `bazas_ganadas`, accumulated score).
- `motor.py`: `MotorCorazones` — full game loop (tricks, scoring, `obtener_jugadas_legales()` with 4-filter cascade, shooting the moon, broken hearts, Q♠).

**`src/entorno/`** — Gymnasium RL environment.
- `single_agent.py`: `CorazonesEnv(gym.Env)` — the primary training environment. Wraps `MotorCorazones`, handles action masking, calls opponent policies.
- `multi_agent.py`: `CorazonesAEC` — PettingZoo AEC environment for multi-agent experiments.
- `observacion.py`: `ObservacionBuilder` — single source of truth for the 194-dim observation vector. Call `construir()` for the full observation; `construir_desde_motor()` for a minimal observation (used by opponent snapshots in self-play).
- `recompensas.py`: `RewardConfig` (frozen dataclass) and `CalculadoraRecompensas` — all reward logic isolated here (SRP).

**`src/agentes/`** — Agent strategies (Strategy pattern: `(motor, idx, legales) → Carta`).
- `heuristicos.py`: Three bots — `bot_conservador` (play lowest), `bot_agresivo` (play highest), `bot_evasivo`.
- `politica_rl.py`: `PoliticaSB3` — adapts a `MaskablePPO` model to the policy callable signature.

**`src/red.py`** — `CorazonesFeatureExtractor`: MLP `input → 256 → 256 → 128` (ReLU), compatible with `MaskablePPO`. `obtener_policy_kwargs()` returns `policy_kwargs` for SB3.

**`src/torneo/`** — Evaluation infrastructure.
- `elo.py`: Least-squares Elo (no order bias). Runs round-robin tournaments; snapshots and bots both receive ratings. Invokable as `python -m src.torneo.elo`.
- `evaluacion.py`: Win-rate evaluation against the three heuristic bots.
- `normalizacion.py`: VecNormalize loading/saving utilities.

**`src/entrenamiento/`** — Training plumbing.
- `config.py`: SSOT for all paths and hyperparameters (`Hiperparametros` dataclass, `HP_DEFAULT`). Also provides `directorio_*_version(version)` path helpers.
- `self_play.py`: `crear_entorno_self_play()` — builds a `CorazonesEnv` with mixed opponents (bots + historical snapshots loaded from `modelos/{version}/snapshots/`).

**`train_auto_v6.py`** — Main autonomous training pipeline. Manages the full loop: snapshot saving, VecNormalize, cosine decay of `prob_bot` (50%→20%), LR schedule (3 phases), async Elo tournaments, elite snapshot pruning.

**`train_self_play.py`** — Legacy constants and helpers still imported by `train_auto_v6.py`.

### Observation Vector (194 dims, v6)

| Range | Content |
|-------|---------|
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

All positions are **relative to the agent** (`(player_idx - agent_idx) % 4`).

### Model Storage Layout

```
modelos/{version}/
├── config.json          # Version metadata
├── snapshots/           # Training checkpoints (snapshot_NNNNNNNNN.zip)
├── vecnorm/             # VecNormalize stats per snapshot (*_vecnorm.pkl)
├── elite/               # Best N snapshots after Elo tournament
└── torneos/             # Elo results (elo_paso_*.txt, eval_log.jsonl)
```

Golden (frozen) baselines: `modelos/v5_golden/` (190-dim, ~1500 Elo) and `modelos/v7_golden/` (194-dim, 1723 Elo).

### Key Design Constraints

- **VecNormalize coupling**: every `MaskablePPO` snapshot has a paired `_vecnorm.pkl`. Loading a model without its VecNormalize degrades play quality. The `_Modelo190Wrapper` in `elo.py` handles cross-gen compatibility for 190-dim vs 194-dim models.
- **Action masking**: `CorazonesEnv` always provides an action mask via `action_masks()`. Use `MaskablePPO` (not `PPO`) and `MaskableEvalCallback`.
- **Self-play pool**: snapshots older than `min_snapshot_steps` (500k) are excluded from the opponent pool. Pool is capped at `max_snapshots` (50) most-recent entries.
- **Elo is the primary metric**; win rate against bots is a weaker signal because the bots are simple.
