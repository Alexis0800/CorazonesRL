# Corazones RL — Agente de Hearts por Reinforcement Learning

Agente de RL para el juego de cartas **Corazones (Hearts)** que aprende a jugar
**partidas completas a 100 puntos** (con pase de cartas) mediante una combinación
de **Behavioral Cloning desde un oráculo PIMC** + **PPO con self-play diverso**
(Ray RLlib). Incluye un **copiloto** que te recomienda qué pasar y qué jugar en
partidas reales.

> Estado actual: **v10c** es el mejor modelo — partida completa + pase + self-play
> consistente. Ver `models/produccion/`. Documentación de diseño en
> `docs/Rediseño_v10_partida_completa.md`. Próximos pasos en `docs/ROADMAP.md`.

---

## Resultados (vs 3 BotExperto, 150 partidas; azar = win 0.25 / top2 0.50)

| Modelo | win vs experto | top2 vs experto | win global | top2 global |
|---|---|---|---|---|
| v9 (legacy, mano aislada) | — (perdía al experto) | — | — | — |
| **v10c (campeón, con pase)** | **0.66** | **0.83** | 0.76 | 0.93 |
| v10_nopase (campeón sin pase) | 0.49 | 0.81 | 0.70 | 0.925 |
| BC-solo (sin PPO) | ~0.62 | ~0.87 | ~0.69 | ~0.90 |
| PIMC-experto (oráculo, referencia) | 0.83 | 0.97 | — | — |

El agente generaliza a 7 arquetipos distintos (evasivo, conservador, agresivo,
experto, castigador, lunático/cazador-de-pozo, atacante-del-líder) y **defiende
el pozo** (top2 ~0.96 vs el cazador de pozo).

---

## Cómo funciona (pipeline)

```
1. MOTOR (dominio puro)         → reglas de Hearts: bazas, pase, pozo, puntuación.
2. ENTORNO RLlib (partida)      → 1 episodio = 1 PARTIDA completa a 100 pts.
   - Observación 228 dims (juego + 4 features de pase).
   - Recompensa = puesto final (R_terminal) + shaping PBRS (no-farmeable).
   - Pase = 3 sub-decisiones de carta al inicio de cada mano.
3. ORÁCULO PIMC                 → Perfect-Information Monte Carlo (mejor jugada).
4. BEHAVIORAL CLONING           → red imita a PIMC (dataset (obs, acción)).
5. PPO FINE-TUNE + SELF-PLAY    → parte del BC, refina con pool de arquetipos
                                  humanos diversos (generaliza, robustez).
```

### Claves de diseño
- **Episodio = partida completa** (no mano aislada): permite estrategia de
  marcador real (atacar al líder, timing del pozo). El marcador persiste entre manos.
- **Recompensa PBRS** (Ng-Harada-Russell): `F = γΦ(s') − Φ(s)` sobre el marcador.
  Garantiza que el agente no "farmea" señales intermedias y optimiza ganar la partida.
- **BC desde PIMC**: el salto que rompió el estancamiento — la red estática captura
  gran parte de la fuerza de búsqueda de PIMC.
- **Self-play con pool diverso**: rivales = arquetipos humanos (no solo BotExperto)
  → el agente generaliza a estilos variados, no se sobreajusta a un oponente.
- **Pase neuronal en self-play** (v10c): los rivales-snapshot pasan con su propia
  red → self-play 100% consistente (simetría sana 0.21≈0.25).

---

## Estructura del código

```
src/
├── dominio/            Lógica pura del juego (sin RL)
│   ├── carta.py, baraja.py, jugador.py
│   └── motor.py        MotorCorazones: bazas, PASE (direccion/ejecutar_pase), pozo, puntuación, partida completa
├── entorno/            Gymnasium env + observación + recompensa
│   ├── corazones_rllib.py   CorazonesEnvRLlib (partida completa + fase de pase)
│   ├── observacion.py       ObservacionBuilder (228 dims, DIM_V12)
│   ├── dimensiones.py       SSOT de dimensiones (DIM_V12=228)
│   └── recompensas_partida.py  R_terminal + PBRS (v10 SSOT)
├── agentes/            Bots (arquetipos humanos), todos con pasar() por perfil
│   ├── heuristicos.py  conservador / agresivo / evasivo
│   ├── bot_experto.py  fuerte (conteo, Q♠, pozo, alimentar, duck, sangrar picas)
│   ├── bot_castigador.py  presiona picas para forzar la Q♠
│   ├── bot_lunatico.py    cazador de pozo
│   ├── bot_atacante_lider.py  carga puntos al líder
│   └── pase.py         pase_heuristico (de los MD de estrategia)
├── mcts/               Oráculo PIMC
│   ├── pimc.py         pimc_mejor_jugada (+ rollouts mixtos de arquetipos)
│   └── dataset.py, pimc_recursivo.py
├── rllib/              Pipeline RLlib
│   ├── model.py        HeartsActionMaskModel (MLP) / HeartsLSTMModel
│   ├── config.py       build_ppo_config (PPO old API stack)
│   ├── opponent_pool.py  OpponentPool + SnapshotPolicy (con pase neuronal)
│   ├── eval_bots.py    evaluación en partidas completas (vía env)
│   ├── callbacks.py, utils.py (elite, snapshots)
└── torneo/             Elo, evaluación
```

### Scripts de nivel superior
| Script | Para qué |
|---|---|
| `train_rllib.py` | Entrenar (PPO + self-play). Flags clave: `--con-pase`, `--bc-weights`, `--pool-diverso`, `--ancla-experto`. |
| `generar_dataset_bc.py` | Generar dataset BC etiquetado con PIMC (con/sin pase). |
| `entrenar_bc.py` | Entrenar la política por Behavioral Cloning. |
| `evaluar_final.py` | Evaluación final + recomendaciones de ajuste. |
| `monitorear.py` | Ver el avance del entrenamiento en vivo. |
| `jugar_modelo.py` | Ver al modelo jugar (traza) o jugar tú contra él. |
| `recomendador.py` | **Copiloto**: te recomienda qué pasar/jugar en una partida real. |

---

## Uso rápido

```bash
# Activar venv (Python 3.12; Ray 2.55.1)
.venv\Scripts\Activate.ps1

# Copiloto con el mejor modelo
python recomendador.py --modelo models/produccion/v10c_campeon

# Ver jugar al modelo
python jugar_modelo.py --modelo models/produccion/v10c_campeon --ver

# Tests
python -m pytest tests/ -q
```

### Reproducir el entrenamiento de v10c (resumen)
```bash
# 1. dataset BC con pase (etiqueta: pase=BotExperto.pasar, juego=PIMC mixto)
python generar_dataset_bc.py --partidas 600 --mundos 60 --con-pase \
    --teacher mixto --oponentes diverso --out datasets/bc.npz
# 2. BC
python entrenar_bc.py --dataset datasets/bc.npz --con-pase --out models/bc/bc.pkl
# 3. PPO fine-tune con pase + self-play diverso (snapshots pasan neuronalmente)
python train_rllib.py --total-steps 20000000 --workers 8 --con-pase \
    --bc-weights models/bc/bc.pkl --pool-diverso --ancla-experto \
    --lr 1e-4 --lr-end 5e-5 --entropy-coeff 0.02 --output-dir models/v10c
```

---

## Documentación relacionada
- `docs/Hearts.md`, `docs/Estrategias Avanzadas.md` — reglas y estrategia (base de los bots).
- `docs/Rediseño_v10_partida_completa.md` — diseño del rediseño v10 (MDP, PBRS).
- `docs/ROADMAP.md` — próximos pasos (copiloto, app móvil, dataset humano, on-device).
- `models/produccion/README.md` — los mejores modelos y cómo usarlos.
