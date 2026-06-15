# Corazones RL — Instrucciones de Setup y Entrenamiento

> **Versión actual:** v8 (194 dims) — Entrenamiento autónomo
> **Mejor modelo:** v7_golden — Elo 1723
> **Tests:** 286/286 pasando
> **Fecha:** 2026-06-15

---

## Requisitos

- Python 3.12 o 3.13
- Windows, Linux o macOS
- 8+ GB RAM recomendado
- CPU multinúcleo o GPU (CUDA / DirectML / Intel XPU)

---

## Instalación (primera vez)

`powershell
# 1. Clonar el repositorio
git clone <url-del-repo> corazones-neuralnetwork
cd corazones-neuralnetwork

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# 4. Instalar dependencias
pip install torch stable-baselines3 sb3-contrib pettingzoo gymnasium numpy pytest tensorboard tqdm

# 5. Verificar
python -m pytest tests/ -q
# Deben pasar 286 tests
`

---

## Estructura del proyecto

`
corazones-neuralnetwork/
├── src/
│   ├── dominio/               # Reglas del juego (sin deps RL)
│   │   ├── carta.py           # Value Object inmutable (52 cartas cacheadas)
│   │   ├── baraja.py          # Baraja francesa
│   │   ├── jugador.py         # Jugador (mano, puntuación)
│   │   └── motor.py           # Motor del juego (bazas, reglas, 4 filtros)
│   ├── entorno/               # RL Environment (Gymnasium + PettingZoo)
│   │   ├── observacion.py     # Observation builder 194-dim (SRP)
│   │   ├── recompensas.py     # Reward config + calculator (SRP)
│   │   ├── single_agent.py    # CorazonesEnv (Gymnasium)
│   │   └── multi_agent.py     # CorazonesAEC (PettingZoo)
│   ├── agentes/               # Estrategias de juego
│   │   ├── heuristicos.py     # 3 bots (conservador, agresivo, evasivo)
│   │   └── politica_rl.py     # Adaptador MaskablePPO → Strategy
│   ├── red.py                 # Red neuronal MLP [256,256,128]
│   ├── torneo/                # Evaluación y rating Elo
│   │   ├── elo.py             # Sistema Elo convergente (least-squares)
│   │   ├── evaluacion.py      # Evaluación multi-nivel (bots/snapshots/escenarios)
│   │   └── normalizacion.py   # VecNormalize utilities
│   ├── entrenamiento/         # Training pipeline
│   │   ├── config.py          # SSOT: paths + hiperparámetros
│   │   └── self_play.py       # Fictitious Self-Play environment builder
│   ├── cli/                   # Interfaces de usuario
│   │   ├── jugar.py           # Juego interactivo contra el modelo
│   │   └── evaluar.py         # Evaluación standalone
│   └── __init__.py            # Re-exports públicos
├── tests/                     # Tests espejo de src/
│   ├── dominio/               # test_modulo1.py
│   ├── entorno/               # test_modulo2.py, test_entorno_v5.py, test_entorno_v6.py
│   ├── agentes/               # test_modulo3.py
│   ├── torneo/                # test_elo.py, test_evaluacion.py, test_auto_v6.py
│   └── cli/                   # test_asesor.py, test_asesor_partida.py, test_jugar_contra_modelo.py
├── modelos/                   # Modelos entrenados (versionado)
│   └── {version}/
│       ├── config.json        # Metadatos de la versión
│       ├── snapshots/         # Checkpoints de entrenamiento
│       ├── elite/             # Mejores snapshots (torneo Elo)
│       ├── vecnorm/           # Estadísticas VecNormalize
│       └── torneos/           # Resultados de torneos Elo
├── scripts/                   # Entry points CLI
│   ├── entrenar.py            # Entrenamiento autónomo
│   ├── evaluar.py             # Evaluación de modelo
│   ├── jugar.py               # Juego interactivo
│   └── comprimir_modelos.ps1  # Script de transferencia
├── docs/                      # Documentación
│   ├── planes/                # Planes de arquitectura
│   ├── historico/             # Documentos archivados
│   ├── INSTRUCCIONES.md       # Este archivo
│   └── STATUS.md              # Estado actual del proyecto
├── .gitignore
├── train_auto_v6.py           # Training autónomo v6
└── train_self_play.py         # Self-Play pipeline
`

---

## Espacio de observación (v6+ — 194 dimensiones)

`
[0:52]    Mano del agente (one-hot)
[52:104]  Mesa actual / baza en curso
[104:156] Cementerio / cartas jugadas
[156:172] Vacíos conocidos (4 jugadores × 4 palos)
[172:176] Puntajes históricos (/100)
[176:180] Puntos mano actual (/26)
[180]     Corazones rotos
[181]     Posición en la baza
[182:187] Rastreador Dama de Picas (5 estados)
[187]     pozo_viable — ¿shooting the moon viable?
[188]     debo_arriesgar — ¿estoy forzado a arriesgar?
[189]     puedo_alimentar — ¿puedo dar puntos a un rival?
[190]     all_void_treboles — ¿todos los rivales vacíos en tréboles?
[191]     all_void_diamantes
[192]     all_void_corazones
[193]     all_void_picas
`

---

## Comandos principales

### Entrenamiento

`powershell
# Desde cero
python train_auto_v6.py --total-steps 20000000 --output-dir modelos/v9

# Reanudar desde snapshot
python train_auto_v6.py --resume modelos/v8/elite/snapshot_0017900000 --total-steps 25000000 --output-dir modelos/v9

# Con GPU Intel Arc
python train_auto_v6.py --total-steps 20000000 --device dml --output-dir modelos/v9
`

### Evaluación

`powershell
# Win rate contra bots (rápido)
python scripts/evaluar.py --modelo v5_golden.zip --partidas 500

# Torneo Elo entre snapshots
python -m src.torneo.elo --directorio modelos/v8/elite --partidas 50 --elo-puro --incluir-bots
`

### Jugar contra el modelo

`powershell
python scripts/jugar.py --modelo modelos/v8/elite/snapshot_0017900000.zip
`

### Tests

`powershell
# Todos los tests
python -m pytest tests/ -q

# Solo dominio
python -m pytest tests/dominio/ -q

# Solo entorno
python -m pytest tests/entorno/ -q
`

---

## Versiones de modelos

| Versión | Estado | Dims | Elo | Descripción |
|---------|--------|------|-----|-------------|
| 5_golden | baseline | 190 | ~1500 | Mejor v5, baseline dorado |
| 6 | archivada | 194 | — | Self-play real, all_void features |
| 7 | archivada | 194 | — | Decaimiento de bots 50%→20% |
| 7_cont | archivada | 194 | — | Continuación desde v7 |
| 7_golden | baseline | 194 | 1723 | Mejor v7, baseline dorado |
| 8 | **activa** | 194 | — | Entrenamiento autónomo con torneos Elo |

---

## Hiperparámetros actuales (v8)

`python
learning_rate = 1e-5       # LR bajo para fine-tuning
ent_coef = 0.15            # Entropía alta para exploración
n_steps = 4096             # Pasos por rollout
batch_size = 512           # Mini-batch size
arquitectura = [256, 256, 128]  # MLP
prob_bot = 0.50 → 0.20    # Decaimiento coseno del self-play
`

---

## Transferir entre PCs

Ver [CONTINUAR_EN_OTRA_PC.md](CONTINUAR_EN_OTRA_PC.md) para instrucciones de compresión y transferencia de modelos.
