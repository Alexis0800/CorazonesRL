# Corazones RL — Instrucciones de Setup y Entrenamiento

> **Versión actual:** v5 (190 dims) — Fase 5C
> **Mejor modelo:** v5/snapshot_0010400000 — Elo 1604 (combinado v5+v6)
> **Tests:** 251/251 pasando
> **Fecha:** 2026-06-11

---

## Requisitos

- Python 3.12 o 3.13
- Windows, Linux o macOS
- 8+ GB RAM recomendado
- CPU multinúcleo o GPU CUDA

---

## Instalación (primera vez)

```powershell
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
# Deben pasar 251 tests
```

---

## Estructura del proyecto

```
corazones-neuralnetwork/
├── src/
│   ├── carta.py              # Clase Carta (52 cartas cacheadas)
│   ├── jugador.py            # Clase Jugador
│   ├── baraja.py             # Clase Baraja
│   ├── motor.py              # Motor del juego (Módulo 1)
│   ├── entorno.py            # CorazonesEnv Gymnasium 190d (Módulo 2)
│   ├── entorno_multi.py      # PettingZoo AEC 190d
│   ├── bots.py               # 3 bots heurísticos
│   ├── red.py                # MLP [256,256,128] para MaskablePPO
│   ├── evaluacion.py         # Evaluación multi-nivel + Torneo Elo
│   └── elo_torneo.py         # Sistema Elo entre snapshots
├── tests/
│   ├── test_modulo1.py       # Tests del motor
│   ├── test_modulo2.py       # Tests del entorno 190d
│   ├── test_modulo3.py       # Tests del agente RL + pipeline
│   ├── test_entorno_v5.py    # Tests features estratégicas v5
│   ├── test_evaluacion.py    # Tests sistema evaluación
│   ├── test_elo.py           # Tests sistema Elo
│   └── test_asesor*.py       # Tests del asesor
├── train_self_play.py        # Pipeline de entrenamiento
├── evaluar_modelo.py         # Script de evaluación (legacy)
├── asesor_partida.py         # Asesor interactivo (REPL)
├── asesor_carta.py           # Construcción de observaciones
├── test_stress.py            # Stress test (10k manos)
├── modelos_historicos/       # Snapshots (gitignored)
│   ├── v2/                   # Snapshots viejos 187d (obsoletos)
│   ├── v5/                   # Snapshots v5 (190d) ← ACTIVO
│   └── v6/                   # Snapshots v6 (190d, paralelo)
├── logs/                     # TensorBoard logs (gitignored)
├── vecnormalize/             # VecNormalize stats (gitignored)
│   ├── v2/                   # viejos 187d
│   └── v5/                   # actual 190d
└── INSTRUCCIONES.md          # Este archivo
```

---

## Espacio de observación (v5 — 190 dimensiones)

```
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
```

---

## 🚀 Comandos principales

### Verificar todo

```powershell
python -m pytest tests/ -q        # 251 tests
python test_stress.py             # 10k manos en < 3s
```

### Entrenar

```powershell
# Continuar desde el último snapshot v5 (auto-detecta):
python train_self_play.py --self-play --steps 5000000 --eval-every 5

# Desde cero (modelo nuevo, pesos aleatorios):
python train_self_play.py --self-play --from-scratch --steps 5000000 --eval-every 5

# Reanudar desde snapshot específico:
python train_self_play.py --self-play --resume modelos_historicos/v5/snapshot_0010400000 --steps 5000000 --eval-every 5

# Entrenamiento paralelo (directorio v6 independiente):
python train_self_play.py --self-play --v6 --base-model modelos_historicos/v5/snapshot_0008400000 --steps 5000000 --eval-every 5
```

**Flags clave:**

| Flag | Default | Descripción |
|---|---|---|
| `--self-play` | off | Snapshots históricos como oponentes |
| `--from-scratch` | off | Modelo nuevo (pesos aleatorios) |
| `--steps` | 2M | Pasos totales |
| `--snapshot-every` | 100K | Guardar snapshot cada N pasos |
| `--eval-every` | 5 | Evaluar cada N snapshots |
| `--prob-bot` | 0.10 | % bots (10% = 90% self-play) |
| `--v6` | off | Guardar en directorio v6 (paralelo) |
| `--device` | cpu | cpu o cuda |

### Evaluar modelos

```powershell
# Rápido (100 partidas contra bots):
python -m src.evaluacion --ruta modelos_historicos/v5/snapshot_0010400000

# Con Self-Play:
python -m src.evaluacion --ruta modelos_historicos/v5/snapshot_0010400000 --self-play

# Alta precisión (1000 partidas):
python -m src.evaluacion --ruta modelos_historicos/v5/snapshot_0010400000 --partidas 1000
```

### Torneo Elo

```powershell
# Torneo Elo PURO v5 (recomendado, 30 partidas):
python -m src.evaluacion --elo --elo-puro --elo-partidas 30 --elo-min-paso 5000000

# Torneo Elo PURO v6:
python -m src.evaluacion --elo --elo-puro --elo-partidas 30 --elo-min-paso 8000000 --elo-dir modelos_historicos/v6

# Torneo COMBINADO v5 + v6 (← para comparar versiones):
python -m src.evaluacion --elo --elo-puro --elo-partidas 20 --elo-min-paso 8000000 --elo-max-snaps 12 --elo-dirs modelos_historicos/v5 modelos_historicos/v6
```

### Asesor de partida (REPL interactivo)

```powershell
python asesor_partida.py --modelo modelos_historicos/v5/snapshot_0010400000
```

### Monitorear entrenamiento

```powershell
tensorboard --logdir ./logs
# Abrir http://localhost:6006
```

---

## Learning Rate Schedule (3 fases)

| Fase | Pasos | lr | ent_coef | Objetivo |
|---|---|---|---|---|
| 1 | 0 – 2M | 1e-4 | 0.08 | Exploración |
| 2 | 2M – 5M | 5e-5 | 0.05 | Consolidación |
| 3 | 5M+ | 2e-5 | 0.03 | Fine-tuning |

---

## Métricas clave

| Métrica | Interpretación |
|---|---|
| `train/explained_variance` | >0.6 bueno, >0.8 excelente |
| `train/approx_kl` | <0.02 estable |
| `train/entropy_loss` | Negativo = explorando |
| `train/value_loss` | <0.01 bueno |

---

## Resultados históricos (torneo Elo combinado)

| Pos | Snap | Origen | Paso | Elo |
|---|---|---|---|---|
| 1 | 13.4M | v6 | 13,400,000 | 1679 |
| 2 | 13.3M | v6 | 13,300,000 | 1676 |
| 3 | **10.4M** | **v5** | **10,400,000** | **1604** |
| 4 | 13.2M | v6 | 13,200,000 | 1603 |
| 5 | 13.1M | v6 | 13,100,000 | 1519 |

> **Conclusión:** v5 es ~2.5x más eficiente. Continuar solo v5.

---

## Transferir entre PCs

Ver [CONTINUAR_EN_OTRA_PC.md](CONTINUAR_EN_OTRA_PC.md). Resumen:

```powershell
# En PC origen:
.\comprimir_modelos.ps1

# Copiar corazones_modelos.zip → nueva PC y extraer
```
