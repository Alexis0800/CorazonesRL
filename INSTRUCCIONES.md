# Corazones RL — Instrucciones de Setup y Entrenamiento

## Requisitos

- Python 3.12 o 3.13
- Windows, Linux o macOS
- 8+ GB RAM recomendado
- CPU multinúcleo (20 cores ideal) o GPU CUDA

---

## Instalación (primera vez)

```bash
# 1. Clonar el repositorio
git clone <url-del-repo> corazones-neuralnetwork
cd corazones-neuralnetwork

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Instalar dependencias
pip install torch stable-baselines3 sb3-contrib pettingzoo gymnasium numpy pytest tensorboard tqdm rich

# 5. Verificar instalación
python -m pytest tests/ -q
# Deben pasar 97 tests
```

---

## Estructura del proyecto

```
corazones-neuralnetwork/
├── src/
│   ├── carta.py          # Clase Carta (52 cartas cacheadas)
│   ├── jugador.py        # Clase Jugador
│   ├── baraja.py         # Clase Baraja
│   ├── motor.py          # Motor del juego (Módulo 1)
│   ├── entorno.py        # CorazonesEnv Gymnasium (Módulo 2)
│   ├── entorno_multi.py  # PettingZoo AEC (Módulo 3)
│   ├── bots.py           # 3 bots heurísticos
│   └── red.py            # MLP [256,256,128] para MaskablePPO
├── tests/
│   ├── test_modulo1.py   # 47 tests del motor
│   ├── test_modulo2.py   # 31 tests del entorno
│   └── test_modulo3.py   # 19 tests del agente RL
├── train_self_play.py    # Pipeline de entrenamiento
├── evaluar_modelo.py     # Script de evaluación
├── test_stress.py        # Stress test (10k manos)
├── modelos_historicos/   # Snapshots de entrenamiento (gitignored)
├── logs/                 # TensorBoard logs (gitignored)
├── vecnormalize/         # Estadísticas VecNormalize (gitignored)
└── docs/                 # Documentación del proyecto
```

---

## Sistema de Recompensas (versión final)

### Recompensa terminal — GANAR la partida es lo que importa

| Posición | Recompensa |
|---|---|
| 1º lugar | **+500** |
| 2º lugar | +200 |
| 3º lugar | -200 |
| 4º lugar | **-500** |

### Recompensa por eventos (reward shaping)

| Evento | Recompensa |
|---|---|
| Recibir un corazón | -1.0 |
| Recibir Dama de Picas | -10.0 |
| Pleno (Shooting the Moon) | +50.0 |
| No ganar baza con puntos | +0.5 |
| Descartar corazón sin llevárselo | +0.3 |
| Descartar Dama de Picas sin llevársela | +3.0 |
| Ganar baza sin puntos | -0.15 |
| Ganar la mano (menos puntos) | +2.0 |
| Perder la mano (más puntos) | -2.0 |

### Diseño

El gradiente entre 1º (+500) y 2º (+200) es de **300 puntos**, 5× mayor que antes. Esto obliga al agente a priorizar ganar la partida por encima de maximizar recompensas de shaping.

---

## Comandos

### Verificar todo

```bash
python -m pytest tests/ -v        # 97 tests
python test_stress.py             # 10k manos en < 3s
```

### Evaluar un modelo

```bash
python evaluar_modelo.py --ruta modelos_historicos/snapshot_0002000000 --partidas 200
```

### Entrenar desde cero (Fase 1 — contra bots)

```bash
python train_self_play.py --steps 2000000 --snapshot-every 50000
```

### Entrenar Self-Play (Fase 2 — contra snapshots históricos)

```bash
python train_self_play.py --self-play --steps 3000000 --snapshot-every 50000
```

### Reanudar desde un snapshot

```bash
python train_self_play.py --self-play --resume modelos_historicos/snapshot_0002000000 --steps 3000000
```

### Monitorear en tiempo real (otra terminal)

```bash
tensorboard --logdir ./logs
# Abrir http://localhost:6006
```

---

## Transferir modelos entre PCs

Los modelos están en `modelos_historicos/` y **no se suben a git** (son binarios grandes).

Para transferirlos:

```bash
# En la PC origen, comprimir modelos
Compress-Archive -Path modelos_historicos\*.zip, vecnormalize\*.pkl -DestinationPath modelos_backup.zip

# Copiar modelos_backup.zip a la nueva PC y extraer
Expand-Archive modelos_backup.zip -DestinationPath .
```

O simplemente entrenar desde cero en la nueva PC — el código es el mismo.

---

## Métricas clave en TensorBoard

| Métrica | Qué significa |
|---|---|
| `rollout/ep_rew_mean` | Recompensa media por episodio. Con la nueva escala, apunta a >300 |
| `train/explained_variance` | Qué tan bien el crítico predice el valor. >0.8 es bueno |
| `train/approx_kl` | Divergencia KL. <0.03 es estable |
| `train/clip_fraction` | Fracción de acciones recortadas. <0.2 es normal |
| `train/value_loss` | Error del crítico. Debe bajar con el tiempo |
| `time/fps` | Velocidad de entrenamiento (frames por segundo) |
