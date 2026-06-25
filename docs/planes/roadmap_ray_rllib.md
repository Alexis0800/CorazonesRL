# Roadmap: Migración de SB3/sb3-contrib a Ray RLlib

> **Estado**: Borrador — Análisis de viabilidad y plan detallado
> **Fecha**: 2026-06-25
> **Objetivo**: Reemplazar MaskablePPO (sb3-contrib) con PPO recurrente + action masking nativo en Ray RLlib

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Análisis de Viabilidad](#2-análisis-de-viabilidad)
3. [Fase 1 — Setup e Instalación](#3-fase-1--setup-e-instalación)
4. [Fase 2 — Migración del Entorno](#4-fase-2--migración-del-entorno)
5. [Fase 3 — Configuración del Algoritmo](#5-fase-3--configuración-del-algoritmo)
6. [Fase 4 — Self-Play Multiagente](#6-fase-4--self-play-multiagente)
7. [Fase 5 — Pipeline de Entrenamiento](#7-fase-5--pipeline-de-entrenamiento)
8. [Fase 6 — Evaluación (Elo)](#8-fase-6--evaluación-elo)
9. [Fase 7 — VecNormalize y Snapshots](#9-fase-7--vecnormalize-y-snapshots)
10. [Fase 8 — Observación con Memoria Explícita](#10-fase-8--observación-con-memoria-explícita)
11. [Estimación de Esfuerzo](#11-estimación-de-esfuerzo)
12. [Riesgos y Preguntas Abiertas](#12-riesgos-y-preguntas-abiertas)
13. [Checklist de Migración](#13-checklist-de-migración)

---

## 1. Resumen Ejecutivo

Ray RLlib ofrece soporte **nativo y simultáneo** para los tres requerimientos que el proyecto necesita y que SB3/sb3-contrib no puede combinar sin modificaciones mayores:

| Requerimiento | SB3/sb3-contrib | Ray RLlib |
|---|---|---|
| Action Masking | ✅ `MaskablePPO` | ✅ `action_masking` en config |
| LSTM/GRU recurrente | ⚠️ Solo `RecurrentPPO` (sin masking) | ✅ `model.use_lstm: true` |
| Multiagente (self-play) | ❌ Manual (`self_play.py`) | ✅ `MultiAgentEnv` + policy mapping |
| **Los tres juntos** | ❌ **No soportado** | ✅ **Soportado nativamente** |

### Estimación total de esfuerzo

| Concepto | Tiempo estimado |
|---|---|
| Fase 1: Setup | 2-4 horas |
| Fase 2: Migración del entorno | 8-12 horas |
| Fase 3: Config del algoritmo | 4-6 horas |
| Fase 4: Self-play multiagente | 12-16 horas |
| Fase 5: Pipeline entrenamiento | 8-12 horas |
| Fase 6: Evaluación (Elo) | 6-8 horas |
| Fase 7: VecNormalize y snapshots | 4-6 horas |
| Fase 8: Observación con memoria | 6-8 horas |
| Testing y debugging | 16-24 horas |
| **Total** | **66-96 horas (2-3 semanas)** |

### ¿Vale la pena?

| Pro | Contra |
|-----|--------|
| LSTM + action masking nativos | Curva de aprendizaje empinada |
| Multiagente real (no simulado) | Debugging difícil (Ray actors) |
| Escalabilidad a múltiples GPUs/nodos | Overhead de infraestructura |
| Ecosistema maduro (documentación) | Migración de TODO el código |
| No más limitaciones de SB3 | 267 tests que reescribir |

---

## 2. Análisis de Viabilidad

### 2.1 Lo que se conserva (sin cambios)

Estos módulos son **independientes del framework** y se reutilizan tal cual:

```
src/dominio/
├── carta.py          ✅ Sin cambios
├── baraja.py         ✅ Sin cambios
├── jugador.py        ✅ Sin cambios
└── motor.py          ✅ Sin cambios (MotorCorazones)
```

### 2.2 Lo que requiere adaptación

| Módulo actual | Cambio necesario | Complejidad |
|---|---|---|
| `src/entorno/single_agent.py` | Reescribir como `MultiAgentEnv` | 🔴 Alta |
| `src/entorno/observacion.py` | Misma lógica, nueva interfaz | 🟡 Media |
| `src/entorno/recompensas.py` | Adaptar diccionario de rewards | 🟡 Media |
| `src/red.py` | Reemplazar con RLlib `model_config` | 🟢 Baja |
| `src/agentes/politica_rl.py` | Ya no se usa (RLlib lo maneja) | 🟢 Baja |
| `src/entrenamiento/self_play.py` | Reemplazar con policy mapping | 🔴 Alta |
| `src/entrenamiento/config.py` | Reemplazar con `PPOConfig` | 🟡 Media |
| `train.py` | Reescribir completamente | 🔴 Alta |
| `src/torneo/elo.py` | Adaptar (carga de modelos) | 🟡 Media |

### 2.3 Lo que se elimina

```
src/red.py                          ← RLlib tiene su propio model config
src/agentes/politica_rl.py          ← RLlib maneja la política
src/entrenamiento/self_play.py      ← Policy mapping de RLlib
src/entorno/single_agent.py         ← Reemplazado por MultiAgentEnv
```

---

## 3. Fase 1 — Setup e Instalación

### 3.1 Instalación de Ray RLlib

```powershell
# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Instalar Ray con soporte RLlib
pip install "ray[rllib]" tensorflow  # o pytorch (ya instalado)

# Verificar instalación
python -c "import ray; from ray.rllib.algorithms.ppo import PPO; print('OK')"
```

### 3.2 Dependencias nuevas en `requirements.txt`

```
ray>=2.40.0
gymnasium>=1.0.0
lz4                    # compresión de checkpoints
rich                   # logging bonito (opcional)
```

### 3.3 Estructura de archivos nueva

```
src/
├── dominio/           ← Sin cambios
│   ├── carta.py
│   ├── baraja.py
│   ├── jugador.py
│   └── motor.py
├── entorno/
│   ├── corazones_rllib.py     ← NUEVO: MultiAgentEnv para RLlib
│   ├── observacion.py         ← Modificado: nueva interfaz
│   ├── recompensas.py         ← Modificado: dict rewards
│   └── dimensiones.py         ← Modificado: DIM_V12
├── rllib/                     ← NUEVO: todo lo de RLlib
│   ├── __init__.py
│   ├── config.py              ← PPOConfig + model_config
│   ├── policy_mapping.py      ← Self-play policy mapping
│   ├── callbacks.py           ← Custom callbacks (métricas, snapshots)
│   └── utils.py               ← VecNormalize, carga/descarga
├── torneo/
│   └── elo.py                 ← Modificado: carga modelos RLlib
└── train_rllib.py             ← NUEVO: script principal de entrenamiento
```

---

## 4. Fase 2 — Migración del Entorno

### 4.1 `MultiAgentEnv` — Concepto clave

En SB3, el entorno es **single-agent**: un solo `CorazonesEnv` con 1 agente + 3 bots.

En RLlib, el entorno es **multi-agent**: los 4 jugadores son agentes RLlib. RLlib decide qué política usa cada uno mediante `policy_mapping_fn`.

```python
# src/entorno/corazones_rllib.py

from ray.rllib.env.multi_agent_env import MultiAgentEnv

class CorazonesMultiAgentEnv(MultiAgentEnv):
    """Entorno multiagente para Hearts compatible con Ray RLlib.
    
    Los 4 jugadores son agentes RLlib. Cada step avanza el juego
    para un jugador. RLlib invoca la política correspondiente según
    policy_mapping_fn.
    """
    
    def __init__(self, env_config: dict):
        super().__init__()
        self.motor = MotorCorazones()
        self._obs_builder = ObservacionBuilder(dim=env_config.get("obs_dim", 224))
        
        # Observation space: dict[agent_id → Box]
        self.observation_space = spaces.Dict({
            f"player_{i}": spaces.Box(0, 1, (obs_dim,), np.float32)
            for i in range(4)
        })
        
        # Action space: dict[agent_id → Discrete]
        self.action_space = spaces.Dict({
            f"player_{i}": spaces.Discrete(52)
            for i in range(4)
        })
        
        # Action masking: RLlib lo soporta nativamente
        # Se retorna en info["action_mask"] por agente
    
    def reset(self, *, seed=None, options=None):
        # Repartir, iniciar mano
        self.motor.repartir()
        obs = self._build_obs_all()
        return obs, {}
    
    def step(self, action_dict: dict):
        """action_dict = {"player_0": card_id, "player_1": ..., ...}
        
        RLlib invoca step() UNA vez con las acciones de TODOS los agentes
        que actuaron en este 'rollout step'. En Hearts, esto es 1 acción
        por step (solo juega 1 jugador a la vez).
        """
        # Determinar quién es el jugador actual
        actual = self.motor.obtener_jugador_actual()
        agent_id = f"player_{actual}"
        
        carta = Carta._TODAS[action_dict[agent_id]]
        self._ejecutar_jugada(actual, carta)
        
        # Resolver baza si está completa
        if len(self.motor.mesa) == 4:
            self._resolver_baza()
        
        obs = self._build_obs_all()
        rewards = self._compute_rewards()
        terminateds = {"__all__": self._game_over()}
        truncateds = {"__all__": False}
        infos = self._build_infos()  # incluye action_mask
        
        return obs, rewards, terminateds, truncateds, infos
```

### 4.2 Action Masking en RLlib

RLlib soporta action masking a través del campo `action_mask` en las observaciones:

```python
def _build_infos(self) -> dict:
    """Construye infos con action_mask para cada agente."""
    infos = {}
    actual = self.motor.obtener_jugador_actual()
    for i in range(4):
        agent_id = f"player_{i}"
        mask = np.zeros(52, dtype=np.bool_)
        if i == actual:
            legales = self.motor.obtener_jugadas_legales(i)
            for c in legales:
                mask[c.id] = True
        infos[agent_id] = {"action_mask": mask}
    return infos
```

Y en la configuración del modelo:

```python
config = (
    PPOConfig()
    .environment(CorazonesMultiAgentEnv)
    .training(
        model={
            "custom_model": "action_mask_model",  # o built-in
        }
    )
)
```

RLlib tiene un `TorchActionMaskModel` built-in que maneja el enmascaramiento automáticamente cuando recibe `action_mask` en las observaciones.

### 4.3 Preguntas que resolver en esta fase

1. **¿`step()` recibe 1 acción o 4?** En RLlib, `step()` recibe acciones para todos los agentes que actuaron en el rollout. En Hearts solo juega 1 jugador por step, así que solo 1 acción. Pero si usamos `policy_mapping_fn` para mapear todos los agentes, RLlib espera acciones para todos.

   **Solución probable**: Usar `AgentID.agent_id_to_int()` y retornar acción dummy (0) para agentes que no están jugando, ignorando esas acciones en el entorno.

2. **¿Quién controla el orden de juego?** En Hearts, el orden es fijo (0→1→2→3 dentro de cada baza, luego el ganador empieza la siguiente). Con `MultiAgentEnv`, RLlib asume que todos los agentes actúan simultáneamente en cada step.

   **Solución**: Hacer que `step()` solo procese la acción del jugador actual e ignore las demás. Esto requiere modificar el ciclo de juego para que RLlib llame a `step()` 4 veces por baza (una por jugador) en lugar de 1 vez con 4 acciones.

### 4.4 Alternativa: `Env` single-agent con múltiples políticas

RLlib también soporta entornos single-agent con múltiples políticas mediante `policy_mapping_fn`. Esto podría ser más simple:

```python
class CorazonesEnvRLlib(gym.Env):
    """Single-agent Env con múltiples políticas mapeadas por RLlib."""
    
    def step(self, action):
        # RLlib llama a step() con la acción de LA política activa
        # La política activa se determina por policy_mapping_fn
        ...
```

**Recomendación**: Explorar ambas opciones. La single-agent con multi-policy es probablemente más simple para Hearts.

---

## 5. Fase 3 — Configuración del Algoritmo

### 5.1 `PPOConfig` base

```python
# src/rllib/config.py

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

def build_config(obs_dim: int = 250) -> PPOConfig:
    """Construye la configuración de PPO para Hearts.
    
    Características clave:
    - LSTM/GRU recurrente (use_lstm=True)
    - Action masking nativo
    - Múltiples políticas para self-play
    """
    config = (
        PPOConfig()
        .environment(
            env="src.entorno.corazones_rllib.CorazonesEnvRLlib",
            env_config={"obs_dim": obs_dim},
        )
        .framework("torch")
        .training(
            # ─── Modelo recurrente ───
            model={
                "use_lstm": True,              # ← LSTM nativo!
                "lstm_cell_size": 128,          # Tamaño del hidden state
                "max_seq_len": 52,              # Máx cartas por mano (13 bazas × 4)
                "fcnet_hiddens": [512, 512, 256],
                "fcnet_activation": "relu",
                # Action masking model
                "custom_model": None,           # Usar built-in o custom
                "vf_share_layers": False,
            },
            # ─── Hiperparámetros PPO ───
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            train_batch_size=8192,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            # ─── GPU ───
            num_rollout_workers=4,              # 4 workers paralelos
            num_envs_per_worker=1,
        )
        # ─── Multiagente ───
        .multi_agent(
            policies={
                "main": PolicySpec(),           # Política principal (entrenando)
                "snapshot_1": PolicySpec(),     # Snapshots históricos
                "snapshot_2": PolicySpec(),
                "bot_experto": PolicySpec(),    # Bot heurístico
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main"],         # Solo entrenar la principal
        )
        # ─── Recursos ───
        .resources(
            num_gpus=0,                         # Cambiar a 1 si hay GPU
            num_cpus_per_worker=1,
        )
    )
    return config
```

### 5.2 Modelo custom con action masking

Si el modelo built-in no es suficiente:

```python
# src/rllib/model.py

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.utils.annotations import override
import torch.nn as nn

class CorazonesTorchModel(TorchModelV2, nn.Module):
    """Modelo custom con LSTM + action masking para Hearts.
    
    Arquitectura:
        obs → Encoder MLP → LSTM → Policy/Value heads
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        input_dim = obs_space.shape[0]
        lstm_cell_size = model_config.get("lstm_cell_size", 128)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(512, lstm_cell_size, batch_first=True)
        
        # Policy & Value heads
        self.policy_head = nn.Linear(lstm_cell_size, num_outputs)
        self.value_head = nn.Linear(lstm_cell_size, 1)
        
        # Inicialización
        self.policy_head.weight.data = normc_initializer(0.01)
        self.value_head.weight.data = normc_initializer(0.01)
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        x = self.encoder(obs)
        
        # LSTM espera (batch, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        x, new_state = self.lstm(x, [s.unsqueeze(0) for s in state])
        x = x.squeeze(1)  # (batch, features)
        
        logits = self.policy_head(x)
        self._value_out = self.value_head(x)
        
        return logits, [s.squeeze(0) for s in new_state]
    
    @override(TorchModelV2)
    def value_function(self):
        return self._value_out.flatten()
```

### 5.3 Preguntas que resolver

1. **¿Qué `max_seq_len` usar?** En Hearts, una mano tiene ~13 bazas. El agente juega ~13 veces por mano. `max_seq_len=52` cubre toda la mano con margen. Pero esto puede ser ineficiente para BPTT. Probar con `max_seq_len=13` (una baza) o `26` (dos bazas).

2. **¿Training batch size?** Con 4 workers × 1 env × ~52 steps por episodio = ~208 steps por rollout. `train_batch_size=8192` significa ~40 episodios por batch de entrenamiento. Ajustar según memoria GPU.

---

## 6. Fase 4 — Self-Play Multiagente

### 6.1 Estrategia de Self-Play en RLlib

En SB3, el self-play es simulado: el `CorazonesEnv` recibe políticas de oponentes como callables.

En RLlib, el self-play es **real**: cada jugador es un agente RLlib con su propia política. RLlib maneja el ciclo multiagente nativamente.

### 6.2 `policy_mapping_fn`

```python
# src/rllib/policy_mapping.py

import random
from typing import List, Optional

def policy_mapping_fn(agent_id: str, episode, worker, **kwargs):
    """Asigna políticas a agentes según la fase de entrenamiento.
    
    Estrategia (5 fases, igual que v5 actual):
      - Fase 0 (0-5%): 3 bots heurísticos
      - Fase 1 (5-15%): 1 BotExperto + 2 bots
      - Fase 2 (15-35%): 1 main + 1 snapshot + 1 BotExperto
      - Fase 3 (35-60%): 2 main clones + 1 snapshot (DESTETE)
      - Fase 4 (60-100%): 3 main clones (PURO, solo si LSTM lo soporta)
    
    El agente "main" (player_0) siempre usa la política en entrenamiento.
    Los otros 3 usan bots, snapshots o clones según la fase.
    """
    global_timestep = episode.total_timestep if episode else 0
    total_steps = worker.config.get("total_steps", 20_000_000) if worker else 20_000_000
    progress = global_timestep / total_steps
    
    if agent_id == "player_0":
        return "main"  # Siempre entrenando
    
    # Determinar fase
    if progress < 0.05:
        return random.choice(["bot_random", "bot_conservador", "bot_agresivo"])
    elif progress < 0.15:
        return "bot_experto" if random.random() < 0.67 else "bot_conservador"
    elif progress < 0.35:
        pool = ["bot_experto", "snapshot_pool"]
        return random.choice(pool)
    elif progress < 0.60:
        pool = ["main_clone", "main_clone", "snapshot_pool"]
        return random.choice(pool)
    else:
        return "main_clone"  # Self-play puro
```

### 6.3 Políticas de bots heurísticos

Los bots heurísticos (conservador, agresivo, evasivo, experto) no se entrenan. Se implementan como políticas **custom** que ignoran el modelo y usan la heurística:

```python
# src/rllib/bot_policies.py

from ray.rllib.policy.policy import Policy

class BotPolicy(Policy):
    """Política que usa un bot heurístico en lugar de red neuronal."""
    
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.bot_fn = config.get("bot_fn")  # Callable (motor, idx, legales) → Carta
    
    def compute_actions(self, obs_batch, state_batches, **kwargs):
        # Los bots no usan el modelo, usan la heurística directamente
        # Nota: esto requiere acceso al estado del motor, que está en el env
        ...
```

**⚠️ Problema**: Las políticas en RLlib no tienen acceso directo al estado del entorno. Los bots heurísticos necesitan `MotorCorazones` para decidir. Esto requiere pasar el estado del motor en la observación o en `info`.

**Alternativa**: Mantener los bots como lógica dentro del entorno (como en SB3), y solo usar RLlib para el agente en entrenamiento + snapshots. Esto simplifica enormemente.

### 6.4 Preguntas que resolver

1. **¿Bots dentro del env o como políticas RLlib?** Dentro del env es más simple y mantiene compatibilidad con `src/agentes/heuristicos.py`. Fuera (como políticas) es más "puro" pero complejo.

2. **¿Snapshots como checkpoints RLlib?** RLlib guarda checkpoints completos (modelo + optimizer + config). Cargar un snapshot como oponente implica restaurar un checkpoint en una política separada. Esto es más pesado que en SB3 (donde solo se carga el `.zip`).

---

## 7. Fase 5 — Pipeline de Entrenamiento

### 7.1 Script principal

```python
# train_rllib.py

import ray
from ray.rllib.algorithms.ppo import PPO
from src.rllib.config import build_config
from src.rllib.callbacks import HeartsCallback

def main():
    # Inicializar Ray
    ray.init()
    
    # Construir configuración
    config = build_config(obs_dim=250)
    
    # Añadir callbacks
    config.callbacks(HeartsCallback)
    
    # Crear algoritmo
    algo = PPO(config=config)
    
    # Loop de entrenamiento
    total_steps = 20_000_000
    checkpoint_interval = 200_000
    eval_interval = 1_000_000
    
    for i in range(total_steps // checkpoint_interval):
        result = algo.train()
        
        # Guardar checkpoint
        if (i + 1) % (checkpoint_interval // config.train_batch_size) == 0:
            checkpoint = algo.save(checkpoint_dir=f"models/rllib_v1")
            print(f"Checkpoint: {checkpoint}")
        
        # Evaluar
        if result["timesteps_total"] % eval_interval == 0:
            algo.evaluate()
    
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
```

### 7.2 Custom Callbacks

```python
# src/rllib/callbacks.py

from ray.rllib.algorithms.callbacks import DefaultCallbacks

class HeartsCallback(DefaultCallbacks):
    """Callbacks para métricas de Hearts durante entrenamiento."""
    
    def on_episode_end(self, *, episode, **kwargs):
        """Al final de cada episodio, registrar métricas."""
        # Puntos del agente principal
        agent_points = episode.last_info_for("player_0").get("puntos_agente", 0)
        episode.custom_metrics["puntos_agente"] = agent_points
        
        # Ganador de la partida
        episode.custom_metrics["gano"] = 1 if agent_points == min(
            episode.last_info_for(f"player_{i}").get("puntos", 100)
            for i in range(4)
        ) else 0
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Logging de métricas de entrenamiento."""
        # Similar a los diagnostics de train.py actual
        pass
```

### 7.3 Preguntas que resolver

1. **¿RLlib maneja el LR schedule?** Sí, mediante `lr_schedule` en la config. Se puede pasar una función `[[0, 1e-4], [10_000_000, 5e-5], [20_000_000, 1e-6]]`.

2. **¿Cómo se maneja el early stopping?** RLlib no tiene early stopping built-in. Se implementa en el callback `on_train_result`.

---

## 8. Fase 6 — Evaluación (Elo)

### 8.1 Torneo Elo con checkpoints RLlib

```python
# src/torneo/elo_rllib.py

import ray
from ray.rllib.algorithms.ppo import PPO

def cargar_modelo_rllib(checkpoint_path: str):
    """Carga un modelo RLlib desde checkpoint.
    
    A diferencia de SB3 (modelo.zip + vecnorm.pkl),
    RLlib guarda checkpoints completos con ray.air.checkpoint.
    """
    algo = PPO.from_checkpoint(checkpoint_path)
    return algo.get_policy("main")

def torneo_elo_rllib(directorio: str, partidas: int = 50):
    """Torneo Elo entre checkpoints RLlib."""
    # Similar a elo.py actual pero cargando checkpoints RLlib
    ...
```

### 8.2 VecNormalize en RLlib

RLlib tiene `Normalization` wrapper para observaciones y `RewardNormalization` para rewards:

```python
config.training(
    model={...},
).rl_module(
    model_config={
        "observation_normalization": "running_mean_std",
    }
)
```

Pero para cargar snapshots históricos con sus estadísticas de normalización, necesitas guardar las estadísticas junto con el checkpoint.

---

## 9. Fase 7 — VecNormalize y Snapshots

### 9.1 Estrategia de Snapshots

En SB3, los snapshots son archivos `.zip` + `_vecnorm.pkl`. En RLlib:

```python
# Guardar snapshot con estadísticas de normalización
def guardar_snapshot(algo, paso: int, output_dir: str):
    """Guarda un snapshot ligero (solo pesos + norm stats)."""
    checkpoint_dir = algo.save(checkpoint_dir=f"{output_dir}/snapshots")
    # Las estadísticas de normalización están en el checkpoint
    
    # También guardar solo los pesos para carga rápida
    policy = algo.get_policy("main")
    torch.save(policy.get_weights(), f"{output_dir}/snapshots/snapshot_{paso}.pth")
```

### 9.2 Pool de oponentes (self-play)

```python
# src/rllib/opponent_pool.py

class OpponentPool:
    """Pool de snapshots históricos para self-play.
    
    Equivalente a src/entrenamiento/self_play.py.
    """
    
    def __init__(self, snapshot_dir: str, max_snapshots: int = 50):
        self.snapshot_dir = snapshot_dir
        self.max_snapshots = max_snapshots
        self.snapshots: List[str] = []
    
    def add_snapshot(self, path: str):
        self.snapshots.append(path)
        if len(self.snapshots) > self.max_snapshots:
            # Eliminar el más antiguo (política de recency)
            self.snapshots.pop(0)
    
    def sample_opponent(self) -> str:
        """Samplea un snapshot del pool."""
        return random.choice(self.snapshots) if self.snapshots else "bot_experto"
```

---

## 10. Fase 8 — Observación con Memoria Explícita

### 10.1 Vector de observación v12 (~250 dims)

```
[0:52]     Mano del agente (one-hot)
[52:104]   Mesa actual (one-hot)
[104:168]  Historial de últimas 4 bazas (4 × 16 dims)
           Para cada baza (ordenadas de más reciente a más antigua):
             [0:4]   Ganador (one-hot relativo al agente)
             [4:8]   Palo de salida (one-hot)
             [8]     Puntos en la baza / 26
             [9:13]  Cartas jugadas: card_id/52 por posición
             [13:17] Quién jugó cada carta: idx_relativo/3
[168:172]  Tracking baza actual: quién ya jugó (4 flags)
[172:188]  Vacíos conocidos (4 × 4 palos)
[188:192]  Puntajes históricos / 100
[192:196]  Puntos mano actual / 26
[196]      Corazones rotos
[197]      Posición en baza actual / 3
[198:203]  Q♠ tracker (one-hot, 5 estados)
[203]      pozo_viable
[204]      debo_arriesgar
[205]      puedo_alimentar
[206:210]  all_void_X
[210]      Número de baza / 13
[211]      Jugadores cerca de 100 / 3
[212]      Q♠ capturada
[213]      Soy líder
[214]      Mano terminal posible
[215:219]  Cartas restantes por palo / 13
[219:223]  Cartas altas restantes por palo / 4
[223:227]  Prob Q♠ por jugador
[227:231]  Corazones capturados / 13
[231:235]  Alerta pozo
[235]      Palo de salida actual
```

**Total**: ~235-240 dims. Con margen para futuras expansiones: 250 dims.

### 10.2 Tracking de historial en el entorno

```python
class CorazonesEnvRLlib(gym.Env):
    def __init__(self, env_config):
        ...
        self._historial_bazas: list[dict] = []  # Máx 4 entradas
    
    def _resolver_baza(self):
        """Registra la baza resuelta en el historial."""
        ganador = self.motor.resolver_baza()
        
        # Registrar en historial
        self._historial_bazas.append({
            "ganador": (ganador - self.agente_idx) % 4,
            "palo_salida": self.motor.palo_de_salida or 0,
            "puntos": sum(c.puntos for _, c in mesa_snapshot),
            "cartas": [(c.id, (jug - self.agente_idx) % 4) 
                       for jug, c in mesa_snapshot],
        })
        
        # Buffer circular
        if len(self._historial_bazas) > 4:
            self._historial_bazas.pop(0)
```

### 10.3 Interacción LSTM + observación

Con LSTM, el historial explícito de 4 bazas en la observación es **redundante pero beneficioso**: la LSTM comprime la historia completa de la mano en su hidden state, y la observación explícita le da acceso rápido a las bazas recientes sin depender solo del hidden state. Esto acelera el aprendizaje.

---

## 11. Estimación de Esfuerzo

### Por fase (días hábiles de 8h)

| Fase | Descripción | Días | Riesgo |
|------|-------------|------|--------|
| 1 | Setup e instalación | 0.5 | 🟢 Bajo |
| 2 | Migración del entorno | 1.5-2 | 🔴 Alto (API compleja) |
| 3 | Config del algoritmo | 0.5-1 | 🟡 Medio |
| 4 | Self-play multiagente | 1.5-2 | 🔴 Alto |
| 5 | Pipeline entrenamiento | 1-1.5 | 🟡 Medio |
| 6 | Evaluación (Elo) | 1 | 🟡 Medio |
| 7 | VecNormalize y snapshots | 0.5-1 | 🟡 Medio |
| 8 | Observación con memoria | 1 | 🟢 Bajo |
| — | Testing y debugging | 2-3 | 🔴 Alto |
| **Total** | | **10-14 días** | |

### Curva de aprendizaje estimada

```
Día 1-2:   Leer docs de RLlib, entender MultiAgentEnv, action masking
Día 3-5:   Implementar CorazonesEnvRLlib (prueba y error)
Día 6-7:   Configurar PPO + LSTM, verificar que entrena
Día 8-10:  Self-play multiagente (lo más complejo)
Día 11-12: Pipeline completo + evaluación
Día 13-14: Debugging, tests, ajustes finos
```

---

## 12. Riesgos y Preguntas Abiertas

### Riesgos críticos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| RLlib no maneja bien el orden de turnos de Hearts | Alta | Bloqueante | Prototipo rápido en Fase 2 para validar |
| Self-play con múltiples políticas es inestable | Media | Alto | Empezar con bots heurísticos, añadir snapshots gradualmente |
| Checkpoints RLlib son muy pesados (>1GB) | Alta | Medio | Guardar solo pesos de política, no optimizer |
| El modelo no aprende mejor que SB3 | Media | Alto | Comparar A/B con misma observación, solo difiriendo LSTM |
| Curva de aprendizaje de RLlib es más lenta de lo estimado | Alta | Alto | Tener SB3 como fallback, no borrar código actual |

### Preguntas abiertas (requieren prototipo)

1. **¿RLlib `step()` con 1 acción por vez funciona para Hearts?** El diseño de RLlib asume que todos los agentes actúan simultáneamente. Hearts es secuencial. Esto puede requerir un wrapper especial.

2. **¿Cómo se pasan `action_mask` en observaciones dict?** RLlib espera `action_mask` como key en el dict de observación cuando se usa `spaces.Dict`. Pero con `spaces.Box`, se puede incluir en `info`.

3. **¿Cómo se manejan los bots heurísticos como políticas RLlib?** Si los bots son políticas RLlib, necesitan acceso al `MotorCorazones` interno. La alternativa es que el entorno ejecute los bots y solo exponga al agente RL.

4. **¿Vale la pena `MultiAgentEnv` o es mejor `Env` + multi-policy?** Con `Env` normal, RLlib alterna políticas en cada step. Esto podría ser más natural para Hearts.

5. **¿Los checkpoints de RLlib son compatibles entre versiones?** Históricamente, RLlib ha tenido breaking changes en checkpoints. Usar una versión fija en `requirements.txt`.

---

## 13. Checklist de Migración

### Pre-migración

- [ ] Leer documentación de RLlib: `MultiAgentEnv`, `PolicySpec`, `policy_mapping_fn`
- [ ] Leer documentación de RLlib: action masking, LSTM models
- [ ] Hacer un prototipo mínimo: `gym.Env` → RLlib PPO entrenando en 5 minutos
- [ ] Hacer un prototipo con LSTM: verificar que `use_lstm=True` funciona
- [ ] Hacer un prototipo con action masking: verificar que `action_mask` funciona
- [ ] Decidir entre `MultiAgentEnv` vs `Env` + multi-policy
- [ ] Congelar `requirements.txt` con versiones exactas

### Fase 1: Setup

- [ ] Instalar `ray[rllib]` en `.venv`
- [ ] Verificar `import ray; ray.init()` funciona
- [ ] Verificar `from ray.rllib.algorithms.ppo import PPO` funciona
- [ ] Crear estructura `src/rllib/`
- [ ] Actualizar `requirements.txt`

### Fase 2: Entorno

- [ ] Implementar `CorazonesEnvRLlib` (gym.Env)
- [ ] Verificar `check_env(env)` pasa
- [ ] Implementar action masking en observaciones/info
- [ ] Implementar tracking de historial de bazas
- [ ] Test: 1 episodio con acciones aleatorias funciona

### Fase 3: Algoritmo

- [ ] Crear `build_config()` con PPO + LSTM
- [ ] Verificar que PPO entrena 1000 steps sin errores
- [ ] Implementar modelo custom (`CorazonesTorchModel`) si es necesario
- [ ] Test: pérdida decrece en 10K steps

### Fase 4: Self-play

- [ ] Implementar `policy_mapping_fn`
- [ ] Implementar bots heurísticos (como políticas o en el env)
- [ ] Implementar `OpponentPool` para snapshots
- [ ] Test: entrenamiento con 1 main + 3 bots funciona

### Fase 5: Pipeline

- [ ] Crear `train_rllib.py`
- [ ] Implementar `HeartsCallback`
- [ ] Implementar guardado periódico de checkpoints
- [ ] Implementar LR schedule
- [ ] Test: entrenamiento overnight sin crashes

### Fase 6: Evaluación

- [ ] Adaptar `elo.py` para cargar checkpoints RLlib
- [ ] Implementar `evaluar_contra_bots()` con política RLlib
- [ ] Test: torneo Elo con 2 checkpoints funciona

### Fase 7: Snapshots

- [ ] Implementar `guardar_snapshot()` (pesos + norm stats)
- [ ] Implementar `cargar_snapshot()` como política oponente
- [ ] Implementar rotación de snapshots en `OpponentPool`
- [ ] Test: cargar snapshot de hace 1M steps como oponente

### Fase 8: Observación v12

- [ ] Diseñar vector de 250 dims con historial de bazas
- [ ] Implementar en `ObservacionBuilder`
- [ ] Actualizar `dimensiones.py` con `DIM_V12`
- [ ] Test: `construir()` produce vector de shape (250,)

### Testing

- [ ] Tests unitarios para `CorazonesEnvRLlib`
- [ ] Tests de integración: entrenamiento + evaluación
- [ ] Tests de regresión: Elo comparable a v5 actual
- [ ] Stress test: 1M steps sin memory leaks ni crashes

---

## Referencias

- [RLlib MultiAgentEnv docs](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical)
- [RLlib Action Masking docs](https://docs.ray.io/en/latest/rllib/rllib-models.html#action-masking)
- [RLlib LSTM docs](https://docs.ray.io/en/latest/rllib/rllib-models.html#recurrent-models)
- [RLlib PPO config](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)
- [RLlib Self-Play / League Training](https://docs.ray.io/en/latest/rllib/rllib-advanced.html#self-play-and-league-training)
