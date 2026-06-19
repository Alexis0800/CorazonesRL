# 📊 STATUS del Proyecto — Corazones RL

**Fecha:** 2026-06-19
**Último hito:** Consolidación v12 — SSOT de recompensas, dimensiones unificadas (220-dim estándar), value head estabilizado

---

## 🏗️ Arquitectura (refactorizada 2026-06-15)

```
src/
├── dominio/                   # Reglas del juego (sin deps RL)
│   ├── carta.py               # Value Object inmutable
│   ├── baraja.py              # Baraja francesa
│   ├── jugador.py             # Jugador (mano, puntuación)
│   └── motor.py               # Motor del juego (bazas, reglas)
├── entorno/                   # RL Environment
│   ├── dimensiones.py         # SSOT de dimensionalidad (DRY)
│   ├── observacion.py         # Observable builder 220-dim (SRP)
│   ├── recompensas.py         # RewardConfig v12 + Calculadora (SRP)
│   ├── single_agent.py        # CorazonesEnv (Gymnasium)
│   └── multi_agent.py         # CorazonesAEC (PettingZoo)
├── agentes/                   # Estrategias de juego
│   ├── heuristicos.py         # 3 bots (conservador, agresivo, evasivo)
│   └── politica_rl.py         # Adaptador MaskablePPO → Strategy
├── red.py                     # Red neuronal MLP [512,512,256]
├── torneo/                    # Evaluación y rating
│   ├── elo.py                 # Sistema Elo (least-squares convergente)
│   ├── evaluacion.py          # Win rate contra bots
│   └── normalizacion.py       # VecNormalize utilities
├── entrenamiento/             # Training pipeline
│   ├── config.py              # SSOT: paths + hiperparámetros
│   └── self_play.py           # Creación de entornos self-play
├── cli/                       # CLI entry points
│   ├── jugar.py               # Juego interactivo
│   └── evaluar.py             # Evaluación standalone
└── __init__.py                # Re-exports públicos

modelos/                       # Estructura estandarizada por versión
├── {version}/
│   ├── config.json            # Metadatos de la versión
│   ├── snapshots/             # Checkpoints de entrenamiento
│   ├── elite/                 # Mejores snapshots (torneo)
│   ├── vecnorm/               # Estadísticas VecNormalize
│   └── torneos/               # Resultados de torneos Elo

scripts/                       # Entry points
├── entrenar.py
├── evaluar.py
├── jugar.py
└── comprimir_modelos.ps1

tests/                         # Tests espejo de src/
├── dominio/
├── entorno/
├── agentes/
├── torneo/
└── cli/

train_auto_v6.py               # Pipeline de entrenamiento autónomo
train_self_play.py             # Pipeline Self-Play
```

**Modelo:** MaskablePPO (sb3-contrib) con 194-dim obs (v6+)
**Entorno:** CorazonesEnv → DummyVecEnv → VecNormalize
**Self-play:** Fictitious Self-Play con pool de snapshots históricos
**Elo:** Sistema least-squares convergente (sin order bias)

---

## ✅ LO COMPLETADO

### Fase 1 — Motor del Juego

- [x] Clases Carta, Baraja, Jugador
- [x] Bucle de mano completo (repartir, 13 bazas, conteo)
- [x] Shooting the Moon, corazones rotos, Dama de Picas
- [x] `obtener_jugadas_legales()` con 4 filtros en cascada
- [x] Test de estrés: 10,000 manos < 3 segundos

### Fase 2 — Entorno Gymnasium

- [x] `CorazonesEnv(gym.Env)` con action masking
- [x] Vector de observación 194 dimensiones (all_void en v6+)
- [x] Recompensa suma-cero con reward shaping
- [x] `env_checker` pass

### Fase 3 — Agente RL

- [x] MaskablePPO con red MLP [512,512,512]
- [x] 3 bots heurísticos
- [x] Fictitious Self-Play con VecNormalize
- [x] Pipeline autónomo: snapshots, evals, pruning, torneos Elo

### Fase 5 — all_void (4 features extra)

- [x] 190→194 dimensiones
- [x] Feature: "todos los rivales vacíos en palo X"

### Fase 6 — Elo Convergente

- [x] Eliminado order bias (least-squares directo)
- [x] 24/24 tests pasando
- [x] Modo `--elo-puro` (snapshots vs snapshots, no bots)

### Mejoras de infraestructura

- [x] `--output-dir` en `train_auto_v6.py` (entrena en cualquier directorio)
- [x] Cosine decay prob_bot 50%→20%
- [x] Golden checkpoints congelados (`modelos/v5_golden/` y `modelos/v7_golden/`)
- [x] **Refactorización 2026-06-15**: src/ aplanado (sin stubs ni corazones/), tests/ reorganizado en 5 subdirectorios, modelos/ estandarizado, docs/ curado, 10 stubs eliminados
- [x] Tests: 286/286 pasando

---

## 📈 ESTADO ACTUAL

### Rankings históricos (Elo Puro)

| # | Modelo | Elo | Generación |
|---|--------|-----|------------|
| 1 | **v7_14.9M** | **1723** | v7 (desde golden 10M) |
| 2 | v7_15.0M | 1717 | v7 |
| 3 | v7_14.5M | 1708 | v7 |
| 4 | v7_12.5M | 1607 | v7 |
| 5 | v6_10.0M | 1532 | v6 (ex-golden) |
| 6 | v5_17.65M | 1513 | v5 |
| 7 | v5_18.35M | 1499 | v5 |
| 8 | v5_18.4M | 1494 | v5 |
| 9 | v6_14.4M | 1493 | v6 degradado |

### Curva de aprendizaje v7 (torneos internos)

```
10.8M: 1489 → 11.8M: 1608 → 14.0M: 1642 → 14.9M: 1723
```

Mejora continua, sin signos de estancamiento.

### Golden Checkpoints

| Directorio | Modelo | Elo | Estado |
|---|---|---|---|
| `modelos/v5_golden/` | v5_best | ~1500 | ❄️ Congelado (baseline 190-dim) |
| `modelos/v7_golden/` | v7_14.9M | 1723 | ❄️ Congelado (baseline 194-dim) |

### Win rates contra bots (métrica complementaria)

| Modelo | WR | Top-2 | AvgPts |
|--------|-----|-------|--------|
| v7_14.9M | 84% | 93% | 35.7 |
| v7_15.0M | 84% | 96% | 33.8 |
| v6_10M | 73% | 96% | 42.5 |

⚠️ El WR contra bots es una métrica débil. El Elo puro es la métrica real.

---

## ⬜ PENDIENTE

### Corto plazo

- [ ] **A) Continuar v8**: Proseguir entrenamiento autónomo con torneos Elo
- [ ] **B) Nuevo v9**: Desde cero con cosine decay 50%→20%
- [ ] Evaluar si v8 se estanca o sigue mejorando

### Mediano plazo (mejora estratégica)

- [ ] Reward shaping intermedio (no solo terminal)
- [ ] Sistema de 3 modos: MINIMIZAR / POZO / ALIMENTAR
- [ ] Features de planificación multi-baza

### Largo plazo (cambio arquitectónico)

- [ ] HLR: Nivel estratégico + táctico (ver `docs/planes/historico/04_plan_mejora_estrategica.md`)
- [ ] MCTS offline para generar datasets expertos (ver `docs/planes/historico/06_arquitectura_multimodelo_mcts.md`)
- [ ] Modelo especializado de pase de cartas

---

## ❓ EN DUDA / DECISIONES

1. **¿Techo del cosine decay?** v7 pasó de 1532→1723 en 5M pasos. ¿Seguirá mejorando hasta 20M o se estancará?

2. **¿v9 desde cero con cosine decay?** Si v9 desde cero llega más lejos que v7 (que solo tuvo cosine decay post-10M), entonces el cosine decay es mejor aplicarlo desde el principio.

3. **¿HLR o MCTS?** Ambos enfoques (planes archivados) son complementarios, no excluyentes.

4. **¿Reward shaping intermedio?** Dar recompensa por baza (no solo terminal) podría acelerar el aprendizaje. Pero introduce riesgo de que el modelo aprenda a maximizar la recompensa intermedia en vez de ganar.

5. **¿VecNormalize entre generaciones?** v5, v6, v7 usan VecNormalize. ¿Son compatibles entre sí? Actualmente parece que sí (los torneos cross-gen funcionan), pero no está verificado formalmente.

---

## 🔧 COMANDOS RÁPIDOS

```bash
# Activar entorno
.venv\Scripts\Activate.ps1

# Entrenar
python train_auto_v6.py --total-steps 20000000 --output-dir modelos/v9
python train_auto_v6.py --resume MODELO.zip --total-steps 20000000 --output-dir modelos/v9

# Torneo Elo
python -m src.torneo.elo --directorio modelos/v8/elite --partidas 50 --elo-puro --incluir-bots

# Evaluar win rate
python scripts/evaluar.py --modelo MODELO.zip

# Jugar contra el modelo
python scripts/jugar.py --modelo modelos/v8/elite/snapshot_*.zip

# Tests
python -m pytest tests/ -q
```

---

## 🌿 PLAN GIT + ENTRENAMIENTO PARALELO

### Ramas propuestas

```
main          ← commit actual (stable, documentado)
  ├── exp/v7_cont   ← Continuar v7 desde golden_v7 → 20M
  └── exp/v8_scratch ← v8 desde cero con cosine decay
```

### No se requieren cambios de código

Ambos experimentos usan `train_auto_v6.py` sin modificar, solo cambian los argumentos.

### Comandos por máquina

**Máquina A (v7_cont → 20M):**

```bash
git checkout exp/v7_cont
mkdir modelos_historicos\v7_cont
python train_auto_v6.py --resume modelos_historicos/golden_v7/snapshot_0014900000 --total-steps 20000000 --output-dir modelos_historicos/v7_cont
```

**Máquina B (v8 desde cero → 20M):**

```bash
git checkout exp/v8_scratch
mkdir modelos_historicos\v8
python train_auto_v6.py --total-steps 20000000 --output-dir modelos_historicos/v8
```

---

## 📁 Estructura de archivos (al commit)

```
0_Plan_de_Desarrollo_y_Testing.md   ← Plan maestro (Fases 1-3, DoD)
1_Reglas_y_Motor_Corazones.md       ← Reglas del juego
2_Entorno_IA_Gymnasium.md           ← Diseño del entorno RL
3_Arquitectura_Agente_RL.md         ← Arquitectura PPO inicial
4_Plan_Mejora_Estrategica.md        ← Diagnóstico + propuesta HLR
5_Plan_Fase6_AllVoid.md             ← Features all_void
6_Arquitectura_MultiModelo_MCTS.md  ← Propuesta MCTS + multi-modelo
STATUS.md                           ← Este documento
elo.md                              ← Resultados de torneos Elo
observaciones_modelo.md             ← Errores observados en partidas

src/                                ← Código fuente
tests/                              ← Tests (24 Elo + engine + env)
modelos_historicos/
  golden/          ← v6_10M (1532 Elo) ❄️
  golden_v7/       ← v7_14.9M (1723 Elo) ❄️
  v5/              ← Snapshots v5 (1540-1840)
  v6/              ← Snapshots v6 (950-1440)
  v7/              ← Snapshots v7 (1010-1500)

torneos/                            ← Resultados de torneos (solo texto)

train_auto_v6.py                    ← Pipeline autónomo
train_self_play.py                  ← Configuración + constantes
evaluar_modelo.py                   ← Evaluación
jugar_contra_modelo.py              ← Partida interactiva
```

---

*Documento generado automáticamente. Actualizar tras cada hito.*
