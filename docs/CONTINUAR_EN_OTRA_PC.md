# Continuar Entrenamiento en Otra PC — CorazonesRL v8

> **Mejor modelo:** v7_golden — Elo 1723
> **Tests:** 286/286 pasando
> **Fecha:** 2026-06-15

---

## 1. Comprimir modelos para transferir

Solo necesitás los **últimos 10 snapshots de la versión activa** (~50MB) + VecNormalize (~5KB).

### Script automático

```powershell
.\scripts\comprimir_modelos.ps1
```

Esto crea `corazones_modelos.zip` con:
- Últimos 10 snapshots de la versión activa
- VecNormalize completo
- eval_log.jsonl

### Manual

```powershell
$version = 'v8'   # Cambiar según versión activa
$destino = "$env:USERPROFILE\Desktop\corazones_modelos.zip"

$snaps = Get-ChildItem modelos\$version\elite\snapshot_*.zip | Sort-Object Name | Select-Object -Last 10
$vecnorm = Get-ChildItem modelos\$version\vecnorm\*.pkl

Compress-Archive -Path (
    $snaps.FullName +
    $vecnorm.FullName
) -DestinationPath $destino -Force

Write-Host "$destino listo"
```

### ¿Qué modelos NO son necesarios?

| Carpeta | ¿Transferir? | Motivo |
|---|---|---|
| `models/v5_golden/` | ❌ No | Baseline congelado, solo referencia |
| `models/v6/` | ❌ No | Archivada |
| `models/v7/` | ❌ No | Archivada |
| `models/{version}/snapshots/` | ❌ No | Solo necesitás elite/ (mejores) |
| `models/{version}/vecnorm/` | ✅ Sí | Necesario para normalizar |
| `logs/` | ❌ No | Se regeneran |
| `torneos/` | ❌ No | Histórico, no esencial |

---

## 2. Configurar en la nueva PC

### 2.1 Requisitos

- Python 3.12 o 3.13
- Git
- 8+ GB RAM
- CPU multinúcleo o GPU

### 2.2 Clonar e instalar

```powershell
git clone <URL_DEL_REPO> corazones-neuralnetwork
cd corazones-neuralnetwork

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch stable-baselines3 sb3-contrib pettingzoo gymnasium numpy pytest tensorboard tqdm
```

### 2.3 Extraer modelos

```powershell
# Copiar corazones_modelos.zip a la raíz del proyecto
Expand-Archive corazones_modelos.zip -DestinationPath . -Force
```

### 2.4 Verificar

```powershell
python -m pytest tests/ -q
# Deben pasar 286 tests
```

---

## 3. Continuar entrenamiento

```powershell
# Reanudar desde el último snapshot elite:
python train_auto_v6.py --resume models/v8/elite/snapshot_0017900000.zip --total-steps 25000000 --output-dir models/v9

# Desde cero (nueva versión):
python train_auto_v6.py --total-steps 20000000 --output-dir models/v9

# Con GPU Intel Arc:
python train_auto_v6.py --total-steps 20000000 --device dml --output-dir models/v9
```

---

## 4. Empezar desde cero (sin modelos transferidos)

Si preferís entrenar desde cero sin transferir modelos:

```powershell
python train_auto_v6.py --total-steps 20000000 --output-dir models/v9
```

Esto crea un modelo nuevo con pesos aleatorios y entrena con Fictitious Self-Play (decaimiento coseno 50%→20% bots).

---

## 5. Monitorear con TensorBoard

```powershell
tensorboard --logdir logs/
# Abrir http://localhost:6006 en el navegador
```

Métricas clave a vigilar:
- `train/explained_variance` → Debe mantenerse > 0.7
- `train/entropy_loss` → Negativo = explorando (bien)
- `train/value_loss` → < 1.0 y estable
- `rollout/ep_rew_mean` → Subiendo

---

## 6. Evaluar modelo

```powershell
# Win rate contra bots
python scripts/evaluar.py --modelo models/v8/elite/snapshot_*.zip --partidas 500

# Torneo Elo entre snapshots
python -m src.torneo.elo --directorio models/v8/elite --partidas 50 --elo-puro --incluir-bots
```
