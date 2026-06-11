# Continuar Entrenamiento en Otra PC — CorazonesRL v2

> **Último modelo entrenado:** `snapshot_0003500000` — 84.5% win rate, 97% top-2  
> **Tests:** 116/116 pasando  
> **Fecha:** 2026-06-11

---

## 📦 1. Preparar el ZIP para transferir

### En esta PC (origen) — ejecutar en PowerShell

```powershell
# Ir a la raíz del proyecto
cd C:\Programming\Python\CorazonesRL

# Crear ZIP con los modelos históricos (v1 backup + v2 actual)
Compress-Archive -Path `
    "modelos_historicos/v1_backup",
    "modelos_historicos/v2",
    "modelos_historicos/snapshot_0001500000.zip",
    "vecnormalize" `
    -DestinationPath "..\CorazonesRL_modelos.zip" -Force

Write-Host "ZIP creado en C:\Programming\Python\CorazonesRL_modelos.zip"
```

También puedes hacer un ZIP del proyecto completo (sin `.venv`):

```powershell
cd C:\Programming\Python

# Excluir .venv (se recrea en la otra PC) y __pycache__
Compress-Archive -Path "CorazonesRL\*" `
    -DestinationPath "CorazonesRL_completo.zip" -Force

Write-Host "ZIP completo creado"
```

> **Alternativa:** Si usas Git, simplemente haz `git push` y en la otra PC haces `git clone` o `git pull`. Luego transfieres solo `CorazonesRL_modelos.zip`.

---

## 💻 2. Configurar en la nueva PC

### 2.1 Requisitos previos

- **Python 3.11+** instalado (descargar de <https://python.org>)
- ⚠️ **Desactivar los App Execution Aliases** de Python en Windows:
  - Settings → Apps → Advanced app settings → App execution aliases
  - Desactivar `python.exe` y `python3.exe`
- **Git** instalado (<https://git-scm.com>)

### 2.2 Clonar o copiar el proyecto

```powershell
# Opción A: Si usaste Git
git clone <URL_DEL_REPO> CorazonesRL
cd CorazonesRL

# Opción B: Si transferiste el ZIP completo
Expand-Archive CorazonesRL_completo.zip -DestinationPath .
cd CorazonesRL
```

### 2.3 Extraer los modelos

```powershell
# Si transferiste solo los modelos en ZIP aparte:
Expand-Archive CorazonesRL_modelos.zip -DestinationPath . -Force
```

### 2.4 Crear entorno virtual e instalar dependencias

```powershell
# Crear venv
python -m venv .venv

# Activar (si hay error de políticas, ver sección 2.5)
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install gymnasium numpy torch stable-baselines3 sb3-contrib pytest

# Verificar instalación
python -c "import gymnasium, torch, stable_baselines3; print('OK')"
```

### 2.5 Si falla la activación del venv (error de ejecución de scripts)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
.venv\Scripts\Activate.ps1
```

### 2.6 Ejecutar tests para verificar que todo funciona

```powershell
python -m pytest tests/ -q
# Debe mostrar: 116 passed
```

---

## 🚂 3. Continuar el entrenamiento

### 3.1 Evaluar el modelo actual (para tener baseline)

```powershell
python evaluar_modelo.py --ruta modelos_historicos/v2/snapshot_0003500000 --partidas 200
```

### 3.2 Continuar entrenamiento v2 (Self-Play)

El directorio `v2/` ya contiene 20 snapshots de calidad. El script usará automáticamente
**Self-Play v2** (60% snapshots + 40% bots heurísticos).

```powershell
# Ronda 2: Self-Play real con 1M pasos adicionales
python train_self_play.py --self-play --steps 1000000 --snapshot-every 100000
```

> **Duración estimada:** ~15-20 minutos por cada 100k pasos en CPU moderna.

### 3.3 Si el win rate baja con Self-Play

Es normal una caída de 5-10% al introducir self-play. Si baja de 70%:

```powershell
# Opción A: Entrenar solo contra bots (más seguro, más lento)
python train_self_play.py --steps 1000000 --snapshot-every 100000

# Opción B: Aumentar prob_bot para más anclaje
python train_self_play.py --self-play --prob-bot 0.60 --steps 1000000
```

### 3.4 Evaluar periódicamente

Cada 500k pasos, evalúa el último snapshot:

```powershell
# Listar snapshots disponibles
python -c "from train_self_play import listar_snapshots_v2; snaps=listar_snapshots_v2(); print(f'Total: {len(snaps)}'); [print(f'  {s}') for s in snaps[-5:]]"

# Evaluar el más reciente
python evaluar_modelo.py --ruta modelos_historicos/v2/snapshot_0004000000 --partidas 100
```

---

## 📊 4. Monitorear con TensorBoard

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

## 📁 5. Estructura del proyecto

```
CorazonesRL/
├── evaluar_modelo.py          # Script de evaluación (CORREGIDO)
├── train_self_play.py         # Pipeline v2 anti-colapso
├── test_stress.py             # Prueba de estrés (10k manos)
├── INSTRUCCIONES.md
├── 0_Plan_de_Desarrollo_y_Testing.md
├── 1_Reglas_y_Motor_Corazones.md
├── 2_Entorno_IA_Gymnasium.md
├── 3_Arquitectura_Agente_RL.md
├── src/
│   ├── baraja.py
│   ├── bots.py
│   ├── carta.py
│   ├── entorno.py
│   ├── entorno_multi.py
│   ├── jugador.py
│   ├── motor.py
│   └── red.py
├── tests/
│   ├── test_modulo1.py        # Motor del juego
│   ├── test_modulo2.py        # Entorno Gymnasium
│   └── test_modulo3.py        # RL, evaluación, pipeline v2
├── modelos_historicos/
│   ├── snapshot_0001500000.zip  # Mejor modelo v1 (79.5%)
│   ├── v1_backup/               # Snapshots viejos (preservados)
│   └── v2/                      # Snapshots v2 (84.5% en 3.5M)
│       ├── snapshot_0001600000.zip
│       ├── ...
│       └── snapshot_0003500000.zip  # 🏆 MEJOR MODELO
├── vecnormalize/
│   ├── vecnorm.pkl            # VecNormalize v1
│   ├── v1_vecnorm.pkl         # Backup v1
│   ├── v2_vecnorm.pkl         # VecNormalize v2
│   └── v2_vecnorm_final.pkl
└── logs/                      # TensorBoard logs
```

---

## 🔑 6. Comandos rápidos de referencia

```powershell
# Activar entorno
.venv\Scripts\Activate.ps1

# Tests
python -m pytest tests/ -q

# Evaluar modelo
python evaluar_modelo.py --ruta modelos_historicos/v2/snapshot_0003500000 --partidas 200

# Entrenar Self-Play
python train_self_play.py --self-play --steps 1000000 --snapshot-every 100000

# Entrenar solo bots
python train_self_play.py --steps 1000000 --snapshot-every 100000

# Estrés (10k manos aleatorias)
python test_stress.py

# TensorBoard
tensorboard --logdir logs/

# Listar snapshots v2
python -c "from train_self_play import listar_snapshots_v2; [print(s) for s in listar_snapshots_v2()]"
```

---

## ⚙️ 7. Hiperparámetros v2 (aplicados automáticamente)

| Parámetro | v1 | v2 | Motivo |
|-----------|----|----|--------|
| `learning_rate` | 3e-5 | **1e-4** | Reactivar aprendizaje |
| `ent_coef` | 0.05 | **0.08** | Más exploración |
| `n_epochs` | 10 | **8** | Reducir sobreajuste |
| `max_grad_norm` | 0.5 | **1.0** | Permitir gradientes mayores |
| `prob_bot` | 15% | **40%** | Anclaje a heurísticos |
| Quality filter | No | **≥200k pasos** | Evitar oponentes inmaduros |
| Snapshot pruning | No | **Máx 25** | Pool diverso y manejable |

---

## ⚠️ 8. Notas importantes

1. **El VecNormalize debe coincidir con el modelo.** Si entrenas con un nuevo VecNormalize, el modelo anterior no funcionará correctamente. El script `train_self_play.py` maneja esto automáticamente.

2. **No borres `v1_backup/`.** Contiene los snapshots originales por si necesitas referencias o rollback.

3. **El `learning_rate` ahora SÍ se aplica correctamente.** Se agregó la actualización explícita del optimizador de PyTorch (antes solo se cambiaba el atributo del modelo, no el optimizer).

4. **Si cambias de PC frecuentemente**, considera usar Git LFS para los modelos o almacenarlos en la nube (Google Drive, Dropbox).
