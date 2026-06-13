# Fase 6 — Features all_void (190 → 194 dimensiones)

> **Motivación:** El modelo 15.4M recomienda liderar espadas cuando los 3 rivales están vacíos en ese palo, comiendo puntos innecesarios. Sabe que están vacíos (voids en `[156:172]`) pero no infiere la consecuencia: "si todos están vacíos, tirar ese palo = me como la baza".
>
> **Solución:** Agregar 4 features directos `[190:194]` que indiquen "todos los rivales están vacíos en palo X", eliminando la necesidad de que la red aprenda esta regla combinatoria de 3 bits separados.

---

## 1. Vector de observación (v6 — 194 dimensiones)

| Rango | Contenido | Cambio |
|---|---|---|
| `[0:52]` | Mano del agente (one-hot) | — |
| `[52:104]` | Mesa actual | — |
| `[104:156]` | Cementerio | — |
| `[156:172]` | Vacíos conocidos (4×4) | — |
| `[172:176]` | Puntajes históricos (/100) | — |
| `[176:180]` | Puntos mano actual (/26) | — |
| `[180]` | Corazones rotos | — |
| `[181]` | Posición en la baza | — |
| `[182:187]` | Rastreador Dama de Picas (5 estados) | — |
| `[187]` | pozo_viable | — |
| `[188]` | debo_arriesgar | — |
| `[189]` | puedo_alimentar | — |
| **`[190]`** | **all_void_tréboles** | 🆕 |
| **`[191]`** | **all_void_diamantes** | 🆕 |
| **`[192]`** | **all_void_picas** | 🆕 |
| **`[193]`** | **all_void_corazones** | 🆕 |

### Lógica de cálculo

```python
# En _construir_observacion(), después de llenar [156:172]:
# Palos: 0=Tréboles, 1=Diamantes, 2=Picas, 3=Corazones
# obs[190+palo] = 1.0 si los 3 rivales son void en ese palo
for palo in range(4):
    todos_vacios = all(
        palo in self._vacios[j] for j in range(1, 4)
    )
    obs[190 + palo] = 1.0 if todos_vacios else 0.0
```

---

## 2. Archivos a modificar

| Archivo | Cambio | Dificultad |
|---|---|---|
| `src/entorno.py` | `observation_space` → `(194,)`, `_construir_observacion()` → `np.zeros(194, ...)`, agregar `[190:194]` | Baja |
| `src/entorno_multi.py` | Ídem que `entorno.py` | Baja |
| `asesor_carta.py` | `construir_observacion_parcial()` → `np.zeros(194, ...)`, agregar `[190:194]` | Baja |
| `train_self_play.py` | `PoliticaSB3._construir_obs_desde_motor()` → `np.zeros(194, ...)`, `[190:194]` en 0. Agregar función `_transferir_pesos_190_a_194()` y flag `--transfer-from`. | Alta |
| `tests/test_entorno_v6.py` | **Nuevo** — 6 tests para features all_void | Media |
| `tests/test_entorno_v5.py` | Actualizar asserts de `(190,)` → `(194,)` | Baja |
| `tests/test_modulo2.py` | Actualizar asserts de `(190,)` → `(194,)` | Baja |
| `tests/test_modulo3.py` | Actualizar asserts de `(190,)` → `(194,)` | Baja |
| `tests/test_asesor.py` | Actualizar asserts de `(190,)` → `(194,)` | Baja |

---

## 3. Transferencia de pesos (lo más delicado)

### El problema

La primera capa de la MLP tiene pesos `W: (256, 190)` y bias `b: (256,)`. Al cambiar el input de 190→194, necesitamos `W': (256, 194)` y `b': (256,)`.

El bias se puede reutilizar tal cual (`b' = b`). Pero `W` necesita columnas nuevas.

### Solución: Padding con ceros

```python
def transferir_pesos_190_a_194(modelo_190, nuevo_modelo_194):
    """Copia pesos del modelo 190-dim al 194-dim, rellenando con ceros."""
    import torch
    
    state_190 = modelo_190.policy.state_dict()
    state_194 = nuevo_modelo_194.policy.state_dict()
    
    for key in state_194:
        if 'weight' in key and state_190[key].shape != state_194[key].shape:
            # Expandir la primera capa: (256, 190) → (256, 194)
            old_w = state_190[key]  # (256, 190)
            new_w = torch.zeros_like(state_194[key])  # (256, 194)
            new_w[:, :190] = old_w  # copiar columnas existentes
            new_w[:, 190:] = 0.0    # nuevas columnas en cero
            state_190[key] = new_w
        elif key in state_190:
            state_194[key] = state_190[key]
    
    nuevo_modelo_194.policy.load_state_dict(state_194, strict=False)
    return nuevo_modelo_194
```

### Por qué ceros y no aleatorio

- **Ceros:** Las nuevas features empiezan "apagadas" (peso 0). El modelo se comporta inicialmente como el original de 190 dims, y PPO aprende progresivamente a usar las nuevas features.
- **Aleatorio:** Podría introducir ruido que degrade la política inicial, causando un período de bajo rendimiento.

### Validación de la transferencia

Después de transferir, verificar que:

1. La política produce las mismas acciones que el modelo 190-dim original en los mismos estados (ignorando los nuevos features en 0).
2. El modelo puede entrenar sin NaN ni divergencia KL inmediata.

---

## 4. Orden de implementación (TDD)

### Paso 1: Tests

```powershell
# Crear tests/test_entorno_v6.py con 6 tests
```

### Paso 2: Entorno (src/entorno.py + src/entorno_multi.py)

```powershell
# Cambiar 190 → 194 en ambos archivos
# Agregar features [190:194]
```

### Paso 3: Asesor (asesor_carta.py)

```powershell
# Actualizar construir_observacion_parcial() a 194 dims
```

### Paso 4: Train (train_self_play.py)

```powershell
# Actualizar PoliticaSB3._construir_obs_desde_motor()
# Agregar transferir_pesos_190_a_194()
# Agregar flag --transfer-from
```

### Paso 5: Actualizar tests existentes

```powershell
# Buscar todos los (190,) → (194,) en tests/
```

### Paso 6: Validar

```powershell
python -m pytest tests/ -q
```

---

## 5. Comando para continuar entrenamiento después de la migración

```powershell
# Transferir pesos desde 15.4M y continuar:
python train_self_play.py --self-play --transfer-from modelos_historicos/v5/snapshot_0015400000 --steps 5000000 --eval-every 5
```

Esto:

1. Carga el snapshot 15.4M
2. Crea un modelo nuevo con 194 dims
3. Copia los pesos (padding con ceros las 4 columnas nuevas)
4. Guarda en `modelos_historicos/v6/` (para no mezclar con snapshots 190-dim)
5. Continúa entrenando normalmente

---

## 6. Riesgos

| Riesgo | Mitigación |
|---|---|
| Divergencia KL tras transferencia | Monitorear `approx_kl` en TensorBoard; debería ser <0.02 |
| Caída de performance inicial | Los pesos nuevos están en 0; el modelo se comporta igual que 190-dim al inicio. PPO aprende a usar las nuevas features en los primeros ~200K pasos |
| Snapshots incompatibles (190 vs 194) | Usar directorio separado (`--v6`) para evitar mezclar |

---

## 7. Notas adicionales

### Sobre el juego de suerte

El usuario observó correctamente que Corazones tiene un componente de azar. Hay manos donde ningún modelo puede evitar comer puntos. Las nuevas features `all_void` no eliminan esto, pero sí previenen el error específico de "liderar un palo donde todos están vacíos", que es un error evitable y NO producto del azar.

### Feature "todos vacíos" vs feature "cards remaining in suit"

Una alternativa más general sería agregar `[190:194]` = "cuántas cartas quedan en el palo X entre todos los jugadores" (normalizado), en vez de solo "todos vacíos". Esto daría más granularidad. Pero para el caso específico reportado, `all_void` es suficiente y más simple.
