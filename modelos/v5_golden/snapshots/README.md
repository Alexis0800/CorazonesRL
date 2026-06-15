# 🏆 Checkpoint Dorado — snapshot_0010000000

**Fecha**: 2026-06-12
**Pasos de entrenamiento**: 10,000,000
**Generación**: v6 (MaskablePPO, 194-dim obs con all_void)

## ¿Por qué es dorado?

Es el **mejor modelo del proyecto hasta la fecha**:

- **Torneo Elo Puro (cross-gen v5 vs v6, 30 partidas/par)**: #1 con **1545 Elo**, 8-1-0 en enfrentamientos directos. El top 5 fue todo v6; los mejores v5 quedaron ~1400 Elo.
- **Torneo Elo con bots**: 1487 Elo (#7) — pero el modo bots subestima a los modelos fuertes (los bots son presa fácil para todos).
- **Win rate**: Pico de rendimiento en la curva de aprendizaje
- **Punto de quiebre**: Después de 10M, el modelo empezó a degradarse (sobreentrenamiento con prob_bot demasiado bajo)

## Continuación v7

Desde este checkpoint se lanzó `modelos_historicos/v7/` con cosine decay 50%→20%
y 5M pasos adicionales. **Resultado: v7_14.9M alcanzó 1723 Elo (+191).**

👉 El nuevo campeón está en `modelos_historicos/golden_v7/`. Este golden v6 queda
como referencia histórica.

## Uso

```bash
# Evaluar
python evaluar_modelo.py --modelo modelos_historicos/golden/snapshot_0010000000.zip

# Jugar contra él
python jugar_contra_modelo.py --modelo modelos_historicos/golden/snapshot_0010000000.zip

# Reanudar entrenamiento desde aquí
python train_auto_v6.py --resume modelos_historicos/golden/snapshot_0010000000.zip
```

## Archivos

- `snapshot_0010000000.zip` — Pesos del modelo (5.1 MB)
- `snapshot_0010000000_vecnorm.pkl` — Estadísticas VecNormalize (7.3 KB)

## NO borrar ni sobrescribir

Este directorio está congelado. Cualquier reanudación de entrenamiento
debe hacerse desde una copia, no desde este original.
