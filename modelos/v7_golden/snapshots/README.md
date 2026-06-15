# 🏆 Checkpoint Dorado V7 — snapshot_0014900000

**Fecha**: 2026-06-12
**Pasos de entrenamiento**: 14,900,000
**Generación**: v7 (MaskablePPO, 194-dim, reanudado desde golden v6 10M)

## ¿Por qué es dorado?

Es el **mejor modelo del proyecto**, superando por casi 200 Elo al golden anterior.

- **Torneo Elo Puro (cross-gen v7 vs v6 vs v5, 20 partidas/par)**: #1 con **1723 Elo**
- **Top 3**: TODO v7 (14.9M=1723, 15.0M=1717, 14.5M=1708) — diferencia de solo 15 pts
- **vs Golden v6 (10M)**: 14-6 (70% win rate)
- **vs Mejor v5 (18.35M)**: 17-3 (85% win rate)
- **vs v6 degradado (14.4M)**: 17-3 (85% win rate)
- **WR bots**: 84%, Top-2: 93%, AvgPts: 35.7

## Mejora sobre v6

| Métrica | V6 Golden (10M) | V7 Champion (14.9M) |
|---------|-----------------|---------------------|
| Elo Puro | 1532 | **1723** (+191) |
| WR bots | 73% | **84%** (+11pp) |
| vs v5_18.35M | ~10-5 | **17-3** |

## ¿Qué cambió?

- **Cosine decay prob_bot**: 50%→20% (más self-play controlado)
- **Reanudado desde golden 10M**: No empezó de cero, refinó lo mejor
- **5M pasos adicionales**: 10M→15M total

## Uso

```bash
# Evaluar
python evaluar_modelo.py --modelo modelos_historicos/golden_v7/snapshot_0014900000.zip

# Jugar contra él
python jugar_contra_modelo.py --modelo modelos_historicos/golden_v7/snapshot_0014900000.zip

# Reanudar entrenamiento desde aquí
python train_auto_v6.py --resume modelos_historicos/golden_v7/snapshot_0014900000 --output-dir modelos_historicos/v8
```

## Archivos

- `snapshot_0014900000.zip` — Pesos del modelo (~5.1 MB)
- `snapshot_0014900000_vecnorm.pkl` — Estadísticas VecNormalize (~7.3 KB)

## NO borrar ni sobrescribir

Este directorio está congelado.
