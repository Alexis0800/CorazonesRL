# Modelos de producción — Corazones RL

Modelos curados (los mejores) versionados con **Git LFS**. El resto de `models/`
está en `.gitignore` (checkpoints de entrenamiento, no se suben).

## Modelos

| Carpeta / archivo | Qué es | obs_dim | Pase | Rendimiento (150 partidas) |
|---|---|---|---|---|
| `v10c_campeon/` | **Modelo principal.** PPO fine-tune con pase + self-play neuronal consistente. | 228 | ✅ | win vs experto **0.66** / top2 0.83; global win 0.76 / top2 0.93 |
| `v10_nopase_campeon/` | Mejor agente SIN pase (por si juegas variantes sin pase). | 224 | ❌ | win vs experto 0.49 / top2 0.81; global 0.70 / 0.925 |
| `bc_base_con_pase.pkl` | Pesos BC (imitación de PIMC) con pase. Base para re-entrenar/inicializar PPO. | 228 | ✅ | top2 global ~0.90 (sin PPO) |

> Referencia: en una mesa de 4 con 3 rivales fuertes, el azar da win 0.25 / top2 0.50.
> v10c gana ~2 de cada 3 mesas vs tres BotExperto — muy por encima del azar.

## Cómo usarlos

**Copiloto interactivo** (te recomienda qué pasar/jugar):
```bash
python recomendador.py --modelo models/produccion/v10c_campeon
```

**Ver al modelo jugar** (traza completa):
```bash
python jugar_modelo.py --modelo models/produccion/v10c_campeon --ver
```

**Evaluar** (con pase):
```bash
python evaluar_final.py --dir models/produccion --snapshot models/produccion/v10c_campeon --con-pase --partidas 200
```

## Git LFS

Para clonar/descargar estos modelos hace falta `git lfs`:
```bash
git lfs install
git clone <repo>          # LFS baja los .pkl automáticamente
# o en un repo ya clonado:
git lfs pull
```
