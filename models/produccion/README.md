# Modelos de producción — Corazones RL

Modelos curados (los mejores) versionados con **Git LFS**. El resto de `models/`
está en `.gitignore` (checkpoints de entrenamiento, no se suben).

## Modelos

| Carpeta / archivo | Qué es | obs_dim | Pase | Origen (mejor por eval pareada) |
|---|---|---|---|---|
| `v10c_campeon/` | **Modelo principal.** PPO con pase + self-play neuronal consistente. | 228 | ✅ | `v10c/elite/elite_000019169280` — puesto medio **1.703** vs mesa mixta |
| `v10_nopase_campeon/` | Mejor agente SIN pase (variantes sin pase). | 224 | ❌ | `v10_bc_ppo/elite/elite_000016506880` — puesto **1.500** / win 0.58 / top2 0.92 |
| `bc_base_con_pase.pkl` | Pesos BC (imitación de PIMC) con pase. Base para re-init de PPO. | 228 | ✅ | `bc_v10b.pkl` |

> **Cómo se eligieron:** NO por el "score" de entrenamiento (bot_eval de 50 partidas,
> ruidoso y poco fiable — de hecho elegía mal). Se seleccionaron con `comparar_snapshots.py`:
> evaluación **pareada** (mismas semillas) de todos los elites + últimos snapshots,
> 300 partidas vs la **mesa mixta** (experto+castigador+lunático), rankeando por
> **puesto medio** (baja varianza). Para re-verificar:
> `python comparar_snapshots.py --dir models/v10c --con-pase --partidas 300`
>
> Referencia: en mesa de 4 con rivales fuertes, el azar da puesto 2.5 / win 0.25 / top2 0.50.

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
