# Modelos de producción — Corazones RL

Modelos curados (los mejores) versionados con **Git LFS**. El resto de `models/`
está en `.gitignore` (checkpoints de entrenamiento, no se suben).

## Modelos

| Carpeta / archivo | Qué es | obs_dim | Pase | Origen (mejor por eval pareada) |
|---|---|---|---|---|
| `v10c_campeon/` | **Modelo principal (promovido 2026-07-06).** Fine-tune de +5M pasos sobre el v10c anterior, con `BotLunatico` con el doble de peso en el pool de self-play (`src/rllib/opponent_pool.py`). Validado en 33 partidas reales held-out (nunca vistas en el fine-tune): regret medio 1.261→1.227, liderando 1.988→1.764 (-11.3%), manos con pozo 2.435→2.298 — mejora en las 3 métricas, sin retrocesos. Ver `docs/superpowers/plans/2026-07-06-moon-prob-modelo-aprendido.md` y el chat del 2026-07-06 para el diagnóstico completo (moon_prob aprendido no ayudó; corrección BC supervisada tampoco — sí ayudó más self-play con oponentes reforzados). | 228 | ✅ | `models/v10c_finetune_pozo/snapshots/snapshot_000025001984` (resume de `v10c/snapshots/snapshot_000020004864`) |
| `v10c_campeon_20260706_anterior/` | Campeón anterior a la promoción de arriba (preservado para comparar/revertir). | 228 | ✅ | `v10c/elite/elite_000019169280` — puesto medio **1.703** vs mesa mixta |
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
