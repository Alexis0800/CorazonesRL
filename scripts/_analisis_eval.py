"""Analisis del eval_log v3_mcts — tendencias y WR ajustado por dificultad."""
import json
import numpy as np

with open("models/v3_mcts/eval_log.jsonl") as f:
    entries = [json.loads(line) for line in f if line.strip()]

# Filtrar solo evaluaciones
evals = [e for e in entries if "wr" in e]
bcs = [e for e in entries if e.get("tipo") == "bc_finetune"]

pasos = [e["paso"] for e in evals]
wrs = [e["wr"] for e in evals]
scores = [e["avg_score"] for e in evals]
probs = [e["prob_bot"] for e in evals]

print("=" * 65)
print("  ANALISIS DEL ENTRENAMIENTO v3_mcts")
print("=" * 65)

# Bloques de 300K
print(f"\n{'Bloque':<14} {'WR avg':>8} {'Score':>8} {'prob_bot':>9}")
print("-" * 42)
for i in range(0, len(evals), 3):
    cw = wrs[i: i + 3]
    cs = scores[i: i + 3]
    cp = probs[i: i + 3]
    ini = pasos[i] // 1000
    fin = pasos[min(i + 2, len(evals) - 1)] // 1000
    print(f"{ini}K-{fin}K       {np.mean(cw):>7.1%}  {np.mean(cs):>7.2f}  {np.mean(cp):>7.3f}")

# Tendencia
x = np.arange(len(wrs))
z_wr = np.polyfit(x, wrs, 1)
z_sc = np.polyfit(x, scores, 1)
print(
    f"\nTendencia WR:     {z_wr[0]:+.4f}/eval  ({z_wr[0]*len(wrs):+.1%} en todo el run)")
print(f"Tendencia Score:  {z_sc[0]:+.4f}/eval  (negativo = mejora)")

# WR ajustado por dificultad
print(f"\n{'='*65}")
print("  WR AJUSTADO POR DIFICULTAD")
print("  Formula: WR_adj = WR / (0.30 + prob_bot)")
print("  A mayor adj, mejor rendimiento REAL del modelo")
print(f"{'='*65}")
print(f"{'Paso':<8} {'WR':>6} {'pb':>6} {'WR_adj':>8} {'Progreso'}")
print("-" * 48)
for p, w, pb in zip(pasos, wrs, probs):
    adj = w / (0.30 + pb)
    bar = "|" + "=" * int(adj * 30) + ">"
    print(f"{p//1000:>4}K  {w:>5.0%}  {pb:.3f}  {adj:>7.4f}  {bar}")

# Resumen
print(f"\n{'='*65}")
print("  CONCLUSION")
print(f"{'='*65}")
first_half_adj = np.mean([w / (0.30 + pb)
                         for w, pb in zip(wrs[:6], probs[:6])])
second_half_adj = np.mean([w / (0.30 + pb)
                          for w, pb in zip(wrs[6:], probs[6:])])
print(f"  WR ajustado 100K-600K:  {first_half_adj:.4f}")
print(f"  WR ajustado 700K-1.2M:  {second_half_adj:.4f}")
print(f"  Cambio:                 {second_half_adj - first_half_adj:+.4f}")
if second_half_adj > first_half_adj:
    print(f"  >> El modelo MEJORA contra oponentes mas fuertes <<")
else:
    print(f"  >> Rendimiento estable — no hay degradacion <<")
