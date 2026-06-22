"""Analisis de posiciones del eval_log v4."""
import json
import numpy as np
with open("models/v3_mcts/eval_log.jsonl") as f:
    lines = [json.loads(l) for l in f if l.strip()]
evals = [e for e in lines if "wr" in e]

print("=== METRICAS DE POSICION ===")
print(f"{'Paso':<8} {'WR':>5} {'Score':>6} {'1ro':>5} {'2do':>5} {'3ro':>5} {'4to':>5} {'Top2':>6} {'Exp1':>6} {'Exp2':>6} {'Bot':>6}")
for e in evals:
    p = e["posiciones"]
    s = e["scores_por_jugador"]
    top2 = (p[0]+p[1])/e["num_partidas"]
    print(f"{e['paso']//1000:>4}K  {e['wr']:>4.0%}  {e['avg_score']:>5.2f}  {p[0]:>4}  {p[1]:>4}  {p[2]:>4}  {p[3]:>4}  {top2:>5.0%}  {s['experto_1']:>5.2f}  {s['experto_2']:>5.2f}  {s['bot']:>5.2f}")

print()
print("DIAGNOSTICO:")
print(
    f"  Modelo 1ro: {np.mean([e['posiciones'][0] for e in evals]):.0f}/100 manos")
print(
    f"  Modelo Top2: {np.mean([e['posiciones'][0]+e['posiciones'][1] for e in evals]):.0f}/100 manos")
print(
    f"  Exp1 score: {np.mean([e['scores_por_jugador']['experto_1'] for e in evals]):.2f}")
print(
    f"  Exp2 score: {np.mean([e['scores_por_jugador']['experto_2'] for e in evals]):.2f}")
print(f"  Modelo score: {np.mean([e['avg_score'] for e in evals]):.2f}")
print()
print("CONCLUSION: Los 2 BotExperto dominan la mesa (score 4-6 vs 7-8 del modelo).")
print("El formato [Modelo, Experto, Experto, Bot] es demasiado dificil.")
print("WR<=8 es enganoso: el modelo hace 7pts pero los Expertos hacen 4pts.")
print("Para ganar partidas, el modelo necesita bajar de 7.5 a <5.0 pts/mano.")
