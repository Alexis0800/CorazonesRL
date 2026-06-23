"""Análisis comparativo old vs new training run."""
import json

with open('models/v3_mcts/eval_log.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

dificil = [e for e in lines if e.get('formato') == 'dificil']
facil = [e for e in lines if e.get('formato') == 'facil']
bc = [e for e in lines if e.get('tipo') == 'bc_finetune']

print('=' * 70)
print('COMPARATIVA OLD vs NEW RUN')
print('=' * 70)

print('\n--- OLD RUN (BC reg=1e-5, QS=-8, corazon=-1, MCTS=0.40, lr=1e-4) ---')
print('  DIFICIL score range: 7.22 - 8.62')
print('  FACIL   score range: 6.66 - 8.34')
print('  Best DIFICIL: 7.22 (500K)')
print('  Best FACIL:   6.66 (700K)')

print('\n--- NEW RUN (BC reg=OFF, QS=-15, corazon=-3, MCTS=0.60, lr=5e-5) ---')
scores_d = [e['avg_score'] for e in dificil]
scores_f = [e['avg_score'] for e in facil]
print(f'  DIFICIL score range: {min(scores_d):.2f} - {max(scores_d):.2f}')
print(f'  FACIL   score range: {min(scores_f):.2f} - {max(scores_f):.2f}')
best_d = min(dificil, key=lambda e: e['avg_score'])
best_f = min(facil, key=lambda e: e['avg_score'])
print(f'  Best DIFICIL: {best_d["avg_score"]:.2f} ({best_d["paso"]//1000}K)')
print(f'  Best FACIL:   {best_f["avg_score"]:.2f} ({best_f["paso"]//1000}K)')

print('\n--- BC LOSS (aprendizaje del oraculo) ---')
for b in bc:
    print(
        f'  {b["paso"]//1000:>4}K: BC loss={b["bc_loss"]:.4f}  buffer={b["buffer_size"]}')

print('\n--- TENDENCIA DIFICIL (completa) ---')
for e in dificil:
    arrow = '↓' if e['avg_score'] < 7.5 else (
        '↑' if e['avg_score'] > 8.0 else '→')
    bar = '🟢' if e['avg_score'] < 7.0 else (
        '🟡' if e['avg_score'] < 8.0 else '🔴')
    print(f'  {e["paso"]//1000:>4}K: {bar} Score={e["avg_score"]:.2f} {arrow} | '
          f'Top1={e["top1_rate"]:.0%} Top2={e["top2_rate"]:.0%} | '
          f'modelo={e["scores_por_jugador"]["modelo"]:.1f} vs '
          f'exp1={e["scores_por_jugador"]["experto_1"]:.1f}')

print('\n--- MEJORA vs OLD ---')
old_best = 7.22
new_best = min(scores_d)
delta = old_best - new_best
print(f'  Old best DIFICIL: {old_best:.2f}')
print(f'  New best DIFICIL: {new_best:.2f}')
print(f'  Delta: {delta:.2f} pts ({delta/old_best*100:.0f}% mejora)')

old_best_f = 6.66
new_best_f = min(scores_f)
delta_f = old_best_f - new_best_f
print(f'  Old best FACIL:   {old_best_f:.2f}')
print(f'  New best FACIL:   {new_best_f:.2f}')
print(f'  Delta: {delta_f:.2f} pts ({delta_f/old_best_f*100:.0f}% mejora)')

# Análisis: ¿el modelo está mejorando o estancado?
print('\n--- DIAGNOSTICO DE CONVERGENCIA ---')
first_half = [e['avg_score'] for e in dificil[:len(dificil)//2]]
second_half = [e['avg_score'] for e in dificil[len(dificil)//2:]]
print(f'  Primera mitad (media): {sum(first_half)/len(first_half):.2f}')
print(f'  Segunda mitad (media): {sum(second_half)/len(second_half):.2f}')
trend = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
print(
    f'  Tendencia: {"MEJORANDO" if trend < 0 else "EMPEORANDO"} ({trend:+.2f})')

# 4to lugar %
p4_rates = [e['posiciones'][3]/sum(e['posiciones']) for e in dificil]
print(f'  4to lugar (media): {sum(p4_rates)/len(p4_rates):.0%}')
print(
    f'  4to lugar (min):   {min(p4_rates):.0%} (a {dificil[p4_rates.index(min(p4_rates))]["paso"]//1000}K)')
