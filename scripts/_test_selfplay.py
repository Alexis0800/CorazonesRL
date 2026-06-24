"""Test de mano completa con self-play v3.1."""
from src.v3_1.train import crear_entorno_self_play_v31
import time
import sys
sys.path.insert(0, r'd:\Python\corazones-neuralnetwork')


print('Creando env self-play...')
env = crear_entorno_self_play_v31(version='v3_1', prob_bot=0.5, agente_idx=0)
print('Reset...')
obs = env.reset(seed=42)[0]
steps = 0
done = False
start = time.time()
while not done and steps < 200:
    mask = env.action_masks()
    if mask.sum() == 0:
        print(f'  Step {steps}: 0 legales!')
        break
    action = int(mask.argmax())
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
    if steps % 13 == 0:
        print(
            f'  Step {steps}: baza={env.motor.numero_baza}, mesa={len(env.motor.mesa)}')
elapsed = time.time() - start
score = info.get('score', '?')
print(
    f'Completado: {steps} steps en {elapsed:.1f}s, done={done}, score={score}')
env.close()
print('OK')
