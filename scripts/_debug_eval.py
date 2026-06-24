"""Debug: test exacto de _evaluar_estandar sin modelo."""
from src.agentes.heuristicos import BOTS_DISPONIBLES
from src.agentes.bot_experto import BotExperto
from src.v3_1.entorno import CorazonesEnvV31
import sys
import time
import random
sys.path.insert(0, r'd:\Python\corazones-neuralnetwork')


print('Test 1: env standalone con [Experto, Experto, Bot]...')
for seed_offset in range(3):
    agente_idx = seed_offset % 4
    politicas = {}
    oponentes = [(agente_idx + d) % 4 for d in (1, 2, 3)]
    politicas[oponentes[0]] = BotExperto()
    politicas[oponentes[1]] = BotExperto()
    politicas[oponentes[2]] = random.choice(BOTS_DISPONIBLES)

    env = CorazonesEnvV31(agente_idx=agente_idx, politicas_oponentes=politicas)
    print(f'  Seed {seed_offset}: reset...')
    start = time.time()
    obs, _ = env.reset(seed=42 + seed_offset)
    print(f'    reset OK in {time.time()-start:.2f}s, obs={obs.shape}')

    steps = 0
    done = False
    while not done and steps < 50:
        mask = env.action_masks()
        if mask.sum() == 0:
            break
        action = int(mask.argmax())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    score = info.get('score', '?')
    print(f'    {steps} steps, score={score}, done={done}')
    env.close()

print('OK - all tests passed')
