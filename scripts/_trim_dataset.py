"""Recorta el dataset PIMC de 265 dims (v3) a 228 dims (v3.1)."""
import numpy as np
import os

os.chdir(r'd:\Python\corazones-neuralnetwork')

data = np.load('datasets/pimc_v2.npz')
obs = data['observations']
scores = data['all_scores']
bazas = data['bazas']
sits = data['situaciones']

print(f'Dataset original: {obs.shape[1]} dims, {len(obs):,} muestras')

obs_228 = obs[:, :228].astype(np.float32)

np.savez_compressed('datasets/pimc_v3_1.npz',
                    observations=obs_228,
                    all_scores=scores.astype(np.float32),
                    bazas=bazas,
                    situaciones=sits,
                    )
print(f'Dataset v3.1: {obs_228.shape[1]} dims, {len(obs_228):,} muestras')
print(f'Tamano: {os.path.getsize("datasets/pimc_v3_1.npz") / 1024:.0f} KB')
print('Guardado: datasets/pimc_v3_1.npz')
