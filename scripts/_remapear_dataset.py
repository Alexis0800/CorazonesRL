"""Remapea dataset v3 (265 dims) → v3.1 (228 dims) con mapeo correcto."""
import numpy as np
import sys
import os
sys.path.insert(0, r'd:\Python\corazones-neuralnetwork')


def remapear_v3_a_v31(obs_v3: np.ndarray) -> np.ndarray:
    """Convierte observación v3 (265 dims) a v3.1 (228 dims).

    Mapeo de bloques (v3 → v3.1):
      [0:52]     mano           → [0:52]     mano
      [52:104]   mesa           → [52:104]   mesa
      [104:156]  cementerio     → [104:156]  cementerio
      [156:172]  vacíos         → [156:172]  vacíos
      [172:176]  histórico/100  → [172:176]  histórico/100
      [245:249]  puntos raw     → [176:180]  puntos raw  (v3.1 usa raw, no /26)
      [180]      corazones rotos→ [180]      ♥ rotos
      [181]      posición       → [181]      posición
      [182:187]  Q♠ tracker     → [182:187]  Q♠ tracker
      [187]      pozo_viable    → [187]      pozo_viable
      [194]      baza/13        → [188]      baza/13
      [220:224]  cartas rest raw→ [189:193]  cartas rest raw
      [219]      palo_salida    → [193]      palo_salida
      [224:228]  peligro Q♠     → [194:198]  peligro Q♠
      [249]      peligro Q♠ inmin→ [198]     peligro Q♠ inmin
      [260:264]  prob Q♠ v2     → [199:203]  prob Q♠ v2
      [228:232]  altas en mano  → [203:207]  altas en mano
      [252:256]  máxima absoluta→ [207:211]  máxima absoluta
      [236:240]  control palo   → [211:215]  control palo
      NUEVO                      → [215:219]  ♥ altos rival (0)
      NUEVO                      → [219:223]  ♠ altas rival (0)
      NUEVO                      → [223:227]  ¿jugó ♥? (0)
      [264]      forzado        → [227]      forzado
    """
    N = obs_v3.shape[0]
    obs_v31 = np.zeros((N, 228), dtype=np.float32)

    # Bloques que van en la misma posición
    obs_v31[:, 0:52] = obs_v3[:, 0:52]         # mano
    obs_v31[:, 52:104] = obs_v3[:, 52:104]     # mesa
    obs_v31[:, 104:156] = obs_v3[:, 104:156]   # cementerio
    obs_v31[:, 156:172] = obs_v3[:, 156:172]   # vacíos
    obs_v31[:, 172:176] = obs_v3[:, 172:176]   # histórico /100
    obs_v31[:, 180] = obs_v3[:, 180]           # corazones rotos
    obs_v31[:, 181] = obs_v3[:, 181]           # posición
    obs_v31[:, 182:188] = obs_v3[:, 182:188]   # Q♠ tracker + pozo_viable

    # Puntos mano actual: v3 tiene /26 en [176:180], v3.1 usa raw [245:249]
    obs_v31[:, 176:180] = obs_v3[:, 245:249]   # puntos raw

    # Estado de mano
    obs_v31[:, 188] = obs_v3[:, 194]           # baza/13
    obs_v31[:, 189:193] = obs_v3[:, 220:224]   # cartas restantes raw
    obs_v31[:, 193] = obs_v3[:, 219]           # palo_salida

    # Q♠ enriquecido
    obs_v31[:, 194:198] = obs_v3[:, 224:228]   # peligro Q♠
    obs_v31[:, 198] = obs_v3[:, 249]           # peligro Q♠ inminente
    obs_v31[:, 199:203] = obs_v3[:, 260:264]   # prob Q♠ v2

    # Control de palo
    obs_v31[:, 203:207] = obs_v3[:, 228:232]   # altas en mano
    obs_v31[:, 207:211] = obs_v3[:, 252:256]   # máxima absoluta
    obs_v31[:, 211:215] = obs_v3[:, 236:240]   # control palo

    # Patrones de rivales: no existen en v3, se dejan en 0
    # [215:227] = 0

    # Forzado
    obs_v31[:, 227] = obs_v3[:, 264]           # forzado

    return obs_v31


# ── Aplicar remapeo ──
os.chdir(r'd:\Python\corazones-neuralnetwork')

print('Cargando dataset v3 (265 dims)...')
data = np.load('datasets/pimc_v2.npz')
obs_v3 = data['observations']
scores = data['all_scores']
bazas = data['bazas']
sits = data['situaciones']
print(f'  {len(obs_v3):,} muestras, {obs_v3.shape[1]} dims')

print('Remapeando a v3.1 (228 dims)...')
obs_v31 = remapear_v3_a_v31(obs_v3)
print(f'  {obs_v31.shape[1]} dims')

# Verificar algunos mapeos
print()
print('Verificación de mapeo:')
i = 100
print(f'  Muestra {i}:')
print(f'    v3 palo_salida[219] = {obs_v3[i, 219]:.3f}')
print(f'    v31 palo_salida[193] = {obs_v31[i, 193]:.3f}')
print(f'    v3 baza[194] = {obs_v3[i, 194]:.3f}')
print(f'    v31 baza[188] = {obs_v31[i, 188]:.3f}')
print(f'    v3 forzado[264] = {obs_v3[i, 264]:.3f}')
print(f'    v31 forzado[227] = {obs_v31[i, 227]:.3f}')
print(f'    v3 prob_qs[260:264] = {obs_v3[i, 260:264]}')
print(f'    v31 prob_qs[199:203] = {obs_v31[i, 199:203]}')
assert abs(obs_v3[i, 219] - obs_v31[i, 193]) < 0.01, "palo_salida mismatch"
assert abs(obs_v3[i, 194] - obs_v31[i, 188]) < 0.01, "baza mismatch"
print('  ✅ Mapeos verificados')

# Guardar
print()
print('Guardando datasets/pimc_v3_1.npz...')
np.savez_compressed('datasets/pimc_v3_1.npz',
                    observations=obs_v31.astype(np.float32),
                    all_scores=scores.astype(np.float32),
                    bazas=bazas,
                    situaciones=sits,
                    )
print(f'  Tamaño: {os.path.getsize("datasets/pimc_v3_1.npz")/1024:.0f} KB')
print('✅ Dataset v3.1 remapeado correctamente')
