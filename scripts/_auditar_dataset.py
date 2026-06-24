"""Audita el dataset PIMC verificando reglas del juego de Corazones."""
from src.dominio.carta import Carta
import numpy as np
import sys
sys.path.insert(0, r'd:\Python\corazones-neuralnetwork')


data = np.load('datasets/pimc_v3_1.npz')
obs = data['observations']      # (N, 228)
scores = data['all_scores']     # (N, 52)
bazas = data['bazas']           # (N,)
sits = data['situaciones']      # (N,) — 'seguir', 'liderar', 'primera'

N = len(obs)
print(f'Dataset: {N:,} muestras, {obs.shape[1]} dims')
print()

# ── Regla 1: Primera baza, mesa vacía → solo 2♣ es legal ──
print('=' * 60)
print('REGLA 1: Primera jugada (baza 1, mesa vacía) → solo 2♣')
print('=' * 60)

idx_b1 = np.where(bazas == 1)[0]
violaciones_b1 = 0
for i in idx_b1[:1000]:  # muestra
    o = obs[i]
    mano = o[0:52]  # one-hot de la mano
    mesa = o[52:104]  # one-hot de la mesa

    if mesa.sum() > 0:
        continue  # no es primera jugada (ya hay cartas en mesa)

    cartas_en_mano = np.where(mano == 1.0)[0]
    if len(cartas_en_mano) == 0:
        continue

    # ¿Tiene 2♣? (id=0)
    tiene_2c = mano[0] == 1.0

    # Verificar scores: ¿hay scores válidos para cartas que NO son 2♣?
    sc = scores[i]
    for cid in cartas_en_mano:
        if cid != 0 and sc[cid] >= 0:
            violaciones_b1 += 1
            if violaciones_b1 <= 3:
                carta = Carta._TODAS[cid]
                print(
                    f'  ⚠️  Muestra {i}: baza 1, mesa vacía, score válido para {carta} (id={cid}) = {sc[cid]:.1f}')
            break

if violaciones_b1 == 0:
    print(f'  ✅ {min(len(idx_b1), 1000)} muestras verificadas, 0 violaciones')
else:
    print(f'  ❌ {violaciones_b1} violaciones encontradas')

# ── Regla 2: Asistir al palo ──
print()
print('=' * 60)
print('REGLA 2: Si hay palo de salida, debo asistir si tengo')
print('=' * 60)

violaciones_palo = 0
for i in range(min(5000, N)):
    o = obs[i]
    mano = o[0:52]
    mesa = o[52:104]
    palo_salida_feat = o[193]  # en v3.1, palo_salida está en [193]

    # palo_salida: 0.0 si None, else palo/3.0
    # palo 0 (trébol) = 0.0 → ambiguity con None!
    # Pero en baza 1 con mesa vacía, palo_salida=None → 0.0
    # Si mesa NO está vacía, palo_salida debe ser != 0 o es trébol

    if mesa.sum() == 0:
        continue  # no hay palo de salida (mesa vacía)

    # Determinar palo de salida
    if palo_salida_feat == 0.0 and mesa.sum() > 0:
        # Puede ser None o trébol (0/3=0). Con mesa no vacía, es trébol.
        palo_salida = 0
    elif palo_salida_feat > 0:
        palo_salida = int(round(palo_salida_feat * 3))
    else:
        continue

    cartas_en_mano = np.where(mano == 1.0)[0]
    tengo_del_palo = any(
        Carta._TODAS[cid].palo == palo_salida for cid in cartas_en_mano
    )

    if not tengo_del_palo:
        continue  # no tengo del palo → puedo jugar cualquier cosa

    # Tengo del palo → debo jugar una de ese palo
    # Verificar que los scores para cartas de OTRO palo sean -1 (ilegales)
    sc = scores[i]
    for cid in cartas_en_mano:
        carta = Carta._TODAS[cid]
        if carta.palo != palo_salida and sc[cid] >= 0:
            violaciones_palo += 1
            if violaciones_palo <= 5:
                print(f'  ⚠️  Muestra {i}: baza={bazas[i]}, palo_salida={palo_salida}, '
                      f'tengo {carta} (palo={carta.palo}) pero score={sc[cid]:.1f} ≥ 0')
            break

if violaciones_palo == 0:
    print(f'  ✅ {min(5000, N)} muestras verificadas, 0 violaciones')
else:
    print(f'  ❌ {violaciones_palo} violaciones de "asistir al palo"')

# ── Regla 3: No liderar corazones sin romper ──
print()
print('=' * 60)
print('REGLA 3: No liderar corazones si no están rotos')
print('=' * 60)

violaciones_cor = 0
for i in range(min(5000, N)):
    o = obs[i]
    mano = o[0:52]
    mesa = o[52:104]
    corazones_rotos = o[180]  # [180] en v3.1

    if mesa.sum() > 0:
        continue  # no estoy liderando

    if corazones_rotos >= 0.5:
        continue  # corazones rotos → puedo liderar con corazones

    # Corazones NO rotos, mesa vacía → no puedo liderar con corazones
    # A MENOS que solo tenga corazones
    cartas_en_mano = np.where(mano == 1.0)[0]
    tengo_no_corazones = any(
        not Carta._TODAS[cid].es_corazon for cid in cartas_en_mano
    )

    if not tengo_no_corazones:
        continue  # solo tengo corazones → forzado

    # Tengo no-corazones → los scores de corazones deberían ser -1
    sc = scores[i]
    for cid in cartas_en_mano:
        carta = Carta._TODAS[cid]
        if carta.es_corazon and sc[cid] >= 0:
            violaciones_cor += 1
            if violaciones_cor <= 5:
                print(f'  ⚠️  Muestra {i}: baza={bazas[i]}, ♥ rotos={corazones_rotos}, '
                      f'tengo {carta} pero score={sc[cid]:.1f} ≥ 0')
            break

if violaciones_cor == 0:
    print(f'  ✅ {min(5000, N)} muestras verificadas, 0 violaciones')
else:
    print(f'  ❌ {violaciones_cor} violaciones de "no liderar corazones"')

# ── Estadísticas generales ──
print()
print('=' * 60)
print('ESTADÍSTICAS DEL DATASET')
print('=' * 60)

# ¿Cuántas muestras tienen scores válidos (≥0)?
scores_validos = (scores >= 0).sum(axis=1)
print(f'  Scores válidos por muestra: min={scores_validos.min()}, max={scores_validos.max()}, '
      f'media={scores_validos.mean():.1f}')

# ¿La acción óptima (argmin de scores) es siempre una carta en mano?
optimas_erroneas = 0
for i in range(min(5000, N)):
    o = obs[i]
    mano = o[0:52]
    sc = scores[i]
    masked = np.where(sc >= 0, sc, np.inf)
    if masked.min() == np.inf:
        continue
    optima_id = np.argmin(masked)
    if mano[optima_id] != 1.0:
        optimas_erroneas += 1
        if optimas_erroneas <= 3:
            print(
                f'  ⚠️  Muestra {i}: acción óptima id={optima_id} NO está en la mano!')

if optimas_erroneas == 0:
    print(
        f'  ✅ Acción óptima siempre en mano (verificado {min(5000, N)} muestras)')
else:
    print(f'  ❌ {optimas_erroneas} acciones óptimas fuera de la mano')

print()
print('✅ Auditoría completada')
