"""Tests de src/rllib/opponent_pool.py."""
from collections import Counter

from src.rllib.opponent_pool import OpponentPool


def test_pool_diverso_favorece_a_botlunatico():
    """BotLunatico debe salir ~2x más seguido que los otros 3 arquetipos
    (ver comentario en _bot_dificil: regret 2.35 vs 1.05 en manos con pozo).

    Fase 1 exige >=1 "snapshot" para no caer en fase 0 (bootstrap sin
    `_bot_dificil`); se inyectan placeholders directamente (`_bot_dificil` no
    los toca, evita instanciar `SnapshotPolicy`/cargar pesos de PyTorch)."""
    pool = OpponentPool(pool_diverso=True, anclar_experto=True)
    pool._snapshots = [object()]
    factory = pool.make_factory(progress=0.1)

    nombres = Counter()
    for _ in range(600):
        oponentes = factory(agente_idx=0)
        for fn in oponentes.values():
            nombres[type(fn).__name__] += 1

    lunatico = nombres.get("BotLunatico", 0)
    experto = nombres.get("BotExperto", 0)
    castigador = nombres.get("BotCastigador", 0)
    assert lunatico > experto
    assert lunatico > castigador
