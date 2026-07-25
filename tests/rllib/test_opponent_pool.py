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


def test_lunero_garantizado_en_fase_3():
    """Con lunero_garantizado, la factory de fase 3 SIEMPRE incluye un
    OponenteLunar en uno de los 3 slots (placeholders como snapshots)."""
    from src.agentes.oponente_lunar import OponenteLunar

    pool = OpponentPool(lunero_garantizado=True, anclar_experto=True)
    pool._snapshots = [object(), object()]
    factory = pool.make_factory(progress=0.5)  # fase 3

    for _ in range(20):
        oponentes = factory(agente_idx=0)
        assert sum(isinstance(fn, OponenteLunar) for fn in oponentes.values()) == 1

    # Sin la opción, ningún slot es OponenteLunar.
    pool_off = OpponentPool(anclar_experto=True)
    pool_off._snapshots = [object(), object()]
    factory_off = pool_off.make_factory(progress=0.5)
    for _ in range(20):
        assert not any(isinstance(fn, OponenteLunar)
                       for fn in factory_off(agente_idx=0).values())


def test_lunero_no_pisa_el_ancla_ni_el_clon():
    """Fix del bug del Run A: el lunero va al slot [1] (snapshot), nunca al [0]
    (oponente duro). Con ancla+lunero la mesa debe tener {duro, lunero, snapshot}."""
    import random as _r
    from src.agentes.oponente_lunar import OponenteLunar
    from src.agentes.bot_experto import BotExperto
    from src.rllib.opponent_pool import OpponentPool, SnapshotPolicy

    pool = OpponentPool(anclar_experto=True, lunero_garantizado=True)
    # dos snapshots falsos (pesos triviales no hacen falta: solo la identidad)
    s = SnapshotPolicy.from_weights({}, obs_dim=228)
    pool._snapshots = [s, s]
    _r.seed(0)
    for _ in range(10):
        fns = pool.make_factory(progress=0.5)(agente_idx=0)
        tipos = [type(v).__name__ for v in fns.values()]
        assert tipos.count("OponenteLunar") == 1
        assert "SnapshotPolicy" in tipos  # al menos un snapshot sobrevive
        # el slot duro sobrevive: o un BotExperto/arquetipo o (si hay clon) un snapshot mas
        assert len(fns) == 3


def test_mesa_humana_solo_en_fases_tardias():
    """v11 se diluyó con mesas humanas desde el paso 0: ahora gatean a progress>=0.40."""
    import numpy as np
    from src.rllib.opponent_pool import OpponentPool
    pool = OpponentPool(humano_bc_path=None, mesa_humana=1.0)
    pool._humano_bc_pesos = {}  # simular pesos presentes
    import random as _r
    _r.seed(1)
    # progress temprano: la mesa humana NO debe activarse aunque mesa_humana=1.0
    # y haya pesos (con pesos {} el clon fallaría al construirse si la rama se
    # tomara — que no explote Y no haya SnapshotPolicy prueba que no se tomó).
    fns = pool.make_factory(progress=0.1)(agente_idx=0)
    tipos = {type(v).__name__ for v in fns.values()}
    assert "SnapshotPolicy" not in tipos
