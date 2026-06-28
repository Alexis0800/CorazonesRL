"""
PIMC-regret: mide la CALIDAD DE DECISIÓN de cada modelo (cuántos errores comete),
con baja varianza. Para un conjunto fijo de estados de juego:

  regret(estado) = E[puntos | carta del modelo]  −  min_c E[puntos | c]   (vía PIMC)

regret = 0 → el modelo eligió una carta óptima (según PIMC). Mayor regret = peor.
Promedio sobre muchos estados = puntos esperados que el modelo "regala" por jugada.

Eficiente: los puntajes PIMC por carta dependen SOLO del estado (no del candidato),
así que se computan UNA vez por estado (lo caro) y cada candidato solo hace forward.

Compara todos los candidatos sobre los MISMOS estados (pareado, baja varianza).
Solo evalúa decisiones de JUEGO (PIMC no evalúa el pase).

Uso:
    python pimc_regret.py --ref models/produccion/v10c_campeon --con-pase \
        --candidatos models/v10c/elite/elite_000019169280 \
                     models/v10b/snapshots/snapshot_000020004864 \
                     models/produccion/v10_nopase_campeon \
        --estados 250 --mundos 60
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.observacion import ObservacionBuilder
from src.agentes.bot_experto import BotExperto
from src.agentes.bot_castigador import BotCastigador
from src.agentes.bot_lunatico import BotLunatico
from src.mcts.pimc import (_puntaje_esperado_por_carta, _clonar_motor,
                           _crear_bots_mixto)
from src.rllib.utils import cargar_policy_desde_checkpoint


def _moon_prob(motor, idx):
    for i, jug in enumerate(motor.jugadores):
        if i != idx and jug.contar_puntos_bazas() > 0:
            return 0.0
    jug = motor.jugadores[idx]
    todas = list(jug.mano) + list(jug.bazas_ganadas)
    hh = sum(1 for c in todas if c.es_corazon and c.valor >= 10)
    hg = sum(1 for c in jug.bazas_ganadas if c.es_corazon)
    qs = any(c.es_dama_de_picas for c in todas)
    her = sum(1 for i, jr in enumerate(motor.jugadores) if i != idx
              for c in jr.mano if c.es_corazon)
    return max(0.0, min(1.0, (hh/5.0)*0.6 + (0.2 if qs else 0.0)
                        + min(hg/13.0, 1.0)*0.1 - min(her*0.015, 0.10)))


def _fin_mano(m):
    return m.numero_baza > 13 or all(len(j.mano) == 0 for j in m.jugadores)


def generar_estados(ref_snap, n_estados, con_pase, seed=0):
    """Juega partidas con ref_snap (seat 0) vs mesa mixta, capturando estados en
    cada decisión de JUEGO del agente: (clon_motor, vacios, dama_picas_en)."""
    import random as pyr
    estados = []
    g = 0
    while len(estados) < n_estados:
        pyr.seed(1000 + seed + g); g += 1
        m = MotorCorazones(); m.nueva_partida()
        opps = {1: BotExperto(), 2: BotCastigador(), 3: BotLunatico()}
        while not m.partida_terminada(100):
            vacios = [set() for _ in range(4)]
            dama = None
            # pase
            if con_pase and m.direccion_pase() is not None:
                sel = {}
                for i in range(4):
                    if i == 0:
                        sel[0] = ref_snap.pasar(m, 0) if hasattr(ref_snap, "pasar") else list(m.jugadores[0].mano[:3])
                    else:
                        pf = getattr(opps[i], "pasar", None)
                        sel[i] = pf(m, i) if callable(pf) else list(m.jugadores[i].mano[:3])
                m.ejecutar_pase(sel)
            # juego
            while not _fin_mano(m):
                idx = m.obtener_jugador_actual()
                legales = m.obtener_jugadas_legales(idx)
                ps = m.palo_de_salida
                if idx == 0:
                    if len(legales) > 1 and len(estados) < n_estados:
                        estados.append((_clonar_motor(m),
                                        [set(v) for v in vacios], dama))
                    carta = ref_snap(m, 0, legales, obs_vec=_obs(ref_snap, m, 0, vacios, dama))
                else:
                    carta = opps[idx](m, idx, legales)
                m.jugar_carta(idx, carta)
                if ps is not None and carta.palo != ps:
                    vacios[idx].add(ps)
                if len(m.mesa) == 4:
                    cartas = m.mesa[:]
                    gan = m.resolver_baza()
                    if any(c.es_dama_de_picas for _, c in cartas):
                        dama = gan
            m.aplicar_puntuacion()
            if not m.partida_terminada(100):
                m.repartir()
                for o in opps.values():
                    r = getattr(o, "reset", None)
                    if callable(r):
                        try: r()
                        except Exception: pass
            if len(estados) >= n_estados:
                break
    return estados[:n_estados]


def _obs(snap, motor, idx, vacios, dama):
    """Obs para un SnapshotPolicy en un estado (perspectiva idx)."""
    b = snap._obs_builder
    sc = motor.puntuaciones_historicas()
    return b.construir(
        motor=motor, agente_idx=idx, vacios=vacios, puntuacion_historica=sc,
        puntos_mano_actual=[j.contar_puntos_bazas() for j in motor.jugadores],
        dama_picas_en=dama,
        moon_prob_agente=_moon_prob(motor, idx),
        moon_prob_rival=max(_moon_prob(motor, i) for i in range(4) if i != idx),
        puedo_alimentar=any(sc[j] >= 85 for j in range(4) if j != idx))


def carta_modelo(snap, motor, vacios, dama):
    legales = motor.obtener_jugadas_legales(0)
    obs = _obs(snap, motor, 0, vacios, dama)
    mask = np.zeros(52, dtype=np.float32)
    for c in legales:
        mask[c.id] = 1.0
    model = snap._get_model()
    with torch.no_grad():
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mm = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        st = [s.unsqueeze(0) for s in model.get_initial_state()] if model.get_initial_state() else []
        logits, _ = model.forward({"obs": {"obs": o, "action_mask": mm}}, st, None)
        a = int(logits.argmax(1).item())
    return a if any(c.id == a for c in legales) else legales[0].id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="modelo que genera los estados")
    p.add_argument("--candidatos", nargs="+", required=True)
    p.add_argument("--con-pase", action="store_true")
    p.add_argument("--estados", type=int, default=250)
    p.add_argument("--mundos", type=int, default=60)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    ref = cargar_policy_desde_checkpoint(args.ref)
    print(f"Generando {args.estados} estados de juego (ref {args.ref})...", flush=True)
    estados = generar_estados(ref, args.estados, args.con_pase)

    # PIMC: puntaje por carta UNA vez por estado (lo caro).
    print(f"Computando PIMC ({args.mundos} mundos, rollout mixto) por estado...", flush=True)
    pimc_por_estado = []
    for k, (m, vac, dama) in enumerate(estados):
        legales = m.obtener_jugadas_legales(0)
        vac_dict = {i: vac[i] for i in range(4)}
        scores = _puntaje_esperado_por_carta(
            m, 0, legales, vacios=vac_dict, num_mundos=args.mundos, rng=rng,
            crear_bots=lambda: _crear_bots_mixto(rng))
        pimc_por_estado.append(scores)
        if (k + 1) % max(1, len(estados)//10) == 0:
            print(f"  {k+1}/{len(estados)} estados", flush=True)

    print(f"\n{'candidato':<48}{'regret medio':>13}{'% óptimo':>10}")
    print("-" * 72)
    cands = cargar_y_evaluar(args.candidatos, estados, pimc_por_estado)
    for nombre, reg, opt in sorted(cands, key=lambda x: x[1]):
        print(f"{nombre:<48}{reg:>13.3f}{opt:>9.1f}%")
    print(f"\n>>> MENOS ERRORES: {min(cands, key=lambda x: x[1])[0]}")
    print("(regret = puntos esperados regalados por jugada; menor = mejor. "
          "% óptimo = veces que eligió una carta PIMC-óptima.)")


def cargar_y_evaluar(candidatos, estados, pimc_por_estado):
    out = []
    for ruta in candidatos:
        snap = cargar_policy_desde_checkpoint(ruta)
        regrets, optimos = [], 0
        for (m, vac, dama), scores in zip(estados, pimc_por_estado):
            cid = carta_modelo(snap, m, vac, dama)
            mejor = min(scores.values())
            reg = scores.get(cid, max(scores.values())) - mejor
            regrets.append(reg)
            if reg <= 1e-9:
                optimos += 1
        out.append((ruta, float(np.mean(regrets)), 100.0 * optimos / len(estados)))
    return out


if __name__ == "__main__":
    main()
