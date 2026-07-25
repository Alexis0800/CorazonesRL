"""
Calibración del rival "lunero oportunista" (OponenteLunar) para el pool.

Mesa fija: CAMPEÓN (asiento 0) vs [OponenteLunar, BotExperto, BotCastigador].
Mide, sobre N partidas completas:
  - lunas coronadas por el lunero /mano  (OBJETIVO: 2-3%/mano; humanos 2.5%)
  - lunas de cualquier rival /mano
  - win_rate del campeón
  - stats internas de ModoLunar (pases ofensivos, compromisos, abortos)

Uso:
    python scripts/calibrar_pool_lunero.py --partidas 200
    python scripts/calibrar_pool_lunero.py --partidas 200 \
        --umbral-pase 0.05 --umbral-juego 0.20   # config agresiva
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- bootstrap path ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import numpy as np
import torch

from src.agentes.bot_castigador import BotCastigador
from src.agentes.bot_experto import BotExperto
from src.agentes.oponente_lunar import OponenteLunar
from src.entorno.corazones_rllib import CorazonesEnvRLlib
from src.entorno.dimensiones import DIM_V12
from src.entorno.moon_model import RUTA_MOON
from src.rllib.eval_bots import _obtener_modelo
from src.rllib.utils import cargar_policy_desde_checkpoint

LUNERO_IDX = 1  # el agente ocupa el asiento 0


def calibrar(model, n_partidas: int, umbral_pase: float, umbral_juego: float,
             seed: int = 0) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    luneros: list[OponenteLunar] = []

    def factory(ai: int = 0):
        ol = OponenteLunar(umbral_pase=umbral_pase, umbral_juego=umbral_juego)
        luneros.append(ol)
        return {LUNERO_IDX: ol, 2: BotExperto(), 3: BotCastigador()}

    env = CorazonesEnvRLlib({
        "obs_dim": DIM_V12,
        "agente_idx": 0,
        "random_position": False,
        "opponent_factory": factory,
        "gamma": 0.999,
        "con_pase": True,
        "moon_dir": RUTA_MOON,
    })

    # Instrumentar el cierre de mano: quien tenga los 26 pts coronó la luna.
    manos = [0]
    lunas = [0, 0, 0, 0]
    orig_cerrar = env._cerrar_mano

    def cerrar_instrumentado():
        for i, j in enumerate(env._motor.jugadores):
            if j.contar_puntos_bazas() == 26:
                lunas[i] += 1
        manos[0] += 1
        orig_cerrar()

    env._cerrar_mano = cerrar_instrumentado

    initial = model.get_initial_state()
    es_recurrente = bool(initial)
    puestos = []
    for _ in range(n_partidas):
        obs, _ = env.reset()
        state = [s.unsqueeze(0) for s in initial] if es_recurrente else []
        done = False
        info: dict = {}
        while not done:
            with torch.no_grad():
                o = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
                logits, new_state = model.forward(
                    {"obs": {"obs": o, "action_mask": m}}, state, None)
                if es_recurrente:
                    state = new_state
                accion = int(logits.argmax(dim=1).item())
            obs, _, done, _, info = env.step(accion)
        puestos.append(info["puesto"])

    stats = {"pases_ofensivos": 0, "manos_comprometidas": 0,
             "compromisos_midmano": 0, "abortos_gate": 0, "abortos_prob": 0}
    for ol in luneros:
        if ol._modo is not None:
            for k in stats:
                stats[k] += ol._modo.stats[k]

    n_manos = manos[0]
    return {
        "partidas": n_partidas,
        "manos": n_manos,
        "lunas_lunero": lunas[LUNERO_IDX],
        "lunas_rivales": lunas[1] + lunas[2] + lunas[3],
        "lunas_campeon": lunas[0],
        "lunero_pct_mano": 100.0 * lunas[LUNERO_IDX] / max(n_manos, 1),
        "rivales_pct_mano": 100.0 * (lunas[1] + lunas[2] + lunas[3]) / max(n_manos, 1),
        "win_rate": float(np.mean([p == 1 for p in puestos])),
        "puesto_medio": float(np.mean(puestos)),
        **stats,
    }


def imprimir(r: dict, umbral_pase: float, umbral_juego: float) -> None:
    print(f"== Config: umbral_pase={umbral_pase} umbral_juego={umbral_juego} ==")
    print(f"partidas={r['partidas']}  manos={r['manos']}")
    print(f"lunas lunero:   {r['lunas_lunero']:4d}  ({r['lunero_pct_mano']:.2f} %/mano)"
          f"   [objetivo 2-3%; humanos 2.5%]")
    print(f"lunas rivales:  {r['lunas_rivales']:4d}  ({r['rivales_pct_mano']:.2f} %/mano)")
    print(f"lunas campeon:  {r['lunas_campeon']:4d}")
    print(f"win_rate campeon: {r['win_rate']:.3f}   puesto_medio: {r['puesto_medio']:.2f}")
    print(f"lunero: pases_ofensivos={r['pases_ofensivos']}  "
          f"comprometidas={r['manos_comprometidas']}  "
          f"midmano={r['compromisos_midmano']}  "
          f"abortos_gate={r['abortos_gate']}  abortos_prob={r['abortos_prob']}")
    conv = (100.0 * r["lunas_lunero"] / r["manos_comprometidas"]
            if r["manos_comprometidas"] else 0.0)
    print(f"conversion compromiso->luna: {conv:.1f} %")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partidas", type=int, default=200)
    p.add_argument("--modelo", default="models/produccion/v10c_campeon")
    p.add_argument("--umbral-pase", type=float, default=0.10)
    p.add_argument("--umbral-juego", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    policy = cargar_policy_desde_checkpoint(args.modelo)
    model = _obtener_modelo(policy, DIM_V12)

    r = calibrar(model, args.partidas, args.umbral_pase, args.umbral_juego,
                 seed=args.seed)
    imprimir(r, args.umbral_pase, args.umbral_juego)


if __name__ == "__main__":
    main()
