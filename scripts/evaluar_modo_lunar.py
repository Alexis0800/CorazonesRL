"""
Evaluación A/B del MODO LUNAR (src/agentes/modo_lunar.py): campeón puro vs
campeón + ofensiva de luna compuesta, contra 3 clones humanos (el mejor proxy
disponible). Ver docs/auditoria_moon_2026-07-20.md §Auditoría de OFENSIVA.

Métricas: win_rate / top2 / puesto_medio, y de luna: manos comprometidas,
lunas logradas (conversión), abortos por gate, y puntos comidos en manos
comprometidas SIN luna (el coste de los intentos fallidos).

⚠ Gate de evaluación conocido: el clon humano casi nunca enfrentó lunas
nuestras en su dataset → su defensa anti-luna es optimista. Un resultado
positivo aquí se confirma en partidas reales del bridge antes de producción.

Uso:
    python scripts/evaluar_modo_lunar.py --modelo models/produccion/v10c_campeon \
        --humano models/humano_bc/pesos.npz --partidas 300 \
        --umbral-pase 0.10 --umbral-juego 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --- bootstrap path ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---


def _jugar_partidas(model, env, n_partidas: int, modo=None) -> tuple[list, dict]:
    """Loop de eval idéntico para ambos brazos; `modo` intercepta acciones.

    Devuelve (infos_terminales, stats_luna). stats_luna incluye la conversión
    por mano comprometida y el coste de los intentos fallidos, medidos desde el
    delta del marcador (verdad del motor, sin depender de internals de la mano).
    """
    import torch
    from src.dominio.carta import Carta

    resultados = []
    luna = {"lunas_logradas": 0, "pts_comprometidas_fallidas": [], "manos_totales": 0}

    for i in range(n_partidas):
        obs, _ = env.reset(seed=i)  # mismas semillas en ambos brazos (parea repartos)
        if modo is not None:
            modo.nueva_partida()
        pase_cola: list[int] = []
        prev_scores = list(env._motor.puntuaciones_historicas())
        prev_manos = 0
        comprometida_en_curso = False
        done = False
        info: dict = {}

        while not done:
            accion = None
            if modo is not None:
                if env._fase_pase:
                    if not env._pase_seleccion and not pase_cola:
                        cartas = modo.elegir_pase(env._motor, env._agente_idx)
                        if cartas is not None:
                            pase_cola = [c.id for c in cartas]
                    if pase_cola:
                        accion = pase_cola.pop(0)
                else:
                    legales = env._motor.obtener_jugadas_legales(env._agente_idx)
                    carta = modo.elegir_jugada(env._motor, env._agente_idx, legales)
                    if carta is not None:
                        accion = carta.id
                    comprometida_en_curso = comprometida_en_curso or modo.comprometida

            if accion is None:
                with torch.no_grad():
                    o = torch.as_tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                    m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
                    logits, _ = model.forward({"obs": {"obs": o, "action_mask": m}}, [], None)
                    accion = int(logits.argmax(dim=1).item())

            obs, _, done, _, info = env.step(accion)

            # ¿Cerró una mano? Atribuir el resultado a la mano comprometida.
            if env._manos_jugadas > prev_manos or done:
                scores = list(env._motor.puntuaciones_historicas())
                delta = [s - p for s, p in zip(scores, prev_scores)]
                luna["manos_totales"] += 1
                if comprometida_en_curso:
                    if delta[env._agente_idx] == 0 and sum(delta) == 78:
                        luna["lunas_logradas"] += 1
                    else:
                        luna["pts_comprometidas_fallidas"].append(delta[env._agente_idx])
                prev_scores = scores
                prev_manos = env._manos_jugadas
                comprometida_en_curso = False

        resultados.append(info)

    if modo is not None:
        luna.update(modo.stats)
    return resultados, luna


def main() -> None:
    from src.rllib.utils import cargar_policy_desde_checkpoint
    from src.rllib.opponent_pool import SnapshotPolicy
    from src.rllib.eval_bots import _obtener_modelo, _agregar
    from src.entorno.corazones_rllib import CorazonesEnvRLlib
    from src.entorno.dimensiones import DIM_V12, con_pase_de_obs
    from src.entorno.moon_model import RUTA_MOON, EstimadorMoonProb
    from src.agentes.modo_lunar import ModoLunar

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True)
    p.add_argument("--humano", default="models/humano_bc/pesos.npz")
    p.add_argument("--partidas", type=int, default=300)
    p.add_argument("--obs-dim", type=int, default=DIM_V12)
    p.add_argument("--umbral-pase", type=float, default=0.10)
    p.add_argument("--umbral-juego", type=float, default=0.15)
    p.add_argument("--moon-dir", default=RUTA_MOON)
    p.add_argument("--sin-baseline", action="store_true",
                   help="Solo el brazo lunar (la línea base del campeón ya se conoce)")
    args = p.parse_args()

    policy = cargar_policy_desde_checkpoint(args.modelo)
    model = _obtener_modelo(policy, args.obs_dim)
    d = np.load(args.humano)
    pesos = {k: d[k] for k in d.files}

    def factory(ai=0):
        return {i: SnapshotPolicy.from_weights(pesos, obs_dim=args.obs_dim, temperatura=1.0)
                for i in range(4) if i != ai}

    def make_env():
        return CorazonesEnvRLlib({
            "obs_dim": args.obs_dim, "agente_idx": 0, "random_position": False,
            "opponent_factory": factory, "gamma": 0.999,
            "con_pase": con_pase_de_obs(args.obs_dim), "moon_dir": args.moon_dir,
        })

    salida = {"config": vars(args)}

    if not args.sin_baseline:
        res, _ = _jugar_partidas(model, make_env(), args.partidas)
        agg = _agregar(res)
        salida["baseline"] = agg
        print(f"\n=== BASELINE campeón puro ({args.partidas} partidas vs 3 clones) ===")
        print(f"  win_rate: {agg['win']:.3f}  top2: {agg['top2']:.3f}  "
              f"puesto: {agg['puesto']:.3f}  moon/partida: {agg['moon']:.3f}")

    modo = ModoLunar(EstimadorMoonProb(args.moon_dir),
                     umbral_pase=args.umbral_pase, umbral_juego=args.umbral_juego)
    res, luna = _jugar_partidas(model, make_env(), args.partidas, modo=modo)
    agg = _agregar(res)
    salida["lunar"] = agg

    fallidas = luna.pop("pts_comprometidas_fallidas")
    luna["conversion"] = (luna["lunas_logradas"] / luna["manos_comprometidas"]
                          if luna["manos_comprometidas"] else 0.0)
    luna["pts_medios_intento_fallido"] = float(np.mean(fallidas)) if fallidas else 0.0
    salida["luna"] = luna

    print(f"\n=== MODO LUNAR (umbral pase {args.umbral_pase} / juego {args.umbral_juego}) ===")
    print(f"  win_rate: {agg['win']:.3f}  top2: {agg['top2']:.3f}  "
          f"puesto: {agg['puesto']:.3f}  moon/partida: {agg['moon']:.3f}")
    print(f"  manos: {luna['manos_totales']}  pases ofensivos: {luna['pases_ofensivos']}  "
          f"comprometidas: {luna['manos_comprometidas']}")
    print(f"  lunas logradas: {luna['lunas_logradas']} "
          f"(conversión {luna['conversion']:.1%})  abortos gate: {luna['abortos_gate']}")
    print(f"  pts medios en intento fallido: {luna['pts_medios_intento_fallido']:.2f}")
    print("\n" + json.dumps(salida, indent=2, default=str))


if __name__ == "__main__":
    main()
