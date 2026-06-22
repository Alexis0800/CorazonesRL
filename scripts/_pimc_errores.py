"""Analisis PIMC: compara decisiones del modelo vs oraculo en bazas >= 8."""
from sb3_contrib import MaskablePPO
from src.v3.entorno import CorazonesEnvV3
from src.v3.red import obtener_policy_kwargs_transformer
from src.v3.observacion import ObservacionBuilderV3, DIM_V3
from src.mcts.pimc import _puntaje_esperado_por_carta
from src.dominio.motor import MotorCorazones
from src.dominio.carta import Carta
import os
import sys
import argparse
import time
import pickle
import zipfile
import io
from collections import defaultdict
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


_PALO = {0: "T", 1: "D", 2: "P", 3: "C"}


def _nc(c): return f"{c.valor}{_PALO[c.palo]}"


def _sit(motor, idx):
    if not motor.mesa:
        return "liderar"
    p = motor.palo_de_salida
    leg = motor.obtener_jugadas_legales(idx)
    return "seguir" if any(c.palo == p for c in leg) else "descartar"


def _fin(motor):
    return all(len(j.mano) == 0 for j in motor.jugadores) and len(motor.mesa) == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--manos", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Cargando {os.path.basename(args.snapshot)}...")
    path = args.snapshot.replace(".zip", "")

    # Cargar modelo con recursion limit aumentado + monkey-patch VecEnv
    import sys as _sys
    _sys.setrecursionlimit(10000)
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
    _orig_getattr = VecEnv.__getattr__

    def _safe_getattr(self, name):
        if name == "class_attributes":
            return {}
        try:
            return _orig_getattr(self, name)
        except RecursionError:
            return {}
    VecEnv.__getattr__ = _safe_getattr

    mdl = MaskablePPO.load(path, device="cpu")
    VecEnv.__getattr__ = _orig_getattr  # restaurar

    # Cerrar env problemático y setear a None
    try:
        mdl.env.close()
    except Exception:
        pass
    mdl.env = None

    # VecNormalize
    om, ov = None, None
    vp = path + "_vecnorm.pkl"
    if os.path.exists(vp):
        with open(vp, "rb") as f:
            vn = pickle.load(f)
        r = vn.get("obs_rms", None)
        if r is not None:
            om, ov = r.mean, r.var

    def pred(obs, mask):
        x = np.asarray(obs, dtype=np.float32)
        if om is not None:
            x = np.clip((x - om) / np.sqrt(ov + 1e-8), -10, 10)
        x = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            f = mdl.policy.extract_features(x)
            if (mdl.policy.mlp_extractor is not None
                    and mdl.policy.mlp_extractor.policy_net is not None):
                f = mdl.policy.mlp_extractor.policy_net(f)
            logits = mdl.policy.action_net(f)
        m = logits.squeeze(0).clone()
        m[~torch.from_numpy(np.asarray(mask, dtype=bool))] = -1e9
        return int(torch.argmax(m).numpy())

    def jugar_ops(motor, agente, builder):
        while not _fin(motor):
            idx = motor.obtener_jugador_actual()
            if idx == agente:
                break
            leg = motor.obtener_jugadas_legales(idx)
            if not leg:
                break
            o = builder.construir_desde_motor(motor, idx)
            mk = np.zeros(52, dtype=bool)
            for c in leg:
                mk[c.id] = True
            motor.jugar_carta(idx, Carta._TODAS[pred(o, mk)])
            if len(motor.mesa) == 4:
                motor.resolver_baza()

    builder = ObservacionBuilderV3(dim=DIM_V3)
    rng = np.random.default_rng(args.seed)

    total, ok, err = 0, 0, 0
    coste_total = 0.0
    e_sit, o_sit = defaultdict(int), defaultdict(int)
    e_baz, o_baz = defaultdict(int), defaultdict(int)
    top = []

    print(f"Analizando {args.manos} manos (PIMC 50 mundos)...")
    t0 = time.time()

    for mi in range(args.manos):
        motor = MotorCorazones()
        motor.repartir()
        ag = mi % 4
        jugar_ops(motor, ag, builder)

        while not _fin(motor):
            idx = motor.obtener_jugador_actual()
            if idx != ag:
                jugar_ops(motor, ag, builder)
                if _fin(motor):
                    break
                idx = motor.obtener_jugador_actual()

            leg = motor.obtener_jugadas_legales(idx)
            if not leg:
                break
            o = builder.construir_desde_motor(motor, idx)
            mk = np.zeros(52, dtype=bool)
            for c in leg:
                mk[c.id] = True
            elegida = Carta._TODAS[pred(o, mk)]

            if motor.numero_baza >= 8 and len(leg) >= 2:
                total += 1
                sc = _puntaje_esperado_por_carta(
                    motor, ag, leg, num_mundos=50, rng=rng)
                opt = min(leg, key=lambda c: sc[c.id])
                sit = _sit(motor, ag)
                if elegida.id == opt.id:
                    ok += 1
                    o_sit[sit] += 1
                    o_baz[motor.numero_baza] += 1
                else:
                    err += 1
                    cs = sc[elegida.id] - sc[opt.id]
                    coste_total += cs
                    e_sit[sit] += 1
                    e_baz[motor.numero_baza] += 1
                    top.append((cs, _nc(elegida), _nc(
                        opt), sit, motor.numero_baza))

            motor.jugar_carta(idx, elegida)
            if len(motor.mesa) == 4:
                motor.resolver_baza()

        if (mi + 1) % 10 == 0:
            wr = ok / max(total, 1)
            print(f"  Mano {mi+1:>3}/{args.manos} | Dec: {total} | "
                  f"Ok: {ok}/{total} ({wr:.0%}) | Coste: {coste_total:.1f}")

    elapsed = time.time() - t0
    tasa = ok / max(total, 1)
    cmed = coste_total / max(err, 1)

    print(f"\n{'='*60}")
    print(f"  RESULTADOS ({args.manos} manos en {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Decisiones (baza>=8): {total}")
    print(f"  Aciertos PIMC:  {ok}/{total} ({tasa:.0%})")
    print(f"  Errores:        {err}/{total} ({1-tasa:.0%})")
    print(f"  Coste total:    {coste_total:.2f} pts")
    print(f"  Coste/error:    {cmed:.2f} pts")

    print(f"\n  POR SITUACION:")
    for sit in ["liderar", "seguir", "descartar"]:
        a = o_sit.get(sit, 0)
        e = e_sit.get(sit, 0)
        t = a + e
        print(f"    {sit:<12} ok={a:>3}  err={e:>3}  tasa={a/max(t, 1):.0%}")

    print(f"\n  POR BAZA:")
    for bz in sorted(set(o_baz) | set(e_baz)):
        a = o_baz.get(bz, 0)
        e = e_baz.get(bz, 0)
        t = a + e
        print(
            f"    Baza {bz:<3}  ok={a:>3}  err={e:>3}  tasa={a/max(t, 1):.0%}")

    print(f"\n  TOP 10 ERRORES MAS COSTOSOS:")
    for cs, el, opt, sit, bz in sorted(top, reverse=True)[:10]:
        print(f"    +{cs:>5.2f}pts  {el:>5} -> {opt:<5}  {sit:<12} baza {bz}")

    print(f"\n{'='*60}")
    print(
        f"  DIAGNOSTICO: tasa PIMC = {tasa:.0%}, coste/error = {cmed:.2f} pts")
    if tasa >= 0.70:
        print(f"  Buena alineacion con el oraculo PIMC.")
    elif tasa >= 0.55:
        print(f"  Alineacion moderada. BC fine-tuning puede cerrar la brecha.")
    else:
        print(f"  Baja alineacion. Recomendado mas BC fine-tuning + PPO.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
