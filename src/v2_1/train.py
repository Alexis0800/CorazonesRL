"""
Entrenamiento autónomo v2_1 — Single-hand, enhanced tactical rewards.

Extiende v2_ronda con 4 nuevas señales de recompensa por ronda:
  1. Q♠ dump: +12
  2. Moon block: +15
  3. Early safe burn: +1.5
  4. Liability hold: -5

Uso:
    # Con BC checkpoint
    python -m src.v2_1.train --resume models/v2_bc_v2/snapshots/snapshot_0000000000 --total-steps 5000000 --output-dir models/v2_1
    # Desde cero (sin BC)
    python -m src.v2_1.train --total-steps 5000000 --output-dir models/v2_1 --no-bc
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
DIRECTORIO_LOGS: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    "logs",
)

SNAPSHOT_EVERY: int = 100_000
EVAL_EVERY: int = 5
EVAL_HANDS: int = 100
EVAL_EXPERTO_HANDS: int = 100
MAX_SNAPSHOTS_POOL: int = 50
MIN_SNAPSHOT_STEPS: int = 500_000

HP_DEFAULT: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extraer_paso_de_ruta(ruta: str) -> int:
    import re
    match = re.search(r'snapshot_(\d+)', ruta)
    return int(match.group(1)) if match else 0


def _listar_snapshots(directorio: str) -> List[str]:
    if not os.path.isdir(directorio):
        return []
    zips = [os.path.join(directorio, f) for f in os.listdir(directorio)
            if f.startswith("snapshot_") and f.endswith(".zip")]
    zips.sort(key=_extraer_paso_de_ruta)
    return zips


# ------------------------------------------------------------------
# Self-play
# ------------------------------------------------------------------

def crear_entorno_self_play_v21(
    directorio_snapshots: str,
    agente_idx: int = 0,
    seed: int = 42,
    prob_bot: float = 0.7,
    prob_experto: float = 0.15,
    obs_dim: int = 220,
):
    """Crea un CorazonesEnvV21 con oponentes mixtos."""
    from src.v2_1.entorno import CorazonesEnvV21
    from src.agentes.heuristicos import bot_evasivo, bot_conservador, bot_agresivo
    from src.agentes.bot_experto import BotExperto
    from src.agentes.politica_rl import PoliticaSB3

    random.seed(seed)
    np.random.seed(seed)

    snapshots = _listar_snapshots(directorio_snapshots)
    snapshots = [s for s in snapshots
                 if _extraer_paso_de_ruta(s) >= MIN_SNAPSHOT_STEPS]
    if len(snapshots) > MAX_SNAPSHOTS_POOL:
        snapshots = snapshots[-MAX_SNAPSHOTS_POOL:]

    bots_pool = [bot_evasivo, bot_conservador, bot_agresivo]
    random.shuffle(bots_pool)
    politicas: Dict[int, object] = {}

    for i in range(4):
        if i == agente_idx:
            continue
        r = random.random()
        if r < prob_bot:
            politicas[i] = bots_pool[i % len(bots_pool)]
        elif r < prob_bot + prob_experto:
            politicas[i] = BotExperto()
        elif snapshots:
            snap = random.choice(snapshots)
            try:
                from sb3_contrib import MaskablePPO
                modelo_oponente = MaskablePPO.load(snap)
                vecnorm_path = snap.replace(".zip", "_vecnorm.pkl")
                politicas[i] = PoliticaSB3(modelo_oponente, i, vecnorm_path)
            except Exception:
                politicas[i] = bots_pool[i % len(bots_pool)]
        else:
            politicas[i] = bots_pool[i % len(bots_pool)]

    return CorazonesEnvV21(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
        obs_dim=obs_dim,
    )


# ------------------------------------------------------------------
# Normalización
# ------------------------------------------------------------------

def _cargar_normalizador(venv_stats_path: str):
    import pickle
    with open(venv_stats_path, "rb") as f:
        vn = pickle.load(f)
    return vn.obs_rms.mean, vn.obs_rms.var


def _normalizar_obs(obs_raw: np.ndarray, obs_mean, obs_var) -> np.ndarray:
    return np.clip(
        (obs_raw - obs_mean) / (np.sqrt(obs_var) + 1e-8),
        -10.0, 10.0,
    )


# ------------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------------

def evaluar_score_promedio(
    modelo,
    venv_stats_path: Optional[str] = None,
    num_hands: int = EVAL_HANDS,
) -> Dict[str, float]:
    """Evalúa score promedio contra bots heurísticos."""
    from src.v2_1.entorno import CorazonesEnvV21
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo

    obs_mean = obs_var = None
    if venv_stats_path and os.path.exists(venv_stats_path):
        obs_mean, obs_var = _cargar_normalizador(venv_stats_path)

    scores: List[int] = []
    posiciones: List[int] = []

    for seed in range(num_hands):
        bots = [bot_conservador, bot_agresivo, bot_evasivo]
        random.shuffle(bots)
        env = CorazonesEnvV21(
            agente_idx=0,
            politicas_oponentes={1: bots[0], 2: bots[1], 3: bots[2]},
        )
        obs_raw, _ = env.reset(seed=seed)
        obs = _normalizar_obs(
            obs_raw, obs_mean, obs_var) if obs_mean is not None else obs_raw

        terminated = False
        while not terminated:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True,
            )
            obs_raw, _, terminated, truncated, _ = env.step(int(action))
            obs = _normalizar_obs(
                obs_raw, obs_mean, obs_var) if obs_mean is not None else obs_raw

        mi_score = env._puntos_mano_actual[0]
        rivales = [env._puntos_mano_actual[i] for i in range(1, 4)]
        scores.append(mi_score)
        todas = [mi_score] + rivales
        ranking = sorted(range(4), key=lambda i: todas[i])
        posiciones.append(ranking.index(0))

    arr = np.array(scores, dtype=np.float64)
    return {
        "total_hands": num_hands,
        "score_promedio": float(np.mean(arr)),
        "score_mediana": float(np.median(arr)),
        "score_std": float(np.std(arr)),
        "pct_mejor": sum(1 for p in posiciones if p == 0) / num_hands,
        "pct_top2": sum(1 for p in posiciones if p in (0, 1)) / num_hands,
        "pct_peor": sum(1 for p in posiciones if p == 3) / num_hands,
        "pct_cero": sum(1 for s in scores if s == 0) / num_hands,
    }


def evaluar_vs_experto_v21(
    modelo,
    venv_stats_path: Optional[str] = None,
    num_hands: int = EVAL_EXPERTO_HANDS,
) -> Dict[str, float]:
    """Evalúa contra 3x BotExperto."""
    from src.v2_1.entorno import CorazonesEnvV21
    from src.agentes.bot_experto import BotExperto

    obs_mean = obs_var = None
    if venv_stats_path and os.path.exists(venv_stats_path):
        obs_mean, obs_var = _cargar_normalizador(venv_stats_path)

    scores: List[int] = []
    posiciones: List[int] = []

    for seed in range(num_hands):
        env = CorazonesEnvV21(
            agente_idx=0,
            politicas_oponentes={
                1: BotExperto(), 2: BotExperto(), 3: BotExperto(),
            },
        )
        obs_raw, _ = env.reset(seed=seed)
        obs = _normalizar_obs(
            obs_raw, obs_mean, obs_var) if obs_mean is not None else obs_raw

        terminated = False
        while not terminated:
            mask = env.action_masks()
            action, _ = modelo.predict(
                obs, action_masks=mask, deterministic=True,
            )
            obs_raw, _, terminated, truncated, _ = env.step(int(action))
            obs = _normalizar_obs(
                obs_raw, obs_mean, obs_var) if obs_mean is not None else obs_raw

        mi_score = env._puntos_mano_actual[0]
        rivales = [env._puntos_mano_actual[i] for i in range(1, 4)]
        scores.append(mi_score)
        todas = [mi_score] + rivales
        ranking = sorted(range(4), key=lambda i: todas[i])
        posiciones.append(ranking.index(0))

    arr = np.array(scores, dtype=np.float64)
    return {
        "total_hands": num_hands,
        "score_promedio": float(np.mean(arr)),
        "score_mediana": float(np.median(arr)),
        "score_std": float(np.std(arr)),
        "pct_mejor": sum(1 for p in posiciones if p == 0) / num_hands,
        "pct_top2": sum(1 for p in posiciones if p in (0, 1)) / num_hands,
        "pct_peor": sum(1 for p in posiciones if p == 3) / num_hands,
        "pct_cero": sum(1 for s in scores if s == 0) / num_hands,
    }


# ------------------------------------------------------------------
# Diagnóstico
# ------------------------------------------------------------------

def _capturar_diagnosticos(modelo) -> Dict[str, float]:
    try:
        raw = modelo.logger.name_to_value
    except (AttributeError, TypeError):
        return {}
    if not raw:
        return {}
    return {
        key.replace("train/", ""): round(float(value), 6)
        for key, value in raw.items()
        if key.startswith("train/")
    }


def _alertas_diagnostico(diag: Dict[str, float]) -> list:
    alertas = []
    entropy = diag.get("entropy_loss")
    if entropy is not None and entropy > -0.15:
        alertas.append(f"COLAPSO DE ENTROPIA: entropy_loss={entropy:.4f}")
    kl = diag.get("approx_kl")
    if kl is not None and kl > 0.05:
        alertas.append(f"KL DIVERGENTE: approx_kl={kl:.4f}")
    clip = diag.get("clip_fraction")
    if clip is not None and clip > 0.5:
        alertas.append(f"CLIP SATURADO: clip_fraction={clip:.4f}")
    v_loss = diag.get("value_loss")
    if v_loss is not None and v_loss > 3.0:
        alertas.append(f"VALUE LOSS ELEVADO: value_loss={v_loss:.4f}")
    ev = diag.get("explained_variance")
    if ev is not None and ev < 0.0:
        alertas.append(
            f"EXPLAINED VARIANCE NEGATIVO: explained_variance={ev:.4f}")
    return alertas


# ------------------------------------------------------------------
# Entrenamiento principal
# ------------------------------------------------------------------

def entrenar_v21(
    total_steps: int = 5_000_000,
    output_dir: str = "models/v2_1",
    resume_from: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu",
    eval_experto: bool = True,
) -> int:
    """Entrenamiento autónomo v2_1: enhanced tactical rewards."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dir_snapshots = os.path.join(output_dir, "snapshots")
    dir_vecnorm = os.path.join(output_dir, "vecnorm")
    best_dir = os.path.join(output_dir, "best")
    eval_log_path = os.path.join(dir_snapshots, "eval_log.jsonl")
    etiqueta = os.path.basename(output_dir)

    os.makedirs(dir_snapshots, exist_ok=True)
    os.makedirs(dir_vecnorm, exist_ok=True)

    if resume_from:
        if not resume_from.endswith(".zip"):
            resume_from += ".zip"
        ruta_resume = resume_from if os.path.isabs(resume_from) else os.path.join(
            os.getcwd(), resume_from)
        if not os.path.exists(ruta_resume):
            print(f"ERROR: No se encuentra {ruta_resume}")
            sys.exit(1)

        paso_actual = _extraer_paso_de_ruta(ruta_resume)
        print("=" * 60)
        print(f"REANUDANDO {etiqueta} desde paso {paso_actual:,}")
        print("=" * 60)

        modelo = MaskablePPO.load(ruta_resume, device=device)
        vn_path = ruta_resume.replace(".zip", "_vecnorm.pkl")

        def _make_env():
            return crear_entorno_self_play_v21(
                dir_snapshots, seed=seed + paso_actual)
        if os.path.exists(vn_path):
            venv = VecNormalize.load(vn_path, DummyVecEnv([_make_env]))
        else:
            venv = DummyVecEnv([_make_env])
            venv = VecNormalize(
                venv, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
            )
        modelo.set_env(venv)
        snapshot_count = paso_actual // SNAPSHOT_EVERY
    else:
        for d in [dir_snapshots, dir_vecnorm]:
            if os.path.isdir(d):
                shutil.rmtree(d)
        os.makedirs(dir_snapshots, exist_ok=True)
        os.makedirs(dir_vecnorm, exist_ok=True)
        os.makedirs(DIRECTORIO_LOGS, exist_ok=True)

        print("=" * 60)
        print(f"ENTRENAMIENTO v2_1: {etiqueta}")
        print("=" * 60)
        print(f"  Total steps: {total_steps:,}")
        print(f"  Recompensa: score-based + tactical (Q♠ dump, moon block, etc)")
        print(f"  Episode: 1 mano (~52 steps)")
        print(f"  Device: {device}")
        print("-" * 60)

        np.random.seed(seed)
        random.seed(seed)

        def _make_env():
            return crear_entorno_self_play_v21(dir_snapshots, seed=seed)
        env_base = _make_env()
        env_base.reset(seed=seed)

        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(
            venv, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
        )

        modelo = MaskablePPO(
            "MlpPolicy", venv,
            learning_rate=HP_DEFAULT["learning_rate"],
            n_steps=HP_DEFAULT["n_steps"],
            batch_size=HP_DEFAULT["batch_size"],
            n_epochs=HP_DEFAULT["n_epochs"],
            gamma=HP_DEFAULT["gamma"],
            gae_lambda=HP_DEFAULT["gae_lambda"],
            clip_range=HP_DEFAULT["clip_range"],
            ent_coef=HP_DEFAULT["ent_coef"],
            vf_coef=HP_DEFAULT["vf_coef"],
            max_grad_norm=HP_DEFAULT["max_grad_norm"],
            target_kl=HP_DEFAULT["target_kl"],
            verbose=1,
            device=device,
            tensorboard_log=DIRECTORIO_LOGS,
        )

        paso_actual = 0
        snapshot_count = 0

    # --- Value head warmup (if resuming from BC) ---
    VALUE_WARMUP_STEPS: int = 200_000
    if paso_actual == 0 and resume_from:
        print(
            f"\n  [Warmup] Calentando value head ({VALUE_WARMUP_STEPS:,} steps)...")
        for param in modelo.policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = False
        for param in modelo.policy.action_net.parameters():
            param.requires_grad = False
        for param in modelo.policy.features_extractor.parameters():
            param.requires_grad = False

        modelo.vf_coef = 1.0
        modelo.ent_coef = 0.0
        modelo.learn(total_timesteps=VALUE_WARMUP_STEPS,
                     reset_num_timesteps=False, progress_bar=True)

        for param in modelo.policy.parameters():
            param.requires_grad = True
        modelo.vf_coef = HP_DEFAULT["vf_coef"]
        modelo.ent_coef = HP_DEFAULT["ent_coef"]
        modelo.learning_rate = HP_DEFAULT["learning_rate"]
        if hasattr(modelo.policy, "optimizer") and modelo.policy.optimizer is not None:
            for param_group in modelo.policy.optimizer.param_groups:
                param_group["lr"] = HP_DEFAULT["learning_rate"]
        paso_actual += VALUE_WARMUP_STEPS
        print(f"  [Warmup] Value head calentado. Paso actual: {paso_actual:,}")
        venv.save(os.path.join(dir_vecnorm, f"{etiqueta}_vecnorm.pkl"))
        nombre = f"snapshot_{paso_actual:010d}"
        ruta = os.path.join(dir_snapshots, nombre)
        modelo.save(ruta)
        venv.save(ruta + "_vecnorm.pkl")

    vecnorm_path = os.path.join(dir_vecnorm, f"{etiqueta}_vecnorm.pkl")

    # --- Bucle principal ---
    while paso_actual < total_steps:
        bloque = min(SNAPSHOT_EVERY, total_steps - paso_actual)

        print(f"\n--- Bloque {snapshot_count + 1} | Paso {paso_actual:,} "
              f"→ {paso_actual + bloque:,} ---")

        modelo.learn(
            total_timesteps=bloque,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        paso_actual += bloque
        snapshot_count += 1

        # Diagnóstico
        diag = _capturar_diagnosticos(modelo)
        alertas = _alertas_diagnostico(diag)
        if alertas:
            for a in alertas:
                print(f"  {a}")
            with open(eval_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "paso": paso_actual,
                    "tipo": "alerta",
                    "alertas": alertas,
                }) + "\n")
        if diag:
            with open(eval_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "paso": paso_actual,
                    "tipo": "diagnostico",
                    **diag,
                }) + "\n")
            ent = diag.get("entropy_loss", "?")
            kl = diag.get("approx_kl", "?")
            vf = diag.get("value_loss", "?")
            print(f"  [Diag] entropy={ent} | kl={kl} | v_loss={vf}")

        # Guardar snapshot
        venv.save(vecnorm_path)
        nombre = f"snapshot_{paso_actual:010d}"
        ruta = os.path.join(dir_snapshots, nombre)
        modelo.save(ruta)
        venv.save(ruta + "_vecnorm.pkl")
        print(f"  [Snapshot] {ruta}.zip")

        # Pruning
        snaps = _listar_snapshots(dir_snapshots)
        if len(snaps) > MAX_SNAPSHOTS_POOL:
            a_eliminar = snaps[:len(snaps) - MAX_SNAPSHOTS_POOL]
            for s in a_eliminar:
                os.remove(s)
                vec = s.replace(".zip", "_vecnorm.pkl")
                if os.path.exists(vec):
                    os.remove(vec)

        # Evaluación
        if EVAL_EVERY > 0 and snapshot_count % EVAL_EVERY == 0:
            print(f"\n  [Eval] Score vs bots ({EVAL_HANDS} manos)...")
            try:
                res = evaluar_score_promedio(
                    modelo, venv_stats_path=vecnorm_path)
                sc = res["score_promedio"]
                print(f"  [Eval] Score avg={sc:.1f}±{res['score_std']:.1f} | "
                      f"Mejor={res['pct_mejor']:.1%} | Top2={res['pct_top2']:.1%} | Cero={res['pct_cero']:.1%}")
                with open(eval_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "paso": paso_actual,
                        "snapshot": nombre,
                        "timestamp": datetime.now().isoformat(),
                        **res,
                    }) + "\n")
            except Exception as e:
                print(f"  [Eval] Error: {e}")

            if eval_experto:
                print(
                    f"  [Eval] Score vs 3x BotExperto ({EVAL_EXPERTO_HANDS} manos)...")
                try:
                    res_exp = evaluar_vs_experto_v21(
                        modelo, venv_stats_path=vecnorm_path)
                    sc_exp = res_exp["score_promedio"]
                    print(f"  [Eval] vsExperto: Score avg={sc_exp:.1f}±{res_exp['score_std']:.1f} | "
                          f"Mejor={res_exp['pct_mejor']:.1%} | Top2={res_exp['pct_top2']:.1%} | Cero={res_exp['pct_cero']:.1%}")
                    with open(eval_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "paso": paso_actual,
                            "snapshot": nombre,
                            "timestamp": datetime.now().isoformat(),
                            "vs": "experto",
                            **res_exp,
                        }) + "\n")
                except Exception as e:
                    print(f"  [Eval] Error: {e}")

    # --- Best snapshot ---
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETO")
    print("=" * 60)

    # Guardar último snapshot como best si no se hizo
    os.makedirs(best_dir, exist_ok=True)
    ultimo_nombre = f"snapshot_{paso_actual:010d}"
    if os.path.exists(os.path.join(dir_snapshots, ultimo_nombre + ".zip")):
        shutil.copy2(
            os.path.join(dir_snapshots, ultimo_nombre + ".zip"),
            os.path.join(best_dir, ultimo_nombre + ".zip"))
        shutil.copy2(
            os.path.join(dir_snapshots, ultimo_nombre + "_vecnorm.pkl"),
            os.path.join(best_dir, ultimo_nombre + "_vecnorm.pkl"))

    return paso_actual


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento v2_1 — Enhanced tactical rewards",
    )
    parser.add_argument("--total-steps", type=int, default=5_000_000,
                        help="Pasos totales de entrenamiento.")
    parser.add_argument("--output-dir", type=str, default="models/v2_1",
                        help="Directorio de salida.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Snapshot .zip para reanudar.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo (cpu, cuda, dml).")
    parser.add_argument("--no-eval-experto", action="store_true",
                        help="Desactivar evaluación vs BotExperto.")

    args = parser.parse_args()

    paso_final = entrenar_v21(
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        resume_from=args.resume,
        seed=args.seed,
        device=args.device,
        eval_experto=not args.no_eval_experto,
    )

    print(f"\nEntrenamiento finalizado en paso {paso_final:,}")


if __name__ == "__main__":
    main()
