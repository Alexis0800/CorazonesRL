"""
Entrenamiento autonomo v2_ronda — Single-hand, score-based rewards.

Una mano = un episodio. Recompensa basada exclusivamente en score propio.
Sin distancia, sin ruido de rivales, sin contexto de partida.

Arquitectura independiente bajo src/v2_ronda/. Comparte dominio y agentes con v1.

Uso:
    # Desde cero
    python -m src.v2_ronda.train --total-steps 5000000 --output-dir models/v2_score

    # Reanudar desde snapshot
    python -m src.v2_ronda.train --resume models/v2_score/snapshots/snapshot_0001000000 --total-steps 10000000 --output-dir models/v2_score
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random

# Asegurar que el proyecto esta en el path
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
EVAL_EVERY: int = 5          # evaluar cada N snapshots
EVAL_HANDS: int = 100        # manos por evaluacion vs bots
# manos vs BotExperto (misma cantidad para significancia estadistica)
EVAL_EXPERTO_HANDS: int = 100
MAX_SNAPSHOTS_POOL: int = 50
MIN_SNAPSHOT_STEPS: int = 500_000

# Hiperparametros (v2, calibrados para BC→RL)
HP_DEFAULT: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 512,          # rollouts mas largos para mejor GAE
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,       # mas tolerancia post-BC
}


# ------------------------------------------------------------------
# Helpers: snapshots
# ------------------------------------------------------------------

def _extraer_paso_de_ruta(ruta: str) -> int:
    """Extrae el paso de entrenamiento del nombre del snapshot."""
    import re
    match = re.search(r'snapshot_(\d+)', ruta)
    return int(match.group(1)) if match else 0


def _listar_snapshots(directorio: str) -> List[str]:
    """Lista snapshots .zip ordenados por paso."""
    if not os.path.isdir(directorio):
        return []
    zips = [os.path.join(directorio, f) for f in os.listdir(directorio)
            if f.startswith("snapshot_") and f.endswith(".zip")]
    zips.sort(key=_extraer_paso_de_ruta)
    return zips


# ------------------------------------------------------------------
# Self-play: crear entorno con oponentes mixtos
# ------------------------------------------------------------------

def crear_entorno_self_play_v2(
    directorio_snapshots: str,
    agente_idx: int = 0,
    seed: int = 42,
    prob_bot: float = 0.7,
    prob_experto: float = 0.15,
    obs_dim: int = 220,
):
    """Crea un CorazonesEnvSingleHand con oponentes mixtos.

    Mezcla:
      - Con prob_bot: bot heuristico aleatorio
      - Con prob_experto: BotExperto
      - Resto: snapshot historico (si hay disponibles)

    Args:
        directorio_snapshots: Directorio con snapshots .zip para self-play.
        agente_idx: Indice del agente controlado por RL.
        seed: Semilla aleatoria.
        prob_bot: Probabilidad de bot heuristico.
        prob_experto: Probabilidad de BotExperto.
        obs_dim: Dimension del vector de observacion.

    Returns:
        CorazonesEnvSingleHand configurado.
    """
    from src.v2_ronda.entorno import CorazonesEnvSingleHand
    from src.agentes.heuristicos import bot_evasivo, bot_conservador, bot_agresivo
    from src.agentes.bot_experto import BotExperto
    from src.agentes.politica_rl import PoliticaSB3

    random.seed(seed)
    np.random.seed(seed)

    snapshots = _listar_snapshots(directorio_snapshots)
    # Filtrar snapshots muy recientes
    snapshots = [s for s in snapshots
                 if _extraer_paso_de_ruta(s) >= MIN_SNAPSHOT_STEPS]
    # Limitar pool
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
            # Bot heuristico
            politicas[i] = bots_pool[i % len(bots_pool)]
        elif r < prob_bot + prob_experto:
            # BotExperto
            politicas[i] = BotExperto()
        elif snapshots:
            # Snapshot historico
            snap = random.choice(snapshots)
            try:
                from sb3_contrib import MaskablePPO
                modelo_oponente = MaskablePPO.load(snap)
                vecnorm_path = snap.replace(".zip", "_vecnorm.pkl")
                politicas[i] = PoliticaSB3(modelo_oponente, i, vecnorm_path)
            except Exception:
                # Fallback: bot heuristico
                politicas[i] = bots_pool[i % len(bots_pool)]
        else:
            # Sin snapshots: bot
            politicas[i] = bots_pool[i % len(bots_pool)]

    return CorazonesEnvSingleHand(
        agente_idx=agente_idx,
        politicas_oponentes=politicas,
        obs_dim=obs_dim,
    )


# ------------------------------------------------------------------
# Evaluacion: helpers de normalizacion
# ------------------------------------------------------------------

def _cargar_normalizador(venv_stats_path: str):
    """Carga el VecNormalize desde .pkl y retorna (mean, var) de obs_rms.

    VecNormalize.save() guarda el objeto completo, no un dict.
    """
    import pickle
    with open(venv_stats_path, "rb") as f:
        vn = pickle.load(f)
    # vn es un objeto VecNormalize; acceder a obs_rms directamente
    return vn.obs_rms.mean, vn.obs_rms.var


def _normalizar_obs(obs_raw: np.ndarray, obs_mean, obs_var) -> np.ndarray:
    """Normaliza una observacion con las stats de VecNormalize."""
    return np.clip(
        (obs_raw - obs_mean) / (np.sqrt(obs_var) + 1e-8),
        -10.0, 10.0,
    )


# ------------------------------------------------------------------
# Evaluacion: medir score promedio contra bots
# ------------------------------------------------------------------

def evaluar_score_promedio(
    modelo,
    venv_stats_path: Optional[str] = None,
    num_hands: int = EVAL_HANDS,
) -> Dict[str, float]:
    """Evalua el score promedio del modelo contra bots heuristicos.

    Juega N manos individuales (single-hand). Retorna el promedio
    de puntos tomados por mano. Menor = mejor.

    Args:
        modelo: MaskablePPO entrenado.
        venv_stats_path: Ruta al VecNormalize .pkl para normalizar obs.
        num_hands: Numero de manos a jugar.

    Returns:
        Dict con 'score_promedio', 'score_mediana', 'pct_mejor',
        'pct_peor', 'pct_cero' (manos perfectas, 0 puntos).
    """
    from src.v2_ronda.entorno import CorazonesEnvSingleHand
    from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo

    # Cargar normalizador una sola vez
    obs_mean = obs_var = None
    if venv_stats_path and os.path.exists(venv_stats_path):
        obs_mean, obs_var = _cargar_normalizador(venv_stats_path)

    scores: List[int] = []
    posiciones: List[int] = []

    for seed in range(num_hands):
        bots = [bot_conservador, bot_agresivo, bot_evasivo]
        random.shuffle(bots)
        env = CorazonesEnvSingleHand(
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

        # Score y posicion (ranking real, maneja empates)
        mi_score = env._puntos_mano_actual[0]
        rivales = [env._puntos_mano_actual[i] for i in range(1, 4)]
        scores.append(mi_score)

        todas = [mi_score] + rivales
        ranking = sorted(range(4), key=lambda i: todas[i])
        posicion = ranking.index(0)  # 0=mejor, 3=peor
        posiciones.append(posicion)

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


def evaluar_vs_experto_v2(
    modelo,
    venv_stats_path: Optional[str] = None,
    num_hands: int = EVAL_EXPERTO_HANDS,
) -> Dict[str, float]:
    """Evalua contra 3x BotExperto en manos individuales.

    Args:
        modelo: MaskablePPO entrenado.
        venv_stats_path: Ruta al VecNormalize .pkl.
        num_hands: Numero de manos.

    Returns:
        Dict con score_promedio y posicion.
    """
    from src.v2_ronda.entorno import CorazonesEnvSingleHand
    from src.agentes.bot_experto import BotExperto

    # Cargar normalizador una sola vez
    obs_mean = obs_var = None
    if venv_stats_path and os.path.exists(venv_stats_path):
        obs_mean, obs_var = _cargar_normalizador(venv_stats_path)

    scores: List[int] = []
    posiciones: List[int] = []

    for seed in range(num_hands):
        env = CorazonesEnvSingleHand(
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
        posicion = ranking.index(0)
        posiciones.append(posicion)

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
# Diagnosticos
# ------------------------------------------------------------------

def _capturar_diagnosticos(modelo) -> Dict[str, float]:
    """Extrae metricas de diagnostico del logger interno de SB3."""
    try:
        raw = modelo.logger.name_to_value
    except (AttributeError, TypeError):
        return {}
    if not raw:
        return {}
    out = {}
    for key, value in raw.items():
        if key.startswith("train/"):
            out[key.replace("train/", "")] = round(float(value), 6)
    return out


def _alertas_diagnostico(diag: Dict[str, float]) -> list:
    """Genera alertas si metricas cruzan umbrales peligrosos.

    Umbrales calibrados para v2 (score-based rewards, rango [-31, +52],
    normalizados ~[-10, +10] por VecNormalize):
        - value_loss > 3.0: MSE del value head demasiado alto.
          Con returns en [-10, 10], MSE 0.5-2.0 es normal durante entrenamiento.
        - explained_variance < 0.0: value head predice peor que la media.
          (En v1 el umbral era < 0.3, pero en v2 es normal valores 0.3-0.6)
        - entropy_loss > -0.15: politica colapsada (demasiado deterministica).
        - approx_kl > 0.03: cambio de politica demasiado brusco.
        - clip_fraction > 0.5: mas del 50% de updates clipados.
    """
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
    # v2: returns normalizados ~[-10, 10], MSE 0.5-2.0 normal. Solo alertar > 3.0
    if v_loss is not None and v_loss > 3.0:
        alertas.append(f"VALUE LOSS ELEVADO: value_loss={v_loss:.4f}")

    ev = diag.get("explained_variance")
    # v2: solo alertar si es negativo (predice peor que la media)
    if ev is not None and ev < 0.0:
        alertas.append(
            f"EXPLAINED VARIANCE NEGATIVO: explained_variance={ev:.4f}")

    return alertas


# ------------------------------------------------------------------
# Entrenamiento principal
# ------------------------------------------------------------------

def entrenar_v2(
    total_steps: int = 5_000_000,
    output_dir: str = "models/v2_score",
    resume_from: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu",
    eval_experto: bool = True,
    early_stop_patience: int = 3,
) -> int:
    """Entrenamiento autonomo v2: single-hand, score-based reward.

    Args:
        total_steps: Pasos totales de entrenamiento.
        output_dir: Directorio de salida para snapshots y logs.
        resume_from: Ruta a snapshot .zip para reanudar.
        seed: Semilla aleatoria.
        device: Dispositivo (cpu, cuda, dml).
        eval_experto: Si True, evalua contra BotExperto.
        early_stop_patience: Evaluaciones sin mejora para detener (0=no).

    Returns:
        Paso final alcanzado.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # --- Directorios ---
    dir_snapshots = os.path.join(output_dir, "snapshots")
    dir_vecnorm = os.path.join(output_dir, "vecnorm")
    best_dir = os.path.join(output_dir, "best")
    eval_log_path = os.path.join(dir_snapshots, "eval_log.jsonl")

    etiqueta = os.path.basename(output_dir)

    # Asegurar directorios (tanto para resume como desde cero)
    os.makedirs(dir_snapshots, exist_ok=True)
    os.makedirs(dir_vecnorm, exist_ok=True)

    # --- Resume ---
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

        # Cargar VecNormalize
        vn_path = ruta_resume.replace(".zip", "_vecnorm.pkl")

        def _make_env():  # noqa: E731
            return crear_entorno_self_play_v2(dir_snapshots, seed=seed + paso_actual)
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
        # --- Desde cero ---
        for d in [dir_snapshots, dir_vecnorm]:
            if os.path.isdir(d):
                shutil.rmtree(d)
        os.makedirs(dir_snapshots, exist_ok=True)
        os.makedirs(dir_vecnorm, exist_ok=True)
        os.makedirs(DIRECTORIO_LOGS, exist_ok=True)

        print("=" * 60)
        print(f"ENTRENAMIENTO v2_ronda: {etiqueta}")
        print("=" * 60)
        print(f"  Total steps: {total_steps:,}")
        print(f"  Snapshot cada: {SNAPSHOT_EVERY:,}")
        print(f"  Eval cada: {EVAL_EVERY} snapshots ({EVAL_HANDS} manos)")
        print(f"  Recompensa: score-based (26 - mis_puntos)")
        print(f"  Episode: 1 mano (~52 steps)")
        print(f"  Device: {device}")
        print("-" * 60)

        np.random.seed(seed)
        random.seed(seed)

        # Crear entorno y modelo
        def _make_env():  # noqa: E731
            return crear_entorno_self_play_v2(dir_snapshots, seed=seed)
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

    # --- Early stopping ---
    mejor_score_experto: float = 999.0
    contador_sin_mejora: int = 0

    vecnorm_path = os.path.join(dir_vecnorm, f"{etiqueta}_vecnorm.pkl")

    # --- Value head warmup (tras BC pretraining) ---
    # Si el modelo viene de BC, el value head esta sin entrenar (vf_coef=0).
    # Lo calentamos congelando el actor durante VALUE_WARMUP_STEPS pasos.
    VALUE_WARMUP_STEPS: int = 200_000
    if paso_actual == 0 and resume_from:
        print(
            f"\n  [Warmup] Calentando value head ({VALUE_WARMUP_STEPS:,} pasos, actor congelado)...")
        # Congelar actor (policy + feature extractor)
        for param in modelo.policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = False
        for param in modelo.policy.action_net.parameters():
            param.requires_grad = False
        # Congelar feature extractor (compartido)
        for param in modelo.policy.features_extractor.parameters():
            param.requires_grad = False

        # Entrenar solo value head
        modelo.vf_coef = 1.0
        modelo.ent_coef = 0.0
        modelo.learn(total_timesteps=VALUE_WARMUP_STEPS,
                     reset_num_timesteps=False, progress_bar=True)

        # Descongelar todo y restaurar hiperparametros RL
        for param in modelo.policy.parameters():
            param.requires_grad = True
        modelo.vf_coef = HP_DEFAULT["vf_coef"]
        modelo.ent_coef = HP_DEFAULT["ent_coef"]
        # bajar de 1e-3 (BC) a 3e-4 (RL)
        modelo.learning_rate = HP_DEFAULT["learning_rate"]
        # Actualizar optimizador con el nuevo LR
        if hasattr(modelo.policy, "optimizer") and modelo.policy.optimizer is not None:
            for param_group in modelo.policy.optimizer.param_groups:
                param_group["lr"] = HP_DEFAULT["learning_rate"]
        paso_actual += VALUE_WARMUP_STEPS
        print(f"  [Warmup] Value head calentado. Paso actual: {paso_actual:,}")
        # Guardar snapshot post-warmup
        venv.save(vecnorm_path)
        nombre = f"snapshot_{paso_actual:010d}"
        ruta = os.path.join(dir_snapshots, nombre)
        modelo.save(ruta)
        venv.save(ruta + "_vecnorm.pkl")

    t_start = time.time()

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

        # --- Diagnostico ---
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

        # --- Guardar snapshot ---
        venv.save(vecnorm_path)
        nombre = f"snapshot_{paso_actual:010d}"
        ruta = os.path.join(dir_snapshots, nombre)
        modelo.save(ruta)
        venv.save(ruta + "_vecnorm.pkl")
        print(f"  [Snapshot] {ruta}.zip")

        # --- Pruning ---
        snaps = _listar_snapshots(dir_snapshots)
        if len(snaps) > MAX_SNAPSHOTS_POOL:
            a_eliminar = snaps[:len(snaps) - MAX_SNAPSHOTS_POOL]
            for s in a_eliminar:
                os.remove(s)
                vec = s.replace(".zip", "_vecnorm.pkl")
                if os.path.exists(vec):
                    os.remove(vec)
            print(f"  [Pruning] {len(a_eliminar)} antiguos eliminados")

        # --- Evaluacion ---
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
                    res_exp = evaluar_vs_experto_v2(
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

                    # Early stopping basado en score_experto
                    if early_stop_patience > 0:
                        if sc_exp < mejor_score_experto - 0.5:
                            mejor_score_experto = sc_exp
                            contador_sin_mejora = 0
                            # Guardar en best/
                            os.makedirs(best_dir, exist_ok=True)
                            shutil.copy2(ruta + ".zip",
                                         os.path.join(best_dir, nombre + ".zip"))
                            shutil.copy2(ruta + "_vecnorm.pkl",
                                         os.path.join(best_dir, nombre + "_vecnorm.pkl"))
                            print(
                                f"  [Best] Nuevo mejor score_experto={sc_exp:.1f}")
                        else:
                            contador_sin_mejora += 1
                            print(
                                f"  [EarlyStop] Sin mejora {contador_sin_mejora}/{early_stop_patience}")
                            if contador_sin_mejora >= early_stop_patience:
                                print(
                                    f"\n  Early stopping activado: score_experto no mejora en {early_stop_patience} evals")
                                break
                except Exception as e:
                    print(f"  [Eval] Error vs Experto: {e}")

        # --- Actualizar entorno self-play ---
        if snapshot_count % 2 == 0:
            env_nuevo = crear_entorno_self_play_v2(
                dir_snapshots, seed=seed + paso_actual,
            )
            _env_ref = env_nuevo
            venv = DummyVecEnv([lambda: _env_ref])
            if os.path.exists(vecnorm_path):
                venv = VecNormalize.load(vecnorm_path, venv)
            else:
                venv = VecNormalize(
                    venv, norm_obs=True, norm_reward=True,
                    clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
                )
            modelo.set_env(venv)

    # --- Fin ---
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(
        f"ENTRENAMIENTO COMPLETADO: {paso_actual:,} pasos en {elapsed/3600:.1f}h")
    print(f"  Snapshots: {dir_snapshots}")
    print(f"  Best: {best_dir}")
    print("=" * 60)

    return paso_actual


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento v2_ronda — Single-hand, score-based rewards",
    )
    parser.add_argument("--total-steps", type=int, default=5_000_000,
                        help="Pasos totales de entrenamiento.")
    parser.add_argument("--output-dir", type=str, default="models/v2_score",
                        help="Directorio de salida para snapshots.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta a snapshot .zip para reanudar.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "dml"],
                        help="Dispositivo de computo.")
    parser.add_argument("--no-eval-experto", action="store_true",
                        help="Desactivar evaluacion vs BotExperto.")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="Evals sin mejora para detener (0=no).")

    args = parser.parse_args()

    entrenar_v2(
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        resume_from=args.resume,
        seed=args.seed,
        device=args.device,
        eval_experto=not args.no_eval_experto,
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    main()
