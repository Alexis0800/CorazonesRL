"""
Entrenamiento autónomo v6 desde cero con evaluaciones y torneos ELO.

Ejecuta un ciclo completo de entrenamiento con:
  - Decaimiento progresivo de prob_bot (0.50 → 0.10)
  - LR schedule en 3 fases
  - Evaluación de win rate cada N snapshots → v6/eval_log.jsonl
  - Torneo ELO cada M snapshots (en proceso separado) → v6/elo_*.txt
  - Snapshots + VecNormalize guardados en modelos_historicos/v6/

Uso:
    python train_auto_v6.py --total-steps 20000000 --snapshot-every 100000
"""

from __future__ import annotations
from src.evaluacion import evaluar_snapshot_callback
from src.entorno import CorazonesEnv
from train_self_play import (
    DIRECTORIO_MODELOS_V6, DIRECTORIO_VECNORM_V6, DIRECTORIO_LOGS,
    MIN_SNAPSHOT_STEPS, MAX_SNAPSHOTS_POOL,
    obtener_policy_kwargs, obtener_hiperparametros_v3,
    _aplicar_snapshot_pruning, _extraer_paso_de_ruta,
    PoliticaSB3,
    crear_entorno_self_play_v6,
)

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Asegurar que el directorio del proyecto está en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
PROB_BOT_START: float = 0.50
PROB_BOT_END: float = 0.20
EVAL_PARTIDAS: int = 100
ELO_PARTIDAS: int = 30
ELO_MAX_SNAPSHOTS: int = 12


# ------------------------------------------------------------------
# Decaimiento progresivo de prob_bot
# ------------------------------------------------------------------

def prob_bot_actual(
    paso_actual: int,
    total_pasos: int,
    inicio: float = PROB_BOT_START,
    fin: float = PROB_BOT_END,
) -> float:
    """Calcula prob_bot con decaimiento coseno según el progreso.

    El decaimiento coseno mantiene valores más altos durante más tiempo
    que el lineal, reduciendo el riesgo de sobreajuste al self-play
    en etapas tempranas.

    Fórmula: fin + 0.5 * (inicio - fin) * (1 + cos(pi * progreso))

    Args:
        paso_actual: Paso global actual.
        total_pasos: Pasos totales planeados.
        inicio: prob_bot inicial (default 0.50).
        fin: prob_bot final / piso (default 0.20).

    Returns:
        Probabilidad de usar un bot heurístico como oponente.
    """
    import math
    if total_pasos <= 0:
        return fin
    progreso = min(paso_actual / total_pasos, 1.0)
    return fin + 0.5 * (inicio - fin) * (1.0 + math.cos(math.pi * progreso))


# ------------------------------------------------------------------
# Logging JSONL
# ------------------------------------------------------------------

def log_eval(
    log_path: str,
    paso: int,
    win_rate: float,
    avg_score: float,
    num_partidas: int,
    prob_bot: float,
) -> None:
    """Registra resultado de evaluación en formato JSONL.

    Args:
        log_path: Ruta al archivo .jsonl.
        paso: Paso de entrenamiento.
        win_rate: Tasa de victorias (0.0 a 1.0).
        avg_score: Puntuación promedio del agente.
        num_partidas: Número de partidas evaluadas.
        prob_bot: prob_bot usado en el entrenamiento.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "paso": paso,
        "win_rate_bots": round(win_rate, 4),
        "avg_score": round(avg_score, 2),
        "num_partidas": num_partidas,
        "prob_bot": round(prob_bot, 4),
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------
# Torneo ELO asíncrono
# ------------------------------------------------------------------

def lanzar_torneo_elo(
    directorio: str,
    output_file: str,
    min_paso: int = 0,
    max_snapshots: int = ELO_MAX_SNAPSHOTS,
    partidas: int = ELO_PARTIDAS,
) -> Optional[subprocess.Popen]:
    """Lanza torneo ELO en un proceso separado (no bloqueante).

    Args:
        directorio: Directorio de snapshots.
        output_file: Archivo donde guardar la salida.
        min_paso: Paso mínimo para incluir snapshots.
        max_snapshots: Máximo de snapshots en el torneo.
        partidas: Partidas por enfrentamiento.

    Returns:
        Objeto Popen del proceso, o None si falló.
    """
    cmd = [
        sys.executable, "-m", "src.elo_torneo",
        "--directorio", directorio,
        "--partidas", str(partidas),
        "--min-paso", str(min_paso),
        "--max-snapshots", str(max_snapshots),
        "--elo-puro",
        "--incluir-bots",
    ]
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            proc = subprocess.Popen(
                cmd,
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        return proc
    except Exception as e:
        print(f"  ⚠️  No se pudo lanzar torneo ELO: {e}")
        return None


# ------------------------------------------------------------------
# Entrenamiento autónomo
# ------------------------------------------------------------------

def entrenar_auto(
    total_steps: int = 20_000_000,
    snapshot_every: int = 100_000,
    eval_every: int = 5,
    elo_every: int = 20,
    eval_partidas: int = EVAL_PARTIDAS,
    prob_bot_start: float = PROB_BOT_START,
    prob_bot_end: float = PROB_BOT_END,
    seed: int = 42,
    device: str = "cpu",
    resume_from: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> int:
    """Ejecuta entrenamiento autónomo completo desde cero o reanudando.

    Args:
        total_steps: Pasos totales a entrenar.
        snapshot_every: Guardar snapshot cada N pasos.
        eval_every: Evaluar win rate cada N snapshots.
        elo_every: Lanzar torneo ELO cada N snapshots.
        eval_partidas: Partidas por evaluación.
        prob_bot_start: prob_bot inicial.
        prob_bot_end: prob_bot final.
        seed: Semilla aleatoria.
        device: Dispositivo de cómputo.
        resume_from: Ruta a snapshot .zip para reanudar.
        output_dir: Directorio de salida (default: modelos_historicos/v6).

    Returns:
        Paso final alcanzado.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dir_snapshots = DIRECTORIO_MODELOS_V6
    dir_vecnorm = DIRECTORIO_VECNORM_V6

    # Permitir directorio de salida personalizado (ej: v7)
    if output_dir:
        dir_snapshots = os.path.join(output_dir, "snapshots")
        dir_vecnorm = os.path.join(output_dir, "vecnorm")
        os.makedirs(dir_snapshots, exist_ok=True)
        os.makedirs(dir_vecnorm, exist_ok=True)

    eval_log_path = os.path.join(dir_snapshots, "eval_log.jsonl")
    etiqueta = os.path.basename(output_dir) if output_dir else "v6"

    # --- Modo resume: cargar modelo existente ---
    if resume_from:
        if not resume_from.endswith(".zip"):
            resume_from += ".zip"
        ruta_resume = resume_from if os.path.isabs(resume_from) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), resume_from)
        if not os.path.exists(ruta_resume):
            print(f"ERROR: No se encuentra {ruta_resume}")
            sys.exit(1)

        paso_actual = _extraer_paso_de_ruta(ruta_resume)
        print("=" * 60)
        print(f"🔄 REANUDANDO {etiqueta} desde snapshot paso {paso_actual:,}")
        print("=" * 60)

        # Cargar modelo
        modelo = MaskablePPO.load(ruta_resume, device=device)
        # Cargar VecNormalize
        vn_path = ruta_resume.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vn_path):
            venv = VecNormalize.load(vn_path, DummyVecEnv(
                [lambda: CorazonesEnv(agente_idx=0)]))
        else:
            venv = DummyVecEnv([lambda: CorazonesEnv(agente_idx=0)])
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                                clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        modelo.set_env(venv)

        vecnorm_path = os.path.join(dir_vecnorm,
            f"{'v7' if output_dir and 'v7' in output_dir else 'v6'}_vecnorm.pkl")
        snapshot_count = paso_actual // snapshot_every
        print(f"  Paso actual: {paso_actual:,}")
        print(f"  Snapshots → {dir_snapshots}")
        print(f"  VecNormalize → {dir_vecnorm}")
        print("-" * 60)
    else:
        # --- Modo desde cero: limpiar y crear ---
        for d in [dir_snapshots, dir_vecnorm]:
            if os.path.isdir(d):
                shutil.rmtree(d)
        os.makedirs(dir_snapshots, exist_ok=True)
        os.makedirs(dir_vecnorm, exist_ok=True)
        os.makedirs(DIRECTORIO_LOGS, exist_ok=True)

        print("=" * 60)
        print(f"🤖 ENTRENAMIENTO AUTÓNOMO {etiqueta} (194 dims, desde cero)")
        print("=" * 60)
        print(f"  Pasos totales: {total_steps:,}")
        print(f"  Snapshot cada: {snapshot_every:,}")
        print(
            f"  Evaluación cada: {eval_every} snapshots ({eval_partidas} partidas)")
        print(f"  Torneo ELO cada: {elo_every} snapshots")
        print(f"  prob_bot: {prob_bot_start:.0%} → {prob_bot_end:.0%}")
        print(f"  LR schedule: 3 fases (1e-4 → 5e-5 → 2e-5)")
        print(f"  Snapshots → {dir_snapshots}")
        print(f"  VecNormalize → {dir_vecnorm}")
        print(f"  Eval log → {eval_log_path}")
        print("-" * 60)

        np.random.seed(seed)

        # Crear entorno base
        env_base = CorazonesEnv(agente_idx=0)
        env_base.reset(seed=seed)

        # VecNormalize desde cero
        venv = DummyVecEnv([lambda: CorazonesEnv(agente_idx=0)])
        venv = VecNormalize(
            venv, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
        )

        # Crear modelo desde cero
        hp = obtener_hiperparametros_v3(DIRECTORIO_LOGS, device, 0)
        policy_kwargs = hp.pop(
            "policy_kwargs", None) or obtener_policy_kwargs()
        fase_label = hp.pop("_fase", "?")
        hp.pop("tensorboard_log", None)

        modelo = MaskablePPO(
            policy=hp["policy"],
            env=venv,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            n_epochs=hp["n_epochs"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            clip_range=hp["clip_range"],
            normalize_advantage=hp["normalize_advantage"],
            ent_coef=hp["ent_coef"],
            vf_coef=hp["vf_coef"],
            max_grad_norm=hp["max_grad_norm"],
            target_kl=hp["target_kl"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=DIRECTORIO_LOGS,
        )
        print(f"  [OK] Modelo creado desde cero ({fase_label})")

        # Variables de estado
        paso_actual = 0
        snapshot_count = 0
        vecnorm_path = os.path.join(dir_vecnorm, f"{etiqueta}_vecnorm.pkl")
    procesos_elo: List[subprocess.Popen] = []

    t_start = time.time()

    # Bucle principal
    while paso_actual < total_steps:
        # Calcular prob_bot y LR para esta fase
        pb = prob_bot_actual(paso_actual, total_steps,
                             prob_bot_start, prob_bot_end)
        hp_actual = obtener_hiperparametros_v3(
            DIRECTORIO_LOGS, device, paso_actual)
        nueva_fase = hp_actual.pop("_fase", "?")
        hp_actual.pop("policy", None)
        hp_actual.pop("policy_kwargs", None)
        hp_actual.pop("verbose", None)
        hp_actual.pop("device", None)
        hp_actual.pop("tensorboard_log", None)

        # Aplicar hiperparámetros
        for key in ["learning_rate", "ent_coef", "vf_coef", "gamma",
                    "gae_lambda", "target_kl", "max_grad_norm",
                    "n_steps", "batch_size", "n_epochs"]:
            setattr(modelo, key, hp_actual[key])
        modelo.clip_range = lambda _: hp_actual["clip_range"]

        # Actualizar lr del optimizador
        if hasattr(modelo.policy, "optimizer") and modelo.policy.optimizer is not None:
            for param_group in modelo.policy.optimizer.param_groups:
                param_group["lr"] = hp_actual["learning_rate"]

        # Recrear entorno con el prob_bot actual
        env_nuevo = CorazonesEnv(agente_idx=0)
        env_nuevo.reset(seed=seed + paso_actual)
        venv = DummyVecEnv([lambda: env_nuevo])
        if os.path.exists(vecnorm_path):
            venv = VecNormalize.load(vecnorm_path, venv)
        else:
            venv = VecNormalize(
                venv, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
            )
        modelo.set_env(venv)

        bloque = min(snapshot_every, total_steps - paso_actual)

        print(f"\n--- Bloque {snapshot_count + 1} | Paso {paso_actual:,} → "
              f"{paso_actual + bloque:,} | pb={pb:.0%} | {nueva_fase} ---")

        modelo.learn(
            total_timesteps=bloque,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        paso_actual += bloque
        snapshot_count += 1

        # Guardar snapshot
        venv.save(vecnorm_path)
        nombre = f"snapshot_{paso_actual:010d}"
        ruta = os.path.join(dir_snapshots, nombre)
        modelo.save(ruta)
        venv.save(ruta + "_vecnorm.pkl")
        print(f"  [Snapshot] {ruta}.zip ({paso_actual:,}/{total_steps:,})")

        # Pruning
        eliminados = _aplicar_snapshot_pruning(
            dir_snapshots, MAX_SNAPSHOTS_POOL)
        if eliminados > 0:
            print(f"  [Pruning] {eliminados} antiguos eliminados")

        # Evaluación periódica
        if eval_every > 0 and snapshot_count % eval_every == 0:
            print(f"\n  📊 Evaluando win rate ({eval_partidas} partidas)...")
            try:
                resultado = evaluar_snapshot_callback(
                    ruta_snapshot=ruta,
                    paso=paso_actual,
                    log_path=eval_log_path,  # usa el mismo archivo para no duplicar
                    num_partidas=eval_partidas,
                    min_win_rate=0.0,
                )
                wr = resultado.get("win_rate_bots", -1)
                avg_s = resultado.get("punt_promedio", -1)
                print(f"  📈 WR={wr:.1%} | AvgScore={avg_s:.1f}")
            except Exception as e:
                print(f"  ⚠️  Error en evaluación: {e}")

        # Torneo ELO periódico
        if elo_every > 0 and snapshot_count % elo_every == 0:
            min_paso_elo = max(
                0, paso_actual - snapshot_every * elo_max_snapshots())
            elo_out = os.path.join(
                dir_snapshots,
                f"elo_paso_{paso_actual:010d}.txt",
            )
            print(f"\n  🏆 Lanzando torneo ELO (min_paso={min_paso_elo:,})...")
            proc = lanzar_torneo_elo(
                directorio=dir_snapshots,
                output_file=elo_out,
                min_paso=min_paso_elo,
                max_snapshots=ELO_MAX_SNAPSHOTS,
            )
            if proc:
                procesos_elo.append(proc)
                print(f"  [OK] Torneo ELO en segundo plano (PID {proc.pid})")

        # Limpiar procesos ELO terminados
        procesos_elo = [p for p in procesos_elo if p.poll() is None]

    # --- Fin del entrenamiento ---
    elapsed = time.time() - t_start
    horas = int(elapsed // 3600)
    minutos = int((elapsed % 3600) // 60)

    print("\n" + "=" * 60)
    print(f"✅ ENTRENAMIENTO COMPLETADO en {horas}h {minutos}m")
    print(f"   Paso final: {paso_actual:,}")
    print(f"   Snapshots: {snapshot_count}")
    print(f"   Último modelo: {ruta}.zip")
    print("=" * 60)

    # Esperar procesos ELO pendientes (máx 5 min)
    if procesos_elo:
        print("⏳ Esperando torneos ELO pendientes...")
        for proc in procesos_elo:
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[OK] Torneos ELO finalizados")

    return paso_actual


def elo_max_snapshots() -> int:
    """Número de snapshots a incluir en torneos ELO."""
    return ELO_MAX_SNAPSHOTS


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento autónomo v6 desde cero (194 dims)")
    parser.add_argument("--total-steps", type=int, default=20_000_000,
                        help="Pasos totales a entrenar (default: 20M)")
    parser.add_argument("--snapshot-every", type=int, default=100_000,
                        help="Guardar snapshot cada N pasos (default: 100K)")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluar WR cada N snapshots (default: 5)")
    parser.add_argument("--elo-every", type=int, default=20,
                        help="Torneo ELO cada N snapshots (default: 20)")
    parser.add_argument("--eval-partidas", type=int, default=100,
                        help="Partidas por evaluación (default: 100)")
    parser.add_argument("--prob-bot-start", type=float, default=PROB_BOT_START)
    parser.add_argument("--prob-bot-end", type=float, default=PROB_BOT_END)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="Reanudar desde snapshot .zip existente")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida para snapshots (default: modelos_historicos/v6)")
    args = parser.parse_args()

    paso_final = entrenar_auto(
        total_steps=args.total_steps,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        elo_every=args.elo_every,
        eval_partidas=args.eval_partidas,
        prob_bot_start=args.prob_bot_start,
        prob_bot_end=args.prob_bot_end,
        seed=args.seed,
        device=args.device,
        resume_from=args.resume,
        output_dir=args.output_dir,
    )
    print(f"\nPaso final alcanzado: {paso_final:,}")


if __name__ == "__main__":
    main()
