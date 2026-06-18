"""
Entrenamiento autónomo v6 con evaluaciones, torneos ELO y guardado elite.

Ejecuta un ciclo completo de entrenamiento con:
  - Decaimiento progresivo de prob_bot (0.50 → 0.20, cosine decay)
  - LR schedule en 3 fases
  - Evaluación de win rate cada N snapshots → eval_log.jsonl
  - Torneo ELO cada M snapshots (asíncrono) → elo_paso_*.txt
  - Guardado automático de los N mejores snapshots tras cada torneo → best/
  - Snapshots + VecNormalize guardados en el directorio de salida
  - Soporte GPU: cpu, cuda, dml (DirectML/Intel Arc), xpu (Intel XPU)

Uso:
    # Desde cero
    python train.py --total-steps 20000000 --output-dir modelos/v8

    # Reanudar desde golden
    python train.py --resume modelos/v7_golden/snapshots/snapshot_0014900000 --total-steps 25000000 --output-dir modelos/v7_cont

    # Con GPU Intel Arc
    pip install torch-directml
    python train.py --total-steps 20000000 --device dml --output-dir modelos/v8
"""

from __future__ import annotations
from src.torneo.evaluacion import evaluar_snapshot_callback
from src.entorno.single_agent import CorazonesEnv
from train_self_play import (
    DIRECTORIO_MODELOS_V6, DIRECTORIO_VECNORM_V6, DIRECTORIO_LOGS,
    MIN_SNAPSHOT_STEPS, MAX_SNAPSHOTS_POOL,
    obtener_policy_kwargs, obtener_hiperparametros_v3,
    _aplicar_snapshot_pruning, _extraer_paso_de_ruta,
    crear_entorno_self_play,
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
# Soporte GPU Intel Arc / DirectML
# ------------------------------------------------------------------


def _configurar_dispositivo(device: str) -> str:
    """Configura el dispositivo de cómputo, con soporte para Intel Arc.

    Detecta si se solicita 'dml' (DirectML) o 'xpu' (Intel XPU) y
    verifica que las dependencias estén instaladas.

    Args:
        device: String del dispositivo ('cpu', 'cuda', 'dml', 'xpu').

    Returns:
        String del dispositivo validado.
    """
    if device == "dml":
        try:
            import torch_directml  # noqa: F401
            print("[GPU] DirectML habilitado para Intel Arc / AMD / NVIDIA")
        except ImportError:
            print("ERROR: torch-directml no instalado.")
            print("  Instálalo con: pip install torch-directml")
            sys.exit(1)
    elif device == "xpu":
        import torch
        if torch.xpu.is_available():
            print(
                f"[GPU] Intel XPU habilitado — {torch.xpu.device_count()} dispositivo(s) detectado(s)")
        else:
            print("ERROR: XPU no disponible. Verifica drivers Intel Arc y PyTorch+xpu.")
            print("  Instálalo con: pip install torch==2.12.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu")
            sys.exit(1)
    return device


# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
PROB_BOT_START: float = 0.50
PROB_BOT_END: float = 0.30
EVAL_PARTIDAS: int = 100
ELO_PARTIDAS: int = 30
ELO_MAX_SNAPSHOTS: int = 12
BEST_TOP: int = 2  # Cuántos mejores snapshots guardar tras cada torneo Elo


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
# Diagnóstico automático de entrenamiento
# ------------------------------------------------------------------

def _capturar_diagnosticos(modelo) -> Dict[str, float]:
    """Extrae métricas de diagnóstico del logger interno de SB3.

    Captura las métricas con prefijo 'train/' que SB3 registra
    durante learn(). Útil para detectar colapso de política,
    divergencia de value function, o exploding KL.

    Args:
        modelo: Instancia de MaskablePPO tras una llamada a learn().

    Returns:
        Diccionario {metrica_sin_prefijo: valor} con las métricas
        relevantes, o dict vacío si el logger no tiene datos.
    """
    try:
        raw = modelo.logger.name_to_value
    except (AttributeError, TypeError):
        return {}

    if not raw:
        return {}

    diagnosticas = {}
    for key, value in raw.items():
        if key.startswith("train/"):
            nombre = key.replace("train/", "")
            diagnosticas[nombre] = round(float(value), 6)
    return diagnosticas


def _alertas_diagnostico(diag: Dict[str, float]) -> list:
    """Genera alertas textuales si alguna métrica cruza umbrales peligrosos.

    Umbrales calibrados para Corazones con PPO + action masking:
        - entropy_loss > -0.15: colapso real de entropía.
          Con ~7 acciones legales típicas, entropía < 0.15 = < 8% del máximo.
          entropy_loss = -1.0 es NORMAL (50% del máximo con masking).
        - approx_kl > 0.03: cambio de política demasiado brusco.
        - clip_fraction > 0.5: >50% de updates están siendo clipados.
        - value_loss > 10.0: value function divergiendo.

    Args:
        diag: Diccionario de métricas de _capturar_diagnosticos.

    Returns:
        Lista de strings con alertas (vacía si todo está bien).
    """
    alertas = []

    entropy = diag.get("entropy_loss")
    if entropy is not None and entropy > -0.15:
        alertas.append(
            f"⚠️  COLAPSO DE ENTROPÍA: entropy_loss={entropy:.4f} (> -0.15). "
            f"La política es sobredeterminística (entropía < 0.15 con action masking)."
        )

    kl = diag.get("approx_kl")
    if kl is not None and kl > 0.03:
        alertas.append(
            f"⚠️  KL DIVERGENTE: approx_kl={kl:.4f} (> 0.03). "
            f"La política está cambiando demasiado rápido."
        )

    clip = diag.get("clip_fraction")
    if clip is not None and clip > 0.5:
        alertas.append(
            f"⚠️  CLIP SATURADO: clip_fraction={clip:.4f} (> 0.5). "
            f"Más del 50% de updates están siendo recortados."
        )

    v_loss = diag.get("value_loss")
    if v_loss is not None and v_loss > 10.0:
        alertas.append(
            f"⚠️  VALUE LOSS ALTO: value_loss={v_loss:.2f} (> 10.0). "
            f"La value function puede estar divergiendo."
        )

    return alertas


# ------------------------------------------------------------------
# Torneo ELO asíncrono
# ------------------------------------------------------------------

def lanzar_torneo_elo(
    directorio: str,
    output_file: str,
    min_paso: int = 0,
    max_snapshots: int = ELO_MAX_SNAPSHOTS,
    partidas: int = ELO_PARTIDAS,
    incluir_experto: bool = False,
) -> Optional[subprocess.Popen]:
    """Lanza torneo ELO en un proceso separado (no bloqueante).

    Args:
        directorio: Directorio de snapshots.
        output_file: Archivo donde guardar la salida.
        min_paso: Paso mínimo para incluir snapshots.
        max_snapshots: Máximo de snapshots en el torneo.
        partidas: Partidas por enfrentamiento.
        incluir_experto: Si True, incluye BotExperto como participante.

    Returns:
        Objeto Popen del proceso, o None si falló.
    """
    cmd = [
        sys.executable, "-m", "src.torneo.elo",
        "--directorio", directorio,
        "--partidas", str(partidas),
        "--min-paso", str(min_paso),
        "--max-snapshots", str(max_snapshots),
        "--elo-puro",
        "--incluir-bots",
    ]
    if incluir_experto:
        cmd.append("--incluir-experto")
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
# Guardado automático de mejores snapshots
# ------------------------------------------------------------------

def _guardar_mejores_snapshots(
    output_file: str,
    dir_snapshots: str,
    best_dir: str,
    top_n: int = BEST_TOP,
) -> int:
    """Parsea el resultado de un torneo Elo y copia los top N snapshots.

    Lee el archivo de salida del torneo, extrae el ranking, filtra
    los bots y copia los mejores N modelos + sus VecNormalize a un
    directorio permanente de "mejores".

    Args:
        output_file: Ruta al archivo de salida del torneo Elo.
        dir_snapshots: Directorio donde están los snapshots originales.
        best_dir: Directorio destino para los mejores snapshots.
        top_n: Número de mejores snapshots a guardar (default 2).

    Returns:
        Número de snapshots guardados.
    """
    import re

    if not os.path.exists(output_file):
        print(f"  ⚠️  Output de torneo no encontrado: {output_file}")
        return 0

    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Encontrar sección del ranking
    ranking_start = content.find("CLASIFICACION FINAL")
    if ranking_start == -1:
        print(f"  ⚠️  No se encontró ranking en {output_file}")
        return 0

    ranking_text = content[ranking_start:]

    # Parsear líneas del ranking:  Pos   Snapshot                       Paso         Elo      Diff
    top_snapshots: List[str] = []
    for line in ranking_text.split("\n"):
        # Ej: "1     snapshot_0012000000            12,000,000  1829     -82"
        # Ej: "1     [BOT] evasivo                       (bot)  1911     "
        match = re.match(r'\s*(\d+)\s+(\S+)', line)
        if match:
            name = match.group(2)
            if name.startswith("[BOT]"):
                continue  # Saltar bots
            top_snapshots.append(name)
            if len(top_snapshots) >= top_n:
                break

    if not top_snapshots:
        print("  ⚠️  No se encontraron snapshots de modelo en el ranking")
        return 0

    os.makedirs(best_dir, exist_ok=True)
    guardados = 0

    for name in top_snapshots:
        src_zip = os.path.join(dir_snapshots, name + ".zip")
        src_vec = os.path.join(dir_snapshots, name + "_vecnorm.pkl")

        if os.path.exists(src_zip):
            shutil.copy2(src_zip, os.path.join(best_dir, name + ".zip"))
            print(f"  💾 Mejor snapshot guardado: {name}.zip → {best_dir}/")
            guardados += 1
        if os.path.exists(src_vec):
            shutil.copy2(src_vec, os.path.join(
                best_dir, name + "_vecnorm.pkl"))

    return guardados


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
    prob_experto: float = 0.0,
    seed: int = 42,
    device: str = "cpu",
    resume_from: Optional[str] = None,
    output_dir: Optional[str] = None,
    best_top: int = BEST_TOP,
    obs_dim: int = 220,
    bc_pretrain: Optional[str] = None,
    eval_experto: bool = False,
    eval_experto_partidas: int = 30,
    elo_experto: bool = False,
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
        prob_experto: Prob. de usar BotExperto como oponente (default 0.0).
        seed: Semilla aleatoria.
        device: Dispositivo de cómputo (cpu, cuda, dml, xpu).
        resume_from: Ruta a snapshot .zip para reanudar.
        output_dir: Directorio de salida (default: modelos/v6).
        best_top: Cuántos mejores snapshots guardar tras cada torneo Elo (default 2).
        obs_dim: Dimensión del vector de observación (194 o 220).
        bc_pretrain: Ruta a modelo BC preentrenado .zip para inicializar pesos.

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
    # Validar dispositivo (Intel Arc / DirectML)
    device = _configurar_dispositivo(device)

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
        def _make_env(): return CorazonesEnv(agente_idx=0, obs_dim=obs_dim)  # noqa: E731
        if os.path.exists(vn_path):
            venv = VecNormalize.load(vn_path, DummyVecEnv([_make_env]))
        else:
            venv = DummyVecEnv([_make_env])
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
        print(
            f"ENTRENAMIENTO AUTONOMO {etiqueta} ({obs_dim} dims, desde cero)")
        print("=" * 60)
        print(f"  Pasos totales: {total_steps:,}")
        print(f"  Snapshot cada: {snapshot_every:,}")
        print(
            f"  Evaluación cada: {eval_every} snapshots ({eval_partidas} partidas)")
        print(f"  Torneo ELO cada: {elo_every} snapshots")
        print(f"  prob_bot: {prob_bot_start:.0%} → {prob_bot_end:.0%}")
        print(f"  LR schedule: 3 fases (5e-4 → 3e-4 → 1e-4)")
        print(f"  Snapshots → {dir_snapshots}")
        print(f"  VecNormalize → {dir_vecnorm}")
        print(f"  Eval log → {eval_log_path}")
        print("-" * 60)

        np.random.seed(seed)

        # Crear entorno base
        def _make_env(): return CorazonesEnv(agente_idx=0, obs_dim=obs_dim)  # noqa: E731
        env_base = _make_env()
        env_base.reset(seed=seed)

        # VecNormalize desde cero
        venv = DummyVecEnv([_make_env])
        venv = VecNormalize(
            venv, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8,
        )

        # Crear modelo desde cero (o inicializar desde BC preentrenado)
        hp = obtener_hiperparametros_v3(
            DIRECTORIO_LOGS, device, 0, total_steps)
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

        # Cargar pesos del modelo BC preentrenado (solo actor)
        if bc_pretrain:
            _ruta_bc = bc_pretrain if bc_pretrain.endswith(
                ".zip") else bc_pretrain + ".zip"
            if os.path.exists(_ruta_bc):
                _bc_model = MaskablePPO.load(_ruta_bc, device=device)
                modelo.policy.load_state_dict(_bc_model.policy.state_dict())
                del _bc_model
                print(f"  [BC] Pesos del actor cargados desde {_ruta_bc}")
            else:
                print(
                    f"  [WARN] bc_pretrain no encontrado: {_ruta_bc} — ignorado")

        # Variables de estado
        paso_actual = 0
        snapshot_count = 0
        vecnorm_path = os.path.join(dir_vecnorm, f"{etiqueta}_vecnorm.pkl")

    # Directorio para guardar los mejores snapshots de cada torneo Elo
    best_dir = os.path.join(os.path.dirname(dir_snapshots), "best")

    # Lista de (proceso, output_file) para torneos Elo asíncronos
    procesos_elo: List[tuple] = []

    t_start = time.time()

    # Bucle principal
    while paso_actual < total_steps:
        # Calcular prob_bot y LR para esta fase
        pb = prob_bot_actual(paso_actual, total_steps,
                             prob_bot_start, prob_bot_end)
        hp_actual = obtener_hiperparametros_v3(
            DIRECTORIO_LOGS, device, paso_actual, total_steps)
        nueva_fase = hp_actual.pop("_fase", "?")
        hp_actual.pop("policy", None)
        hp_actual.pop("policy_kwargs", None)
        hp_actual.pop("verbose", None)
        hp_actual.pop("device", None)
        hp_actual.pop("tensorboard_log", None)

        # Aplicar hiperparámetros (incluyendo lr_schedule para SB3)
        for key in ["learning_rate", "ent_coef", "vf_coef", "gamma",
                    "gae_lambda", "target_kl", "max_grad_norm",
                    "n_steps", "batch_size", "n_epochs"]:
            setattr(modelo, key, hp_actual[key])
        modelo.clip_range = lambda _: hp_actual["clip_range"]
        # Forzar actualización del lr_schedule usado por SB3 internamente
        modelo.lr_schedule = lambda _: hp_actual["learning_rate"]

        # Actualizar lr del optimizador
        if hasattr(modelo.policy, "optimizer") and modelo.policy.optimizer is not None:
            for param_group in modelo.policy.optimizer.param_groups:
                param_group["lr"] = hp_actual["learning_rate"]

        # Recrear entorno self-play con prob_bot y prob_experto actuales
        env_nuevo = crear_entorno_self_play(
            directorio=dir_snapshots,
            agente_idx=0,
            seed=seed + paso_actual,
            prob_bot=pb,
            prob_experto=prob_experto,
            obs_dim=obs_dim,
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

        # --- Diagnóstico automático (v13): capturar métricas internas de SB3 ---
        diag = _capturar_diagnosticos(modelo)
        alertas = _alertas_diagnostico(diag)
        if alertas:
            for a in alertas:
                print(f"  {a}")
        if diag:
            diag_entry = {
                "timestamp": datetime.now().isoformat(),
                "paso": paso_actual,
                "tipo": "diagnostico",
                **diag,
            }
            with open(eval_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(diag_entry) + "\n")
            # Mostrar las 3 métricas más importantes en consola
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
            # Evaluación contra 3× BotExperto
            if eval_experto:
                print(
                    f"  🤖 Evaluando vs 3× BotExperto ({eval_experto_partidas} partidas)...")
                try:
                    from src.torneo.evaluacion import evaluar_vs_experto
                    res_exp = evaluar_vs_experto(
                        ruta_snapshot=ruta,
                        num_partidas=eval_experto_partidas,
                    )
                    wr_exp = res_exp["pct_primero"]
                    top2_exp = res_exp["pct_top2"]
                    cuarto_exp = res_exp["pct_cuarto"]
                    avg_exp = res_exp["punt_promedio"]
                    print(
                        f"  🤖 vsExperto: WR={wr_exp:.1%} | Top2={top2_exp:.1%} | 4º={cuarto_exp:.1%} | Avg={avg_exp:.1f}")
                    # Guardar en el mismo eval_log.jsonl
                    entry_exp = {
                        "timestamp": datetime.now().isoformat(),
                        "paso": paso_actual,
                        "snapshot": nombre,
                        "win_rate_experto": round(wr_exp, 4),
                        "top2_experto": round(top2_exp, 4),
                        "pct_cuarto_experto": round(cuarto_exp, 4),
                        "avg_score_experto": round(avg_exp, 2),
                        "num_partidas_experto": eval_experto_partidas,
                    }
                    with open(eval_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry_exp) + "\n")
                except Exception as e:
                    print(f"  ⚠️  Error en eval vs Experto: {e}")
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
                incluir_experto=elo_experto,
            )
            if proc:
                procesos_elo.append((proc, elo_out))
                print(f"  [OK] Torneo ELO en segundo plano (PID {proc.pid})")

        # Procesar torneos ELO terminados y guardar mejores snapshots
        pendientes = []
        for proc, out_file in procesos_elo:
            if proc.poll() is None:
                pendientes.append((proc, out_file))  # Sigue ejecutándose
            else:
                # Torneo terminado → guardar los mejores snapshots
                if proc.returncode == 0:
                    print(
                        f"\n  ✅ Torneo ELO completado → {os.path.basename(out_file)}")
                    guardados = _guardar_mejores_snapshots(
                        out_file, dir_snapshots, best_dir, best_top,
                    )
                    if guardados > 0:
                        print(
                            f"  📁 {guardados} snapshots elite guardados en {best_dir}/")
                else:
                    print(
                        f"\n  ⚠️  Torneo ELO falló (exit code {proc.returncode})")
        procesos_elo = pendientes

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
        for proc, out_file in procesos_elo:
            try:
                proc.wait(timeout=300)
                if proc.returncode == 0:
                    print(
                        f"  ✅ Torneo ELO finalizado → {os.path.basename(out_file)}")
                    _guardar_mejores_snapshots(
                        out_file, dir_snapshots, best_dir, best_top,
                    )
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
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo: cpu, cuda, dml (Intel Arc/DirectML), xpu (Intel XPU)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Reanudar desde snapshot .zip existente")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida para snapshots (default: modelos/v6)")
    parser.add_argument("--best-top", type=int, default=BEST_TOP,
                        help=f"Guardar los N mejores snapshots tras cada torneo Elo (default: {BEST_TOP})")
    parser.add_argument("--obs-dim", type=int, default=220, choices=[194, 220],
                        help="Dimensión del vector de observación (default: 220)")
    parser.add_argument("--prob-experto", type=float, default=0.05,
                        help="Probabilidad de usar BotExperto como oponente (default: 0.05)")
    parser.add_argument("--bc-pretrain", type=str, default=None,
                        help="Ruta al modelo BC preentrenado .zip para inicializar pesos")
    parser.add_argument("--eval-experto", action="store_true", default=False,
                        help="Evaluar también contra 3× BotExperto en cada evaluación")
    parser.add_argument("--eval-experto-partidas", type=int, default=30,
                        help="Partidas contra 3× BotExperto (default: 30)")
    parser.add_argument("--elo-experto", action="store_true", default=False,
                        help="Incluir BotExperto como participante en torneos Elo")
    args = parser.parse_args()

    paso_final = entrenar_auto(
        total_steps=args.total_steps,
        snapshot_every=args.snapshot_every,
        eval_every=args.eval_every,
        elo_every=args.elo_every,
        eval_partidas=args.eval_partidas,
        prob_bot_start=args.prob_bot_start,
        prob_bot_end=args.prob_bot_end,
        prob_experto=args.prob_experto,
        seed=args.seed,
        device=args.device,
        resume_from=args.resume,
        output_dir=args.output_dir,
        best_top=args.best_top,
        obs_dim=args.obs_dim,
        bc_pretrain=args.bc_pretrain,
        eval_experto=args.eval_experto,
        eval_experto_partidas=args.eval_experto_partidas,
        elo_experto=args.elo_experto,
    )
    print(f"\nPaso final alcanzado: {paso_final:,}")


if __name__ == "__main__":
    main()
