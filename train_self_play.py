"""
Pipeline de Entrenamiento con Self-Play real (v4).

CORRECCIÓN: El modelo se estancó en ~47% porque entrenaba contra 3 bots fijos.
Self-Play real entrena contra snapshots históricos del propio agente, creando
un currículum de dificultad creciente.

Fase 1 (hecha): 3.6M pasos contra bots → ~47% win rate
Fase 2 (AHORA): Self-Play contra los 74 snapshots existentes

VecNormalize: Cada snapshot guarda sus propias stats de normalización.
Al cargar un snapshot con MaskablePPO.load(), SB3 restaura sus stats.
Al llamar model.predict(obs), las observaciones se normalizan automáticamente.

Uso:
    python train_self_play.py --self-play --steps 5000000 --snapshot-every 50000
"""

from __future__ import annotations
from src.red import obtener_policy_kwargs
from src.carta import Carta
from src.bots import bot_conservador, bot_agresivo, bot_evasivo
from src.entorno import CorazonesEnv

import os
import sys
import glob
import random
import argparse
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# Configuración global
# ------------------------------------------------------------------
DIRECTORIO_MODELOS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "modelos_historicos")
DIRECTORIO_MODELOS_V2 = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "modelos_historicos", "v2")
DIRECTORIO_LOGS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs")
DIRECTORIO_VECNORM = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "vecnormalize")

# ------------------------------------------------------------------
# Constantes v2 (anti-colapso de política)
# ------------------------------------------------------------------
PROB_BOT_V2: float = 0.40          # 40% bots heurísticos como anclaje
MIN_SNAPSHOT_STEPS: int = 200_000  # Umbral mínimo de pasos para usar un snapshot
MAX_SNAPSHOTS_POOL: int = 25       # Máximo de snapshots en el pool activo


# ------------------------------------------------------------------
# Política SB3 con normalización de observaciones
# ------------------------------------------------------------------
class PoliticaSB3:
    """Adaptador que envuelve un snapshot como política de oponente.

    El modelo se cargó con MaskablePPO.load() que restaura sus stats de
    VecNormalize. Al llamar predict(), las observaciones se normalizan
    automáticamente usando esas stats.
    """

    def __init__(self, model: Any, agente_idx: int, vecnorm_path: Optional[str] = None):
        self.model = model
        self.agente_idx = agente_idx
        self._obs_rms = None

        # Cargar stats de VecNormalize asociadas al snapshot
        if vecnorm_path and os.path.exists(vecnorm_path):
            try:
                with open(vecnorm_path, "rb") as f:
                    data = pickle.load(f)
                self._obs_rms = data.get("obs_rms", None)
            except Exception:
                self._obs_rms = None

    def __call__(self, motor: Any, jugador_idx: int, legales: List[Carta]) -> Carta:
        # Construir observación desde perspectiva de este oponente
        env = motor  # El motor expone suficiente info
        obs = self._construir_obs_desde_motor(motor, jugador_idx)

        # Normalizar si tenemos stats
        if self._obs_rms is not None:
            mean = np.array(self._obs_rms.mean)
            var = np.array(self._obs_rms.var)
            obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -
                          10.0, 10.0).astype(np.float32)

        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True
        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True)
        return Carta._TODAS[int(action)]

    def _construir_obs_desde_motor(self, motor: Any, jugador_idx: int) -> np.ndarray:
        """Construye observación de 187 dims desde la perspectiva de jugador_idx."""
        obs = np.zeros(187, dtype=np.float32)
        a = jugador_idx

        # Mano del jugador
        for c in motor.jugadores[a].mano:
            obs[c.id] = 1.0

        # Mesa actual
        for _, c in motor.mesa:
            obs[52 + c.id] = 1.0

        # Bazas ganadas (cementerio)
        for i in range(4):
            for c in motor.jugadores[i].bazas_ganadas:
                obs[104 + c.id] = 1.0

        # Vacíos (no disponibles desde el motor)
        # Puntajes históricos (no disponibles desde el motor)
        return obs


# ------------------------------------------------------------------
# Factoría de entornos
# ------------------------------------------------------------------
def crear_entorno_con_bots(agente_idx=0, seed=None, shuffle_bots=True):
    bots = [bot_conservador, bot_agresivo, bot_evasivo]
    if shuffle_bots:
        random.shuffle(bots)
    politicas = {}
    bot_idx = 0
    for i in range(4):
        if i != agente_idx:
            politicas[i] = bots[bot_idx % len(bots)]
            bot_idx += 1
    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


def listar_snapshots():
    if not os.path.isdir(DIRECTORIO_MODELOS):
        return []
    snaps = glob.glob(os.path.join(DIRECTORIO_MODELOS, "snapshot_*.zip"))
    snaps.sort(key=lambda p: int(os.path.basename(
        p).replace("snapshot_", "").replace(".zip", "")))
    return [p.replace(".zip", "") for p in snaps]


def listar_snapshots_v2() -> List[str]:
    """Lista snapshots del directorio v2, ordenados por paso de entrenamiento.

    Returns:
        Lista de rutas absolutas sin extensión .zip.
    """
    os.makedirs(DIRECTORIO_MODELOS_V2, exist_ok=True)
    snaps = glob.glob(os.path.join(DIRECTORIO_MODELOS_V2, "snapshot_*.zip"))
    snaps.sort(key=lambda p: int(os.path.basename(
        p).replace("snapshot_", "").replace(".zip", "")))
    return [p.replace(".zip", "") for p in snaps]


def _extraer_paso_de_ruta(ruta_snapshot: str) -> int:
    """Extrae el número de paso de una ruta de snapshot.

    Args:
        ruta_snapshot: Ruta como '.../snapshot_0001500000' o '.../snapshot_0001500000.zip'.

    Returns:
        Número de paso (int), o 0 si no se puede extraer.
    """
    nombre = os.path.basename(ruta_snapshot).replace(".zip", "")
    try:
        return int(nombre.replace("snapshot_", ""))
    except ValueError:
        return 0


def _filtrar_snapshots_por_calidad(
    snapshots: List[str],
    min_steps: int = MIN_SNAPSHOT_STEPS,
) -> List[str]:
    """Filtra snapshots que cumplen el umbral mínimo de pasos de entrenamiento.

    Previene el colapso de política excluyendo snapshots demasiado tempranos
    que representan políticas inmaduras o aleatorias.

    Args:
        snapshots: Lista de rutas de snapshots (sin .zip).
        min_steps: Pasos mínimos de entrenamiento requeridos.

    Returns:
        Lista filtrada de snapshots que cumplen el umbral.
    """
    if not snapshots:
        return []
    return [
        s for s in snapshots
        if _extraer_paso_de_ruta(s) >= min_steps
    ]


def crear_entorno_self_play(agente_idx=0, seed=None, prob_bot=0.15):
    """Crea entorno Self-Play: 85% snapshots, 15% bots."""
    snapshots = listar_snapshots()
    bots = [bot_conservador, bot_agresivo, bot_evasivo]
    random.shuffle(bots)
    politicas = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        usar_bot = random.random() < prob_bot or len(snapshots) < 2

        if not usar_bot:
            # Elegir snapshot aleatorio con peso hacia recientes
            pesos = np.exp(np.linspace(0, 2, len(snapshots)))
            pesos /= pesos.sum()
            snap_elegido = np.random.choice(snapshots, p=pesos)

            from sb3_contrib import MaskablePPO
            try:
                modelo_oponente = MaskablePPO.load(snap_elegido, device="cpu")
                vecnorm_path = snap_elegido + "_vecnorm.pkl"
                politicas[i] = PoliticaSB3(modelo_oponente, i, vecnorm_path)
                continue
            except Exception:
                pass  # Fallback a bot

        politicas[i] = bots[bot_idx % len(bots)]
        bot_idx += 1

    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


def crear_entorno_self_play_v2(
    agente_idx: int = 0,
    seed: Optional[int] = None,
    prob_bot: float = PROB_BOT_V2,
    min_snapshot_steps: int = MIN_SNAPSHOT_STEPS,
) -> CorazonesEnv:
    """Crea entorno Self-Play v2 con quality-filter y mayor anclaje a bots.

    Cambios respecto a v1:
        - Solo usa snapshots con >= min_snapshot_steps pasos (quality filter).
        - 40% de oponentes son bots heurísticos (vs 15% en v1).
        - Los snapshots se eligen con pesos exponenciales hacia los más
          recientes (políticas más maduras).
        - Si no hay snapshots de calidad, todos los oponentes son bots.

    Args:
        agente_idx: Índice del agente controlado por RL (0-3).
        seed: Semilla aleatoria opcional.
        prob_bot: Probabilidad de usar un bot heurístico como oponente.
        min_snapshot_steps: Pasos mínimos para considerar un snapshot.

    Returns:
        Entorno CorazonesEnv configurado con oponentes mixtos.
    """
    todos_snapshots = listar_snapshots_v2()
    snapshots = _filtrar_snapshots_por_calidad(
        todos_snapshots, min_snapshot_steps)

    bots = [bot_conservador, bot_agresivo, bot_evasivo]
    random.shuffle(bots)
    politicas: Dict[int, object] = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        # Usar bot si: probabilidad dicta O no hay suficientes snapshots de calidad
        usar_bot = random.random() < prob_bot or len(snapshots) < 2

        if not usar_bot:
            # Pesos exponenciales: favorecer snapshots más recientes (mayor paso)
            pasos = np.array([_extraer_paso_de_ruta(s) for s in snapshots])
            pesos = np.exp(np.linspace(0, 2, len(snapshots)))
            pesos /= pesos.sum()

            snap_elegido = np.random.choice(snapshots, p=pesos)

            from sb3_contrib import MaskablePPO
            try:
                modelo_oponente = MaskablePPO.load(snap_elegido, device="cpu")
                vecnorm_path = snap_elegido + "_vecnorm.pkl"
                politicas[i] = PoliticaSB3(modelo_oponente, i, vecnorm_path)
                continue
            except Exception:
                pass  # Fallback a bot si falla la carga

        politicas[i] = bots[bot_idx % len(bots)]
        bot_idx += 1

    env = CorazonesEnv(agente_idx=agente_idx, politicas_oponentes=politicas)
    if seed is not None:
        env.reset(seed=seed)
    return env


# ------------------------------------------------------------------
# VecNormalize
# ------------------------------------------------------------------
def crear_entorno_vecnormalizado(env_base, vecnorm_path=None):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    venv = DummyVecEnv([lambda: env_base])
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        print(f"  VecNormalize cargado de {vecnorm_path}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.995, epsilon=1e-8)
        print("  VecNormalize nuevo (norm_obs=True, norm_reward=True)")
    return venv


# ------------------------------------------------------------------
# Hiperparámetros PPO
# ------------------------------------------------------------------
def obtener_hiperparametros_ppo(logdir, device):
    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": 3e-5,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.15,
        "normalize_advantage": True,
        "ent_coef": 0.05,
        "vf_coef": 1.0,
        "max_grad_norm": 0.5,
        "target_kl": 0.02,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
    }


def obtener_hiperparametros_v2(logdir: str, device: str) -> Dict:
    """Hiperparámetros PPO optimizados para prevenir colapso de política.

    Cambios respecto a v1:
        - learning_rate: 3e-5 → 1e-4 (reactivar aprendizaje).
        - ent_coef: 0.05 → 0.08 (forzar más exploración).
        - n_epochs: 10 → 8 (reducir sobreajuste por batch).
        - max_grad_norm: 0.5 → 1.0 (permitir gradientes más grandes).

    Args:
        logdir: Directorio para logs de TensorBoard.
        device: Dispositivo de cómputo ('cpu' o 'cuda').

    Returns:
        Diccionario con hiperparámetros para MaskablePPO.
    """
    policy_kwargs = obtener_policy_kwargs()
    return {
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": 8,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.15,
        "normalize_advantage": True,
        "ent_coef": 0.08,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "target_kl": 0.02,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "device": device,
        "tensorboard_log": logdir,
    }


def _aplicar_snapshot_pruning(
    directorio: str,
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
) -> int:
    """Elimina los snapshots más antiguos si el pool excede el máximo.

    Mantiene los snapshots con mayor número de paso (más recientes).
    Los snapshots eliminados son los de menor paso (más antiguos/inmaduros).

    Args:
        directorio: Directorio donde se almacenan los snapshots .zip.
        max_snapshots: Número máximo de snapshots a conservar.

    Returns:
        Número de snapshots eliminados.
    """
    snaps = glob.glob(os.path.join(directorio, "snapshot_*.zip"))
    if len(snaps) <= max_snapshots:
        return 0

    # Ordenar por paso (mayor primero) y eliminar los excedentes
    snaps.sort(key=lambda p: _extraer_paso_de_ruta(p), reverse=True)
    eliminados = 0
    for snap in snaps[max_snapshots:]:
        try:
            os.remove(snap)
            # También eliminar vecnorm asociado si existe
            vecnorm_file = snap.replace(".zip", "_vecnorm.pkl")
            if os.path.exists(vecnorm_file):
                os.remove(vecnorm_file)
            eliminados += 1
        except OSError:
            pass
    return eliminados


# ------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------
def entrenar(
    modelo,
    venv,
    env_fn,
    total_steps: int,
    snapshot_every: int = 50000,
    inicio_paso: int = 0,
    vecnorm_path: str = "",
    directorio_snapshots: str = "",
    max_snapshots: int = MAX_SNAPSHOTS_POOL,
):
    """Bucle de entrenamiento con snapshots periódicos y pruning automático.

    Args:
        modelo: Instancia de MaskablePPO.
        venv: Entorno vectorizado con VecNormalize.
        env_fn: Función factoría que crea un nuevo entorno base.
        total_steps: Pasos totales a entrenar en esta sesión.
        snapshot_every: Guardar snapshot cada N pasos.
        inicio_paso: Paso global de inicio (para nombrar snapshots).
        vecnorm_path: Ruta para guardar VecNormalize.
        directorio_snapshots: Directorio donde guardar snapshots.
        max_snapshots: Máximo de snapshots a conservar (pruning).
    """
    if not directorio_snapshots:
        directorio_snapshots = DIRECTORIO_MODELOS_V2

    os.makedirs(directorio_snapshots, exist_ok=True)

    steps_restantes = total_steps
    paso_actual = inicio_paso

    while steps_restantes > 0:
        bloque = min(steps_restantes, snapshot_every)
        modelo.learn(
            total_timesteps=bloque,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        paso_actual += bloque
        steps_restantes -= bloque

        # Guardar VecNormalize
        if vecnorm_path:
            os.makedirs(os.path.dirname(vecnorm_path), exist_ok=True)
            venv.save(vecnorm_path)

        # Guardar snapshot v2
        nombre = f"snapshot_{paso_actual:010d}"
        ruta = os.path.join(directorio_snapshots, nombre)
        modelo.save(ruta)
        print(
            f"  [Snapshot] {ruta}.zip | "
            f"Progreso: {paso_actual}/{inicio_paso + total_steps}"
        )

        # Pruning: eliminar snapshots antiguos si exceden el máximo
        eliminados = _aplicar_snapshot_pruning(
            directorio_snapshots, max_snapshots)
        if eliminados > 0:
            print(
                f"  [Pruning] {eliminados} snapshot(s) antiguo(s) eliminado(s)")

    return paso_actual


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Self-Play Corazones RL v5")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None,
                        help="Modelo base para iniciar entrenamiento")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--snapshot-every", type=int, default=100_000)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--prob-bot", type=float, default=PROB_BOT_V2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--logdir", type=str, default=DIRECTORIO_LOGS)
    parser.add_argument("--v2", action="store_true", default=True,
                        help="Usar pipeline v2 con quality-filter y anti-colapso")
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Play Corazones RL v5 (Anti-Colapso)")
    print("=" * 60)
    print(f"  Self-Play: {args.self_play}")
    print(f"  Pasos: {args.steps:,}")
    print(f"  Snapshot cada: {args.snapshot_every:,}")
    print(f"  prob_bot: {args.prob_bot:.0%}")
    print(f"  Quality filter min steps: {MIN_SNAPSHOT_STEPS:,}")
    print(f"  Max snapshots pool: {MAX_SNAPSHOTS_POOL}")
    print(f"  Directorio v2: {DIRECTORIO_MODELOS_V2}")
    print("-" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)

    from sb3_contrib import MaskablePPO

    os.makedirs(DIRECTORIO_MODELOS_V2, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(DIRECTORIO_VECNORM, exist_ok=True)

    vecnorm_path = os.path.join(DIRECTORIO_VECNORM, "v2_vecnorm.pkl")
    inicio_paso = 0

    # --- Determinar el modelo base ---
    if args.resume:
        ruta_modelo = args.resume if args.resume.endswith(
            ".zip") else args.resume + ".zip"
        print(f"Reanudando desde: {ruta_modelo}")
        inicio_paso = _extraer_paso_de_ruta(ruta_modelo)
    elif args.base_model:
        ruta_modelo = args.base_model if args.base_model.endswith(
            ".zip") else args.base_model + ".zip"
        print(f"Modelo base: {ruta_modelo}")
        inicio_paso = _extraer_paso_de_ruta(ruta_modelo)
    else:
        # Usar el snapshot más reciente del directorio v2
        snaps_v2 = listar_snapshots_v2()
        if snaps_v2:
            ruta_modelo = snaps_v2[-1] + ".zip"
            inicio_paso = _extraer_paso_de_ruta(ruta_modelo)
            print(f"Reanudando desde último snapshot v2: {ruta_modelo}")
        else:
            # Buscar en directorio viejo
            snaps_old = listar_snapshots()
            if snaps_old:
                ruta_modelo = snaps_old[-1] + ".zip"
                inicio_paso = _extraer_paso_de_ruta(ruta_modelo)
                print(f"Usando snapshot legacy: {ruta_modelo}")
            else:
                print("ERROR: No hay snapshots. Especifica --base-model")
                sys.exit(1)

    if not os.path.exists(ruta_modelo):
        print(f"ERROR: No se encuentra {ruta_modelo}")
        sys.exit(1)

    # --- Modo de entrenamiento ---
    if args.self_play:
        snaps_v2 = listar_snapshots_v2()
        snaps_calidad = _filtrar_snapshots_por_calidad(
            snaps_v2, MIN_SNAPSHOT_STEPS)
        if len(snaps_calidad) >= 2:
            print(
                f"Modo SELF-PLAY v2: {len(snaps_calidad)} snapshots de calidad")
            def env_fn(): return crear_entorno_self_play_v2(
                seed=None, prob_bot=args.prob_bot)
        else:
            print(
                f"Modo BOTS (snapshots de calidad insuficientes: {len(snaps_calidad)})")

            def env_fn(): return crear_entorno_con_bots(seed=None, shuffle_bots=True)
    else:
        print("Modo BOTS: entrenando contra heurísticos")
        def env_fn(): return crear_entorno_con_bots(seed=None, shuffle_bots=True)

    env_base = env_fn()

    # --- VecNormalize ---
    vecnorm_snap = ruta_modelo.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_snap):
        venv = crear_entorno_vecnormalizado(env_base, vecnorm_snap)
        vecnorm_path = vecnorm_snap
        print(f"  VecNormalize cargado de snapshot: {vecnorm_snap}")
    elif os.path.exists(vecnorm_path):
        venv = crear_entorno_vecnormalizado(env_base, vecnorm_path)
        print(f"  VecNormalize cargado de: {vecnorm_path}")
    else:
        venv = crear_entorno_vecnormalizado(env_base)
        print("  VecNormalize nuevo (desde cero)")

    # --- Cargar modelo ---
    print(f"Cargando modelo: {ruta_modelo}")
    modelo = MaskablePPO.load(
        ruta_modelo,
        env=venv,
        device=args.device,
        tensorboard_log=args.logdir,
    )

    # --- Aplicar hiperparámetros v2 ---
    hp = obtener_hiperparametros_v2(args.logdir, args.device)
    for key in ["learning_rate", "ent_coef", "vf_coef", "gamma",
                "gae_lambda", "target_kl", "max_grad_norm",
                "n_steps", "batch_size", "n_epochs"]:
        setattr(modelo, key, hp[key])

    # Corregir: setattr no actualiza el optimizador de PyTorch.
    # Sin esto, el lr real sigue siendo el del modelo cargado (3e-5).
    if hasattr(modelo.policy, "optimizer") and modelo.policy.optimizer is not None:
        for param_group in modelo.policy.optimizer.param_groups:
            param_group["lr"] = hp["learning_rate"]
        print(f"  [OK] Optimizer lr actualizado a {hp['learning_rate']}")

    print(f"  lr={modelo.learning_rate}, ent_coef={modelo.ent_coef}")
    print(f"  vf_coef={modelo.vf_coef}, n_epochs={modelo.n_epochs}")
    print(f"  max_grad_norm={modelo.max_grad_norm}")
    print("-" * 60)
    print("Métricas clave:")
    print("  train/value_loss           -> <1.0 (normalizado)")
    print("  train/entropy_loss         -> Negativo = explorando")
    print("  train/explained_variance   -> Subiendo hacia >0.8")
    print("  rollout/ep_rew_mean        -> Subiendo")
    print("-" * 60)

    try:
        paso_final = entrenar(
            modelo, venv, env_fn,
            args.steps,
            snapshot_every=args.snapshot_every,
            inicio_paso=inicio_paso,
            vecnorm_path=vecnorm_path,
            directorio_snapshots=DIRECTORIO_MODELOS_V2,
            max_snapshots=MAX_SNAPSHOTS_POOL,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido. Guardando...")
        if vecnorm_path:
            venv.save(vecnorm_path)
        modelo.save(os.path.join(DIRECTORIO_MODELOS_V2, "modelo_final_v2"))
        print("Modelo guardado.")
        env_base.close()
        return

    # Guardado final
    modelo.save(os.path.join(DIRECTORIO_MODELOS_V2, "modelo_final_v2"))
    if vecnorm_path:
        venv.save(os.path.join(DIRECTORIO_VECNORM, "v2_vecnorm_final.pkl"))
    print(f"Entrenamiento v2 completado. Paso final: {paso_final}")
    env_base.close()


if __name__ == "__main__":
    main()
