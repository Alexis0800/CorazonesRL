"""
Pipeline de Entrenamiento con Fictitious Self-Play (Módulo 3).

Entrena un agente de Corazones usando MaskablePPO (sb3-contrib) con
Action Masking nativo. El entrenamiento ocurre en dos fases:

    Fase 1 — Entrenamiento contra bots heurísticos:
        El agente RL aprende las reglas básicas jugando contra 3 bots
        basados en reglas (conservador, agresivo, evasivo).

    Fase 2 — Fictitious Self-Play:
        El agente juega contra snapshots históricos de sí mismo,
        cargados aleatoriamente desde el directorio de modelos.
        Se guarda un snapshot cada N snapshots.

Uso:
    python train_self_play.py [--resume RUTA] [--steps N] [--snapshot-every N]
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Asegurar que src está en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------
# Configuración global
# ------------------------------------------------------------------

DIRECTORIO_MODELOS: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "modelos_historicos"
)

DIRECTORIO_LOGS: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs"
)

# Mapeo de asientos a nombres para logging
NOMBRES_ASIENTOS: Dict[int, str] = {
    0: "Norte", 1: "Este", 2: "Sur", 3: "Oeste"}


# ------------------------------------------------------------------
# Adaptador de política SB3 → oponente de CorazonesEnv
# ------------------------------------------------------------------

class PoliticaSB3:
    """Adaptador que envuelve un modelo SB3 como política de oponente.

    Traduce la interfaz de CorazonesEnv (motor, idx, legales) → Carta
    a la interfaz de SB3 (obs, mask) → action, usando la misma
    construcción de observación que el entorno.

    Attributes:
        model: Modelo SB3 cargado (MaskablePPO).
        agente_idx: Índice del jugador que controla este modelo.
        construir_obs_fn: Función que construye la observación de 187 dims.
    """

    def __init__(
        self,
        model: Any,
        agente_idx: int,
        construir_obs_fn: Callable[[int], np.ndarray],
    ) -> None:
        self.model = model
        self.agente_idx = agente_idx
        self.construir_obs = construir_obs_fn

    def __call__(
        self, motor: Any, jugador_idx: int, legales: List[Carta]
    ) -> Carta:
        """Selecciona una carta usando el modelo SB3.

        Args:
            motor: Motor del juego (no utilizado directamente).
            jugador_idx: Índice del jugador.
            legales: Lista de cartas legales.

        Returns:
            Carta seleccionada por el modelo.
        """
        obs = self.construir_obs(self.agente_idx)
        # Construir máscara
        mask = np.zeros(52, dtype=np.bool_)
        for c in legales:
            mask[c.id] = True
        action, _ = self.model.predict(
            obs, action_masks=mask, deterministic=True)
        return Carta._TODAS[int(action)]


# ------------------------------------------------------------------
# Factoría de entornos de entrenamiento
# ------------------------------------------------------------------

def crear_entorno_entrenamiento(
    agente_idx: int = 0,
    politicas: Optional[Dict[int, Callable]] = None,
    seed: Optional[int] = None,
) -> CorazonesEnv:
    """Crea un entorno de entrenamiento con oponentes configurables.

    Args:
        agente_idx: Índice del agente RL (0-3).
        politicas: Diccionario {jugador_idx: callable} con políticas de oponentes.
        seed: Semilla para reproducibilidad.

    Returns:
        Instancia de CorazonesEnv configurada.
    """
    env = CorazonesEnv(agente_idx=agente_idx,
                       politicas_oponentes=politicas or {})
    if seed is not None:
        env.reset(seed=seed)
    return env


def crear_entorno_con_bots(
    agente_idx: int = 0,
    seed: Optional[int] = None,
) -> CorazonesEnv:
    """Crea un entorno donde los oponentes son bots heurísticos.

    Args:
        agente_idx: Índice del agente RL.
        seed: Semilla para reproducibilidad.

    Returns:
        CorazonesEnv con bots conservador, agresivo y evasivo como oponentes.
    """
    bots_disponibles = [bot_conservador, bot_agresivo, bot_evasivo]
    politicas: Dict[int, Callable] = {}
    bot_idx = 0
    for i in range(4):
        if i != agente_idx:
            politicas[i] = bots_disponibles[bot_idx % len(bots_disponibles)]
            bot_idx += 1
    return crear_entorno_entrenamiento(agente_idx, politicas, seed)


# ------------------------------------------------------------------
# Snapshots y Fictitious Self-Play
# ------------------------------------------------------------------

def guardar_snapshot(model: Any, paso: int) -> str:
    """Guarda un snapshot del modelo en el directorio histórico.

    Args:
        model: Modelo SB3 a guardar.
        paso: Número de paso actual (para el nombre del archivo).

    Returns:
        Ruta del archivo guardado.
    """
    os.makedirs(DIRECTORIO_MODELOS, exist_ok=True)
    ruta = os.path.join(DIRECTORIO_MODELOS, f"snapshot_{paso:010d}")
    model.save(ruta)
    print(f"  [Snapshot] Guardado en {ruta}.zip")
    return ruta


def listar_snapshots() -> List[str]:
    """Lista todos los snapshots guardados en el directorio histórico.

    Returns:
        Lista de rutas base (sin extensión .zip) de snapshots disponibles.
    """
    if not os.path.isdir(DIRECTORIO_MODELOS):
        return []
    snapshots = glob.glob(os.path.join(DIRECTORIO_MODELOS, "snapshot_*.zip"))
    # Ordenar por número de paso (extraído del nombre)
    snapshots.sort(
        key=lambda p: int(os.path.basename(p).replace(
            "snapshot_", "").replace(".zip", ""))
    )
    return [p.replace(".zip", "") for p in snapshots]


def cargar_snapshot_aleatorio() -> Optional[Any]:
    """Carga un snapshot aleatorio desde el directorio histórico.

    Returns:
        Modelo SB3 cargado, o None si no hay snapshots.
    """
    snapshots = listar_snapshots()
    if not snapshots:
        return None
    elegido = random.choice(snapshots)
    # Cargar sin entorno (solo para inferencia)
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(elegido)


def crear_entorno_self_play(
    modelo_principal: Any,
    agente_idx: int = 0,
    seed: Optional[int] = None,
) -> CorazonesEnv:
    """Crea un entorno de Self-Play donde los oponentes son snapshots históricos.

    Si no hay suficientes snapshots, completa con bots heurísticos.

    Args:
        modelo_principal: Modelo SB3 del agente principal (no se usa como oponente).
        agente_idx: Índice del agente RL.
        seed: Semilla para reproducibilidad.

    Returns:
        CorazonesEnv con oponentes históricos + bots de respaldo.
    """
    bots_disponibles = [bot_conservador, bot_agresivo, bot_evasivo]
    politicas: Dict[int, Callable] = {}
    bot_idx = 0

    for i in range(4):
        if i == agente_idx:
            continue

        # Intentar cargar un snapshot para este oponente
        modelo_oponente = cargar_snapshot_aleatorio()
        if modelo_oponente is not None:
            # Crear política SB3 para este oponente
            # Necesitamos una referencia al env para construir obs;
            # creamos un env temporal para obtener la función
            env_temp = CorazonesEnv(agente_idx=i)
            politicas[i] = PoliticaSB3(
                modelo_oponente, i, env_temp._construir_observacion
            )
        else:
            # Fallback a bot heurístico
            politicas[i] = bots_disponibles[bot_idx % len(bots_disponibles)]
            bot_idx += 1

    env = crear_entorno_entrenamiento(agente_idx, politicas, seed)
    return env


# ------------------------------------------------------------------
# Bucle principal de entrenamiento
# ------------------------------------------------------------------

def entrenar(
    modelo: Any,
    env: CorazonesEnv,
    total_steps: int,
    snapshot_every: int = 50000,
    inicio_paso: int = 0,
) -> None:
    """Ejecuta el bucle de entrenamiento principal.

    Args:
        modelo: Modelo MaskablePPO a entrenar.
        env: Entorno de entrenamiento configurado.
        total_steps: Número total de pasos a entrenar.
        snapshot_every: Guardar snapshot cada N pasos.
        inicio_paso: Paso inicial (para reanudación).
    """
    steps_restantes = total_steps
    paso_actual = inicio_paso

    while steps_restantes > 0:
        # Entrenar un bloque
        bloque = min(steps_restantes, snapshot_every)
        modelo.learn(
            total_timesteps=bloque,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        paso_actual += bloque
        steps_restantes -= bloque

        # Guardar snapshot
        guardar_snapshot(modelo, paso_actual)
        print(f"  Progreso: {paso_actual}/{inicio_paso + total_steps} pasos")


def main() -> None:
    """Punto de entrada del pipeline de entrenamiento."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento RL para Corazones con Fictitious Self-Play"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Ruta a un checkpoint para reanudar entrenamiento"
    )
    parser.add_argument(
        "--steps", type=int, default=1_000_000,
        help="Número total de pasos de entrenamiento"
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=50_000,
        help="Guardar snapshot cada N pasos"
    )
    parser.add_argument(
        "--self-play", action="store_true",
        help="Usar Fictitious Self-Play (cargar oponentes históricos)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla aleatoria para reproducibilidad"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Dispositivo de cómputo (cpu, cuda)"
    )
    parser.add_argument(
        "--logdir", type=str, default=DIRECTORIO_LOGS,
        help="Directorio para logs de TensorBoard"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pipeline de Entrenamiento — Corazones RL")
    print("=" * 60)
    print(f"  Dispositivo: {args.device}")
    print(f"  Pasos totales: {args.steps:,}")
    print(f"  Snapshot cada: {args.snapshot_every:,}")
    print(f"  Self-Play: {args.self_play}")
    print(f"  Semilla: {args.seed}")
    print(f"  Directorio modelos: {DIRECTORIO_MODELOS}")
    print(f"  Logs TensorBoard: {args.logdir}")
    print("-" * 60)

    # Configurar semillas globales
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("ERROR: sb3-contrib no está instalado.")
        print("  Instálalo con: pip install sb3-contrib")
        sys.exit(1)

    # Crear entorno inicial
    if args.self_play and listar_snapshots():
        print("Modo Self-Play: buscando snapshots históricos...")
        snapshots = listar_snapshots()
        print(f"  Snapshots disponibles: {len(snapshots)}")
    else:
        print("Modo Bots: entrenando contra bots heurísticos...")

    env = crear_entorno_con_bots(agente_idx=0, seed=args.seed)

    # Crear o cargar modelo
    policy_kwargs = obtener_policy_kwargs()

    # Asegurar que el directorio de logs existe
    os.makedirs(args.logdir, exist_ok=True)

    if args.resume and os.path.exists(args.resume + ".zip"):
        print(f"Cargando checkpoint desde {args.resume}.zip ...")
        modelo = MaskablePPO.load(args.resume, env=env, device=args.device)
        # Actualizar tensorboard_log al cargar
        modelo.tensorboard_log = args.logdir
        print("  Checkpoint cargado.")
    else:
        print("Creando nuevo modelo MaskablePPO...")
        modelo = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=args.device,
            tensorboard_log=args.logdir,
        )
        print(f"  Arquitectura: MLP [256, 256, 128]")
        print(f"  Action Masking: nativo (sb3-contrib)")

    print("-" * 60)
    print("Iniciando entrenamiento...")
    print(f"  Monitoriza en tiempo real: tensorboard --logdir {args.logdir}")
    print("  Métrica clave: rollout/ep_rew_mean (Recompensa Media por Episodio)")

    try:
        entrenar(
            modelo,
            env,
            total_steps=args.steps,
            snapshot_every=args.snapshot_every,
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido. Guardando checkpoint...")
        guardar_snapshot(modelo, 0)
        print("Checkpoint guardado. Puedes reanudar con --resume")

    # Guardar modelo final
    ruta_final = os.path.join(DIRECTORIO_MODELOS, "modelo_final")
    modelo.save(ruta_final)
    print(f"Modelo final guardado en {ruta_final}.zip")
    print("Entrenamiento completado.")

    env.close()


if __name__ == "__main__":
    main()
