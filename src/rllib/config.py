"""
PPOConfig builder para entrenamiento de Hearts con Ray RLlib.

Usa el old API stack de RLlib para máxima compatibilidad con
TorchModelV2 (HeartsActionMaskModel).

Uso:
    from src.rllib.config import build_ppo_config
    config = build_ppo_config(opponent_factory=pool.make_factory(0.0))
    algo = config.build_algo()
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from src.entorno.dimensiones import DIM_ENTORNO
from src.rllib.model import HeartsActionMaskModel

# Registrar modelo custom una sola vez al importar el módulo
ModelCatalog.register_custom_model("hearts_model", HeartsActionMaskModel)


def build_ppo_config(
    opponent_factory: Optional[Callable] = None,
    obs_dim: int = DIM_ENTORNO,
    agente_idx: int = 0,
    random_position: bool = True,
    # PPO hiperparámetros
    lr: float = 3e-4,
    lr_end: float = 1e-4,          # piso del LR — nunca decae a cero para que el
    total_steps: int = 20_000_000, # modelo siempre pueda adaptarse a nuevos snapshots
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    # entropy_coeff alto = más exploración. Con recompensas terminales y self-play
    # es importante mantenerlo por encima de 0.01 para que el modelo nunca converja
    # prematuramente a una estrategia local.
    entropy_coeff: float = 0.015,
    vf_coef: float = 0.25,
    grad_clip: float = 0.5,
    train_batch_size: int = 4096,
    sgd_minibatch_size: int = 512,
    num_sgd_iter: int = 10,
    # Arquitectura
    fcnet_hiddens: List[int] = None,
    # Recursos
    num_rollout_workers: int = 2,
    num_gpus: int = 0,
) -> PPOConfig:
    """Construye la configuración PPO para Hearts.

    Args:
        opponent_factory: callable(agente_idx) -> dict[int, policy_fn].
                          Si es None, se usan bots evasivos por defecto.
        obs_dim:          Dimensión del vector de observación.
        agente_idx:       Índice base del jugador agente (0-3). Ignorado si
                          random_position=True.
        random_position:  Si True, el env sortea agente_idx en cada episodio,
                          entrenando el modelo desde las 4 posiciones de la mesa.
        lr / lr_end:      LR inicial y piso final. Decae linealmente de lr a lr_end
                          durante total_steps. El piso impide que el modelo se "congele"
                          cuando llegan snapshots nuevos al pool de oponentes.
        entropy_coeff:    Coeficiente de entropía para exploración continua.
                          Mantener ≥0.01 para que el modelo nunca deje de explorar.

    Returns:
        PPOConfig lista para .build_algo().
    """
    if fcnet_hiddens is None:
        fcnet_hiddens = [512, 512, 256]

    # Schedule lineal: lr → lr_end a lo largo del entrenamiento.
    # No usamos cosine decay porque decaería a near-zero e impediría adaptarse
    # a nuevos oponentes cuando el pool de snapshots crece.
    lr_schedule = [[0, lr], [total_steps, lr_end]]

    env_config = {
        "obs_dim": obs_dim,
        "agente_idx": agente_idx,
        "random_position": random_position,
        "opponent_factory": opponent_factory,
    }

    config = (
        PPOConfig()
        # ── Old API stack (TorchModelV2 + ModelCatalog) ──────────────────
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        # ── Entorno ──────────────────────────────────────────────────────
        .environment(
            env="src.entorno.corazones_rllib.CorazonesEnvRLlib",
            env_config=env_config,
        )
        # ── Framework ────────────────────────────────────────────────────
        .framework("torch")
        # ── Modelo ───────────────────────────────────────────────────────
        .training(
            model={
                "custom_model": "hearts_model",
                "fcnet_hiddens": fcnet_hiddens,
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            },
            lr=lr,                  # float — para torch.optim.Adam al inicializar
            lr_schedule=lr_schedule, # lista [[step, lr]] — RLlib lo aplica en cada update
            gamma=gamma,
            lambda_=lambda_,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=vf_coef,
            grad_clip=grad_clip,
            train_batch_size=train_batch_size,
            minibatch_size=sgd_minibatch_size,
            num_epochs=num_sgd_iter,
        )
        # ── Env runners ───────────────────────────────────────────────────
        .env_runners(
            num_env_runners=num_rollout_workers,
            # preprocessor_pref=None desactiva el preprocesado para Dict spaces
            preprocessor_pref=None,
        )
        # ── Recursos ─────────────────────────────────────────────────────
        .resources(
            num_gpus=num_gpus,
        )
        # ── Evaluación ───────────────────────────────────────────────────
        .evaluation(
            evaluation_interval=None,
        )
    )

    return config
