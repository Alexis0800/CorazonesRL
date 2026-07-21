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

from typing import Callable, List, Optional

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from src.entorno.dimensiones import DIM_ENTORNO
from src.rllib.model import HeartsActionMaskModel, HeartsLSTMModel

# Registrar modelos custom una sola vez al importar el módulo
ModelCatalog.register_custom_model("hearts_model", HeartsActionMaskModel)
ModelCatalog.register_custom_model("hearts_lstm_model", HeartsLSTMModel)


def build_ppo_config(
    opponent_factory: Optional[Callable] = None,
    obs_dim: int = DIM_ENTORNO,
    agente_idx: int = 0,
    random_position: bool = True,
    reward_config=None,
    con_pase: bool = False,
    moon_dir: Optional[str] = None,  # dir de pesos moon (propio.pt/rival.pt); None = "models/moon"
    # PPO hiperparámetros
    lr: float = 3e-4,
    lr_end: float = 1e-4,          # piso del LR — nunca decae a cero para que el
    total_steps: int = 20_000_000, # modelo siempre pueda adaptarse a nuevos snapshots
    # gamma alto: el episodio es una PARTIDA COMPLETA (~100-170 steps); el puesto
    # final debe propagar hacia atrás. DEBE coincidir con el gamma del env (PBRS).
    gamma: float = 0.999,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    # entropy_coeff alto = más exploración. Con recompensas terminales y self-play
    # es importante mantenerlo por encima de 0.01 para que el modelo nunca converja
    # prematuramente a una estrategia local.
    entropy_coeff: float = 0.05,
    vf_coef: float = 0.25,
    grad_clip: float = 0.5,
    train_batch_size: int = 4096,
    sgd_minibatch_size: int = 512,
    num_sgd_iter: int = 10,
    # Arquitectura
    fcnet_hiddens: List[int] = None,
    use_lstm: bool = False,
    lstm_hidden_size: int = 256,
    # Recursos
    num_rollout_workers: int = 2,
    num_gpus: int = 0,
) -> PPOConfig:
    """Construye la configuración PPO para Hearts.

    Args:
        opponent_factory:   callable(agente_idx) -> dict[int, policy_fn].
                            Si es None, se usan bots evasivos por defecto.
        obs_dim:            Dimensión del vector de observación.
        agente_idx:         Índice base del jugador agente (0-3). Ignorado si
                            random_position=True.
        random_position:    Si True, el env sortea agente_idx en cada episodio,
                            entrenando el modelo desde las 4 posiciones de la mesa.
        lr / lr_end:        LR inicial y piso final. Decae linealmente durante total_steps.
        entropy_coeff:      Coeficiente de entropía. Mantener ≥0.01 para exploración.
        use_lstm:           Si True, usa HeartsLSTMModel en lugar del MLP estándar.
        lstm_hidden_size:   Tamaño del estado oculto del LSTM (default 256).

    Returns:
        PPOConfig lista para .build_algo().
    """
    if fcnet_hiddens is None:
        fcnet_hiddens = [512, 512, 256]

    # Schedule lineal: lr → lr_end a lo largo del entrenamiento.
    lr_schedule = [[0, lr], [total_steps, lr_end]]

    env_config = {
        "obs_dim": obs_dim,
        "agente_idx": agente_idx,
        "random_position": random_position,
        "opponent_factory": opponent_factory,
        # gamma del shaping PBRS — debe ser idéntico al gamma de PPO.
        "gamma": gamma,
        # v10b: fase de pase (requiere obs_dim >= 228).
        "con_pase": con_pase,
    }
    if reward_config is not None:
        env_config["reward_config"] = reward_config
    if moon_dir is not None:
        env_config["moon_dir"] = moon_dir

    if use_lstm:
        model_config = {
            "custom_model": "hearts_lstm_model",
            "lstm_cell_size": lstm_hidden_size,
            # v10: el episodio es una PARTIDA COMPLETA (~130 steps, ~10 manos).
            # max_seq_len es la ventana de BPTT; debe cruzar varias manos para que
            # la LSTM aprenda a modelar patrones de rivales ENTRE manos (no solo
            # dentro de una). 52 ≈ 4 manos de contexto. El estado oculto se arrastra
            # toda la partida; esto solo limita hasta dónde fluye el gradiente.
            "max_seq_len": 52,
            "fcnet_activation": "relu",
            "vf_share_layers": False,
        }
    else:
        model_config = {
            "custom_model": "hearts_model",
            "fcnet_hiddens": fcnet_hiddens,
            "fcnet_activation": "relu",
            "vf_share_layers": False,
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
            model=model_config,
            lr=lr,                   # float — para torch.optim.Adam al inicializar
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
