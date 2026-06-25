"""
Custom callbacks para el entrenamiento RLlib de Hearts.

Registra métricas del juego (puntos del agente, victorias, shooting moon)
en el resultado de entrenamiento para tracking en TensorBoard.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class HeartsCallbacks(DefaultCallbacks):
    """Callbacks para métricas específicas del juego durante entrenamiento."""

    def on_episode_end(
        self,
        *,
        worker=None,
        base_env=None,
        policies=None,
        episode=None,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Registra métricas al final de cada episodio (mano)."""
        if episode is None:
            return

        info = episode.last_info_for() or {}

        # El env puede incluir info adicional en el último step
        puntos = info.get("puntos_agente", None)
        if puntos is not None:
            episode.custom_metrics["puntos_agente"] = puntos

        gano = info.get("gano_mano", None)
        if gano is not None:
            episode.custom_metrics["gano_mano"] = float(gano)

        pleno = info.get("shooting_moon", False)
        episode.custom_metrics["shooting_moon"] = float(pleno)

    def on_train_result(
        self,
        *,
        algorithm=None,
        result: dict = None,
        **kwargs,
    ) -> None:
        """Logging de alertas diagnósticas al final de cada iteración."""
        if result is None or algorithm is None:
            return

        timesteps = result.get("timesteps_total", 0)
        mean_reward = result.get("episode_reward_mean", float("nan"))
        entropy = result.get("info", {}).get("learner", {}).get(
            "default_policy", {}
        ).get("learner_stats", {}).get("entropy", float("nan"))

        # Escribir métricas en eval_log.jsonl
        log_dir = getattr(algorithm, "_logdir", None) or "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "eval_log.jsonl")

        entrada = {
            "tipo": "metricas",
            "paso": timesteps,
            "reward_medio": mean_reward,
            "entropy": entropy,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada) + "\n")

        # Alertas diagnósticas
        alertas = []
        if isinstance(entropy, float) and not (entropy != entropy):  # not NaN
            if entropy < 0.5:
                alertas.append(f"entropy baja ({entropy:.3f}) — posible colapso de política")
            if entropy > 3.5:
                alertas.append(f"entropy alta ({entropy:.3f}) — posible no convergencia")

        for alerta in alertas:
            entrada_alerta = {
                "tipo": "alerta",
                "paso": timesteps,
                "mensaje": alerta,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada_alerta) + "\n")
