"""
Custom callbacks para el entrenamiento RLlib de Hearts — PARTIDA COMPLETA (v10).

Registra métricas de partida (puesto final, win-rate, top-2, manos por partida,
shooting moon) y un chequeo anti-farming: el reward terminal (R_terminal) frente
al reward total. Si el agente "farmeara" el shaping, el reward total se despegaría
del terminal; con PBRS deberían moverse juntos.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class HeartsCallbacks(DefaultCallbacks):
    """Callbacks para métricas específicas de partida completa durante entrenamiento."""

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
        """Registra métricas al final de cada episodio (= una partida completa)."""
        if episode is None:
            return

        info = episode.last_info_for() or {}

        puesto = info.get("puesto", None)
        if puesto is not None:
            episode.custom_metrics["puesto"] = float(puesto)
            episode.custom_metrics["win_rate"] = float(info.get("gano_partida", False))
            episode.custom_metrics["top2_rate"] = float(info.get("top2", False))

        manos = info.get("manos_jugadas", None)
        if manos is not None:
            episode.custom_metrics["manos_por_partida"] = float(manos)

        r_terminal = info.get("r_terminal", None)
        if r_terminal is not None:
            episode.custom_metrics["r_terminal"] = float(r_terminal)

        episode.custom_metrics["shooting_moon"] = float(info.get("shooting_moon", False))

    def on_train_result(
        self,
        *,
        algorithm=None,
        result: dict = None,
        **kwargs,
    ) -> None:
        """Logging de métricas y alertas diagnósticas al final de cada iteración."""
        if result is None or algorithm is None:
            return

        timesteps = result.get("timesteps_total", 0)
        env_r = result.get("env_runners", {})
        mean_reward = env_r.get("episode_reward_mean", float("nan"))
        custom = env_r.get("custom_metrics", {})
        learner_stats = result.get("info", {}).get("learner", {}).get(
            "default_policy", {}
        ).get("learner_stats", {})
        entropy = learner_stats.get("entropy", float("nan"))
        # Telemetría de learner (plan_siguiente_iteracion 2026-07-25 §A1):
        # sin esto el KL adaptativo y la salud del value head son CIEGOS —
        # todas las keys las expone gratis el old-stack de Ray 2.55.1.
        telemetria_learner = {
            k: learner_stats.get(k, float("nan"))
            for k in ("kl", "cur_kl_coeff", "vf_loss", "vf_explained_var",
                      "policy_loss", "grad_gnorm", "cur_lr")
        }

        win_rate = custom.get("win_rate_mean", float("nan"))
        top2_rate = custom.get("top2_rate_mean", float("nan"))
        puesto = custom.get("puesto_mean", float("nan"))
        r_terminal = custom.get("r_terminal_mean", float("nan"))
        manos = custom.get("manos_por_partida_mean", float("nan"))

        log_dir = getattr(self.__class__, "_log_dir", None) or "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "eval_log.jsonl")

        entrada = {
            "tipo": "metricas",
            "paso": timesteps,
            "reward_medio": mean_reward,
            "entropy": entropy,
            "win_rate": win_rate,
            "top2_rate": top2_rate,
            "puesto_medio": puesto,
            "r_terminal_medio": r_terminal,
            "manos_por_partida": manos,
            **telemetria_learner,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada) + "\n")

        # --- Alertas diagnósticas ---
        alertas = []
        if isinstance(entropy, float) and entropy == entropy:  # not NaN
            if entropy < 0.4:
                alertas.append(f"entropy baja ({entropy:.3f}) — posible colapso de política")
            if entropy > 3.5:
                alertas.append(f"entropy alta ({entropy:.3f}) — posible no convergencia")

        # Anti-farming: el reward total no debería despegarse del terminal.
        # Con PBRS el shaping neto ≈ 0, así que reward_medio ≈ r_terminal_medio.
        if (isinstance(mean_reward, float) and mean_reward == mean_reward
                and isinstance(r_terminal, float) and r_terminal == r_terminal):
            brecha = abs(mean_reward - r_terminal)
            if brecha > 0.5:
                alertas.append(
                    f"brecha reward-terminal alta ({brecha:.3f}) — el shaping podría "
                    f"estar dominando (revisar PHI_LAMBDA)"
                )

        for alerta in alertas:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "tipo": "alerta", "paso": timesteps, "mensaje": alerta,
                }) + "\n")
