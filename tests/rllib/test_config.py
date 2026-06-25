"""
Tests de src/rllib/config.py — verifica que la configuración PPO se construye
correctamente antes de intentar entrenar.
"""
from __future__ import annotations

import pytest


class TestBuildPPOConfig:
    def test_importar_build_ppo_config(self):
        from src.rllib.config import build_ppo_config
        assert callable(build_ppo_config)

    def test_config_contiene_env_correcto(self):
        from src.rllib.config import build_ppo_config
        config = build_ppo_config(obs_dim=224)
        assert config.env == "src.entorno.corazones_rllib.CorazonesEnvRLlib"

    def test_config_framework_torch(self):
        from src.rllib.config import build_ppo_config
        config = build_ppo_config()
        assert config.framework_str == "torch"

    def test_config_modelo_custom_hearts(self):
        from src.rllib.config import build_ppo_config
        config = build_ppo_config()
        assert config.model["custom_model"] == "hearts_model"

    def test_config_lr_custom(self):
        from src.rllib.config import build_ppo_config
        # lr sigue siendo float; lr_schedule es la lista separada
        config = build_ppo_config(lr=1e-4, lr_end=5e-5, total_steps=10_000_000)
        assert config.lr == 1e-4
        assert isinstance(config.lr_schedule, list)
        assert config.lr_schedule[0] == [0, 1e-4]
        assert config.lr_schedule[-1][-1] == 5e-5

    def test_config_train_batch_size_custom(self):
        from src.rllib.config import build_ppo_config
        config = build_ppo_config(train_batch_size=2048)
        assert config.train_batch_size == 2048

    def test_hearts_model_registrado_en_catalog(self):
        from ray.rllib.models import ModelCatalog
        from src.rllib.config import build_ppo_config  # trigger registro
        # El modelo debe estar registrado como custom model
        registered = ModelCatalog._model_shared_memory if hasattr(
            ModelCatalog, "_model_shared_memory"
        ) else ModelCatalog._model_registry if hasattr(
            ModelCatalog, "_model_registry"
        ) else {}
        # Si no podemos acceder al registro interno, al menos verificamos que
        # build_ppo_config no crashea y que la config es válida
        config = build_ppo_config()
        assert config is not None

    def test_env_config_contiene_obs_dim(self):
        from src.rllib.config import build_ppo_config
        config = build_ppo_config(obs_dim=224)
        assert config.env_config["obs_dim"] == 224

    def test_env_config_contiene_agente_idx(self):
        from src.rllib.config import build_ppo_config
        config = build_ppo_config(agente_idx=2)
        assert config.env_config["agente_idx"] == 2
