"""Tests for configuration system."""

import os
import tempfile

import pytest
import yaml

from src.config import (
    Config,
    GPUConfig,
    HiddenStatesConfig,
    ModelsConfig,
    ServerConfig,
    get_model_config,
    load_config,
)


class TestConfigDefaults:
    """Test default configuration values."""

    def test_server_defaults(self):
        config = ServerConfig()
        assert config.transport == "http"
        assert config.http_host == "0.0.0.0"
        assert config.http_port == 8080
        assert config.unix_socket == "/tmp/loom.sock"

    def test_gpu_defaults(self):
        config = GPUConfig()
        assert config.devices == [0]
        assert config.default_device == 0
        assert config.memory_fraction == 0.9

    def test_hidden_states_defaults(self):
        config = HiddenStatesConfig()
        assert config.default_layers == [-1]
        assert config.include_attention is False
        assert config.precision == "float32"

    def test_models_defaults(self):
        config = ModelsConfig()
        assert config.cache_dir == "~/.cache/huggingface"
        assert config.preload == []
        assert config.max_loaded == 3

    def test_full_config_defaults(self):
        config = Config()
        assert config.server.http_port == 8080
        assert config.gpu.default_device == 0
        assert config.models.max_loaded == 3


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_from_yaml(self):
        config_data = {
            "server": {
                "http_port": 9000,
            },
            "gpu": {
                "devices": [0, 1],
                "default_device": 1,
            },
            "models": {
                "max_loaded": 5,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.server.http_port == 9000
            assert config.gpu.devices == [0, 1]
            assert config.gpu.default_device == 1
            assert config.models.max_loaded == 5
            # Defaults should still apply
            assert config.server.http_host == "0.0.0.0"
        finally:
            # Clean up temp file
            os.unlink(temp_path)

    def test_load_nonexistent_file_uses_defaults(self):
        config = load_config("/nonexistent/path/config.yaml")
        assert config.server.http_port == 8080  # Default

    def test_model_overrides(self):
        config = Config(
            model_overrides={
                "llama-3.1-8b": {
                    "dtype": "bfloat16",
                    "device": "cuda:1",
                },
            }
        )

        model_config = get_model_config(config, "llama-3.1-8b")
        assert model_config["dtype"] == "bfloat16"
        assert model_config["device"] == "cuda:1"

    def test_model_config_without_override(self):
        config = Config()
        model_config = get_model_config(config, "some-model")
        assert model_config["dtype"] == "auto"
        assert model_config["device"] == "cuda:0"


class TestConfigValidation:
    """Test configuration validation."""

    def test_memory_fraction_bounds(self):
        # Valid
        config = GPUConfig(memory_fraction=0.5)
        assert config.memory_fraction == 0.5

        # Invalid - should raise
        with pytest.raises(ValueError):
            GPUConfig(memory_fraction=1.5)

        with pytest.raises(ValueError):
            GPUConfig(memory_fraction=0.05)

    def test_max_loaded_minimum(self):
        config = ModelsConfig(max_loaded=1)
        assert config.max_loaded == 1

        with pytest.raises(ValueError):
            ModelsConfig(max_loaded=0)
