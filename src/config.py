"""Configuration management for Research Model Server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """Server transport configuration."""

    transport: str = Field(default="http", description="Transport type: http, unix, or both")
    http_host: str = Field(default="0.0.0.0", description="HTTP server host")
    http_port: int = Field(default=8080, description="HTTP server port")
    unix_socket: str = Field(default="/tmp/loom.sock", description="Unix socket path")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use ['*'] for development, specific origins for production",
    )
    cors_allow_credentials: bool = Field(
        default=False,
        description="Allow credentials in CORS requests. Should be False when origins is ['*']",
    )


class GPUConfig(BaseModel):
    """GPU device configuration."""

    devices: list[int] = Field(default=[0], description="Available CUDA device indices")
    default_device: int = Field(default=0, description="Default CUDA device")
    memory_fraction: float = Field(
        default=0.9, ge=0.1, le=1.0, description="Max GPU memory fraction to use"
    )


class HiddenStatesConfig(BaseModel):
    """Hidden state extraction configuration."""

    default_layers: list[int] = Field(
        default=[-1], description="Which layers to return by default (-1 = last)"
    )
    include_attention: bool = Field(
        default=False, description="Include attention weights in response"
    )
    precision: str = Field(
        default="float32", description="Precision for hidden states: float16, float32, bfloat16"
    )


class ModelsConfig(BaseModel):
    """Model loading and caching configuration."""

    cache_dir: str = Field(
        default="~/.cache/huggingface", description="HuggingFace cache directory"
    )
    preload: list[str] = Field(default=[], description="Models to load at startup")
    max_loaded: int = Field(default=3, ge=1, description="Maximum concurrent models in memory")
    default_dtype: str = Field(
        default="auto", description="Default dtype: auto, float16, bfloat16, float32"
    )
    trust_remote_code: bool = Field(
        default=True, description="Allow remote code execution for custom architectures"
    )
    auto_unload_minutes: int = Field(
        default=20,
        ge=0,
        description="Auto-unload models after this many minutes of inactivity (0 = disabled)",
    )


class LoadersConfig(BaseModel):
    """Loader-specific configuration."""

    # Loader priority order for auto-detection
    fallback_order: list[str] = Field(
        default=["transformers", "sentence_transformers", "custom"],
        description="Order to try loaders during auto-detection",
    )

    # TransformersLoader settings
    transformers_default_kwargs: dict[str, Any] = Field(
        default={},
        description="Default kwargs passed to TransformersLoader.load()",
    )

    # SentenceTransformersLoader settings
    sentence_transformers_default_kwargs: dict[str, Any] = Field(
        default={},
        description="Default kwargs passed to SentenceTransformersLoader.load()",
    )

    # CustomLoader settings
    custom_default_kwargs: dict[str, Any] = Field(
        default={},
        description="Default kwargs passed to CustomLoader.load()",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    file: str | None = Field(default=None, description="Log file path (None for stdout only)")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )


class Config(BaseSettings):
    """Main configuration for The Loom."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    hidden_states: HiddenStatesConfig = Field(default_factory=HiddenStatesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    loaders: LoadersConfig = Field(default_factory=LoadersConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Model-specific overrides (loader, dtype, device, etc.)
    model_overrides: dict[str, dict[str, Any]] = Field(
        default={},
        description="Per-model configuration overrides. Keys are model IDs, values are dicts with: loader, dtype, device, etc.",
    )

    class Config:
        env_prefix = "LOOM_"
        env_nested_delimiter = "__"


def find_config_file() -> Path | None:
    """
    Locate a Loom configuration file by searching common filesystem locations in order.
    
    Searches these locations (in order) and returns the first path that exists:
    - ./config.yaml
    - ./config.yml
    - ./loom.yaml
    - ~/.config/loom/config.yaml
    - ~/.config/loom/config.yml
    - /etc/loom/config.yaml
    
    Returns:
        Path | None: The path to the first existing configuration file, or `None` if no file is found.
    """
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path.cwd() / "loom.yaml",
        Path.home() / ".config" / "loom" / "config.yaml",
        Path.home() / ".config" / "loom" / "config.yml",
        Path("/etc/loom/config.yaml"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from file and environment.

    Priority (highest to lowest):
    1. Environment variables (LOOM_*)
    2. Specified config file
    3. Auto-discovered config file
    4. Default values
    """
    config_data: dict[str, Any] = {}

    # Find config file
    if config_path is None:
        config_path = find_config_file()
    elif isinstance(config_path, str):
        config_path = Path(config_path)

    # Load from YAML if found
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data:
                config_data = yaml_data

    # Create config with loaded data (env vars override via pydantic-settings)
    return Config(**config_data)


def get_model_config(config: Config, model_id: str) -> dict[str, Any]:
    """
    Compute the effective configuration for a given model by merging global defaults with any model-specific overrides.
    
    Parameters:
        config (Config): Global server configuration.
        model_id (str): Identifier of the model whose overrides should be applied.
    
    Returns:
        dict[str, Any]: Effective model configuration. Contains default keys such as `dtype`, `device`, and `loader`, with values replaced or extended by any entries from `config.model_overrides[model_id]` when present.
    """
    base_config = {
        "dtype": config.models.default_dtype,
        "device": f"cuda:{config.gpu.default_device}",
        "loader": "auto",
    }

    # Apply model-specific overrides
    if model_id in config.model_overrides:
        base_config.update(config.model_overrides[model_id])

    return base_config


# Global config instance (initialized lazily)
_config: Config | None = None


def get_config() -> Config:
    """
    Return the application's global configuration singleton.
    
    Returns:
        Config: The global Config instance; created and loaded on first access.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """
    Replace the module-level global configuration with the provided Config instance.
    
    This updates the internal singleton used by get_config() so subsequent calls return `config`.
    
    Parameters:
        config (Config): Configuration to install as the module-global instance.
    """
    global _config
    _config = config