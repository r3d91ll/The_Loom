"""Model loaders for different architectures."""

from .base import EmbeddingOutput, GenerationOutput, LoadedModel, ModelLoader
from .custom_loader import CustomLoader, CustomModelConfig, register_custom_model
from .mistral_loader import MistralLoader
from .qwen_loader import QwenLoader
from .registry import (
    LoaderRegistry,
    create_registry_from_config,
    get_registry,
    set_registry,
)
from .sentence_transformers_loader import SentenceTransformersLoader
from .transformers_loader import TransformersLoader

__all__ = [
    # Base classes
    "ModelLoader",
    "LoadedModel",
    "GenerationOutput",
    "EmbeddingOutput",
    # Loaders
    "TransformersLoader",
    "SentenceTransformersLoader",
    "MistralLoader",
    "QwenLoader",
    "CustomLoader",
    "CustomModelConfig",
    "register_custom_model",
    # Registry
    "LoaderRegistry",
    "get_registry",
    "set_registry",
    "create_registry_from_config",
]
