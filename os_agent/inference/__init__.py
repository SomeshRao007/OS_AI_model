from os_agent.inference.engine import InferenceEngine
from os_agent.inference.backend import (
    BackendManager,
    InferenceBackend,
    LocalBackend,
    OpenRouterBackend,
)
from os_agent.inference.model_registry import ModelInfo, ModelRegistry
from os_agent.inference.openrouter import OpenRouterClient
from os_agent.inference.prompt import MASTER_CLASSIFY_PROMPT, SYSTEM_PROMPTS, get_prompt
from os_agent.inference.rag import build_rag_context, detect_command, get_help_context
from os_agent.inference.validator import validate, infer_arg_type

__all__ = [
    "InferenceEngine",
    "BackendManager",
    "InferenceBackend",
    "LocalBackend",
    "OpenRouterBackend",
    "ModelInfo",
    "ModelRegistry",
    "OpenRouterClient",
    "SYSTEM_PROMPTS",
    "MASTER_CLASSIFY_PROMPT",
    "get_prompt",
    "build_rag_context",
    "detect_command",
    "get_help_context",
    "validate",
    "infer_arg_type",
]
