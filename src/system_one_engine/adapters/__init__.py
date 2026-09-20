"""SystemOneEngine - Adapters module.

Contains engine runtime adapters, including the Ollama zero-shot logit adapter
and the high-speed ONNX Runtime CPU inference engine.
"""

from system_one_engine.adapters.ollama_adapter import (
    OllamaConnectionError,
    OllamaSystemOneAdapter,
)
from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter

__all__ = [
    "OllamaSystemOneAdapter",
    "OllamaConnectionError",
    "ONNXRuntimeAdapter",
]
