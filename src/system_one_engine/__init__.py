"""System One Decision Engine.

Fast, calibrated non-autoregressive decision engine for software systems
and autonomous agents. Public API aligned with laya-mlx (Apache-2.0).

Quick start:
    from system_one_engine import Router, triage_questions
    router = Router()
    result = router.predict(state, triage_questions())

    # Or use a specific adapter directly:
    from system_one_engine import load
    agent = load("ollama")          # mock-mode for testing
    result = agent.predict(state, triage_questions())
"""

from system_one_engine.core.router import DEFAULT_MODEL, MODEL_OLLAMA, MODEL_ONNX, Router
from system_one_engine.core.presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from system_one_engine.core.shortlist import predict_shortlist, shortlist_choice
from system_one_engine.core.confidence import compute_entropy_confidence
from system_one_engine.core.calibration import (
    TEMP_MAX,
    TEMP_MIN,
    clamp_temperature,
    temp_bucket,
)
from system_one_engine.core.prompt import (
    build_sequence,
    render_options,
    serialize_state,
    to_internal,
)
from system_one_engine.core.prefix_cache import PrefixCache
from system_one_engine.core.contracts import (
    BooleanRequest,
    BooleanResponse,
    ChoiceRequest,
    ChoiceResponse,
    DecideRequest,
    DecideResponse,
    ScoreRequest,
    ScoreResponse,
)
from system_one_engine.agent import (
    AgentTask,
    LocalOSDriver,
    OSAction,
    OSActionResult,
    OSActionType,
    OSObservation,
    SafetyGuardrail,
    SafetyMode,
    SimulatedOSDriver,
    SystemOneOSAgent,
)

__version__ = "0.3.0"


def load(model: str = DEFAULT_MODEL, **kwargs) -> object:
    """Convenience factory: build and return a named adapter instance.

    Mirrors the laya-mlx ``load()`` helper so callers can use a single
    import to get a working agent without constructing a full Router.

    Args:
        model: Adapter name — ``"onnx"`` or ``"ollama"`` (default: ``"onnx"``).
        **kwargs: Forwarded to the adapter constructor.

    Returns:
        object: Adapter instance with ``predict(state, questions)`` method.

    Example:
        agent = load("ollama", mock_mode=True)
        result = agent.predict(state, guard_questions())
    """
    from pathlib import Path

    if model == MODEL_ONNX:
        onnx_dir = Path("models/onnx")
        enc_path = onnx_dir / "encoder_int8.onnx"
        heads_path = onnx_dir / "heads_int8.onnx"
        ch_path = onnx_dir / "choice_head_int8.onnx"

        if enc_path.exists() and heads_path.exists() and ch_path.exists():
            from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter
            from system_one_engine.training.dataset import SimpleVocabTokenizer

            tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
            return ONNXRuntimeAdapter(
                encoder_path=str(enc_path),
                heads_path=str(heads_path),
                choice_head_path=str(ch_path),
                tokenizer=tokenizer,
                **kwargs,
            )
        # Fall through to Ollama if ONNX files not found

    from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter

    return OllamaSystemOneAdapter(**kwargs)


__all__ = [
    # Factory
    "load",
    # Router
    "Router",
    "MODEL_ONNX",
    "MODEL_OLLAMA",
    "DEFAULT_MODEL",
    # Presets
    "triage_questions",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    # Shortlist
    "shortlist_choice",
    "predict_shortlist",
    # Confidence
    "compute_entropy_confidence",
    # Calibration
    "temp_bucket",
    "clamp_temperature",
    "TEMP_MIN",
    "TEMP_MAX",
    # Prompt layout
    "serialize_state",
    "render_options",
    "build_sequence",
    "to_internal",
    # Prefix cache
    "PrefixCache",
    # Contracts
    "ChoiceRequest",
    "ChoiceResponse",
    "ScoreRequest",
    "ScoreResponse",
    "BooleanRequest",
    "BooleanResponse",
    "DecideRequest",
    "DecideResponse",
    # Computer Agent
    "SystemOneOSAgent",
    "AgentTask",
    "OSObservation",
    "OSAction",
    "OSActionType",
    "OSActionResult",
    "SafetyMode",
    "SafetyGuardrail",
    "BaseOSDriver",
    "SimulatedOSDriver",
    "LocalOSDriver",
]
