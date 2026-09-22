"""SystemOneEngine - Core module.

Defines Pydantic v2 data contracts, confidence computation routines,
post-hoc probability calibration (ECE & temperature scaling), prompt
layout utilities, production presets, prefix caching, and shortlist
helpers (all aligned with laya-mlx, Apache-2.0).
"""

from system_one_engine.core.contracts import (
    AnswerUnion,
    BooleanRequest,
    BooleanResponse,
    ChoiceRequest,
    ChoiceResponse,
    DecideRequest,
    DecideResponse,
    QuestionType,
    QuestionUnion,
    ScoreRequest,
    ScoreResponse,
)
from system_one_engine.core.confidence import (
    compute_boolean_confidence,
    compute_choice_confidence,
    compute_entropy_confidence,
    compute_expected_score,
    compute_score_confidence,
)
from system_one_engine.core.calibration import (
    TEMP_MAX,
    TEMP_MIN,
    TemperatureScaler,
    clamp_temperature,
    compute_ece,
    temp_bucket,
)
from system_one_engine.core.prompt import (
    build_prefix,
    build_sequence,
    render_criterion,
    render_options,
    serialize_state,
    to_internal,
)
from system_one_engine.core.presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from system_one_engine.core.prefix_cache import PrefixCache
from system_one_engine.core.shortlist import predict_shortlist, shortlist_choice
from system_one_engine.core.router import DEFAULT_MODEL, MODEL_OLLAMA, MODEL_ONNX, Router

__all__ = [
    # Contracts
    "QuestionType",
    "ChoiceRequest",
    "ChoiceResponse",
    "ScoreRequest",
    "ScoreResponse",
    "BooleanRequest",
    "BooleanResponse",
    "DecideRequest",
    "DecideResponse",
    "QuestionUnion",
    "AnswerUnion",
    # Confidence
    "compute_choice_confidence",
    "compute_score_confidence",
    "compute_expected_score",
    "compute_boolean_confidence",
    "compute_entropy_confidence",
    # Calibration
    "compute_ece",
    "TemperatureScaler",
    "temp_bucket",
    "clamp_temperature",
    "TEMP_MIN",
    "TEMP_MAX",
    # Prompt layout
    "serialize_state",
    "render_criterion",
    "render_options",
    "build_prefix",
    "build_sequence",
    "to_internal",
    # Presets
    "triage_questions",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    # Prefix cache
    "PrefixCache",
    # Shortlist
    "shortlist_choice",
    "predict_shortlist",
    # Router
    "Router",
    "MODEL_ONNX",
    "MODEL_OLLAMA",
    "DEFAULT_MODEL",
]
