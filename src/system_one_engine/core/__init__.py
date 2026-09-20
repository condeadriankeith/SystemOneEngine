"""SystemOneEngine - Core module.

Defines Pydantic v2 data contracts, confidence computation routines,
and post-hoc probability calibration (ECE & temperature scaling).
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
    compute_expected_score,
    compute_score_confidence,
)
from system_one_engine.core.calibration import (
    TemperatureScaler,
    compute_ece,
)

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
    # Calibration
    "compute_ece",
    "TemperatureScaler",
]
