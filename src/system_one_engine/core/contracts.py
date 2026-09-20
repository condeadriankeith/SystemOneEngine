"""Pydantic v2 data contracts and schema definitions for System One Decision Engine.

Defines request/response payloads for the three core decision primitives:
- Choice[T]: Unordered categorical selection from dynamic candidates.
- Score: Ordinal expected value across 2 to 10 ordered levels.
- Boolean (Noul): Calibrated binary assertion under uncertainty.
- Decide: Speculative fan-out combining multiple heterogeneous questions.
"""

from enum import Enum
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class QuestionType(str, Enum):
    """Supported decision primitive types in the System One engine."""

    CHOICE = "choice"
    SCORE = "score"
    BOOLEAN = "boolean"


# ---------------------------------------------------------------------------
# 1. Choice Primitive Contracts
# ---------------------------------------------------------------------------


class ChoiceRequest(BaseModel):
    """Request payload for evaluating an unordered categorical Choice decision.

    Attributes:
        context: The input text, state description, or conversation history.
        instruction: Clear prompt directive instructing the engine how to evaluate.
        criteria: Mapping of candidate keys to their semantic descriptions (2 to 64 options).
        temperature: Post-hoc temperature multiplier (default 1.0).
    """

    question_type: Literal[QuestionType.CHOICE] = QuestionType.CHOICE
    context: str = Field(..., min_length=1, description="Contextual state payload.")
    instruction: str = Field(..., min_length=1, description="Evaluation prompt or question.")
    criteria: dict[str, str] = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Dictionary mapping candidate keys to descriptions (2 to 64 choices).",
    )
    temperature: float = Field(
        default=1.0, gt=0.0, le=10.0, description="Temperature scaling factor (> 0.0)."
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria_keys_non_empty(cls, criteria: dict[str, str]) -> dict[str, str]:
        """Ensure all candidate keys and descriptions are non-empty strings."""
        for key, desc in criteria.items():
            if not key.strip():
                raise ValueError("Candidate key in criteria cannot be blank.")
            if not desc.strip():
                raise ValueError(f"Description for candidate key '{key}' cannot be blank.")
        return criteria


class ChoiceResponse(BaseModel):
    """Response payload returned by a Choice evaluation.

    Attributes:
        choice: Winning candidate key from the input criteria.
        probabilities: Probability distribution mapped over every candidate key.
        confidence: Normalized lead of peak probability over uniform random chance [0.0, 1.0].
    """

    question_type: Literal[QuestionType.CHOICE] = QuestionType.CHOICE
    choice: str = Field(..., description="Top predicted candidate key.")
    probabilities: dict[str, float] = Field(
        ..., description="Normalized probability mapped to each candidate key."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized choice confidence score in [0.0, 1.0]."
    )

    @model_validator(mode="after")
    def validate_probabilities_and_choice(self) -> "ChoiceResponse":
        """Verify probabilities are non-negative, sum to ~1.0, and choice exists in keys."""
        if not self.probabilities:
            raise ValueError("Probabilities dictionary cannot be empty.")

        if self.choice not in self.probabilities:
            raise ValueError(f"Winning choice '{self.choice}' not found in probabilities dictionary.")

        prob_sum = sum(self.probabilities.values())
        if not (0.98 <= prob_sum <= 1.02):
            raise ValueError(f"Probabilities must sum to approximately 1.0, got {prob_sum:.4f}")

        for key, prob in self.probabilities.items():
            if prob < 0.0 or prob > 1.0:
                raise ValueError(f"Probability for '{key}' is out of bounds [0.0, 1.0]: {prob}")

        return self


# ---------------------------------------------------------------------------
# 2. Score Primitive Contracts
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    """Request payload for evaluating an ordinal Score on an ordered ladder.

    Attributes:
        context: Contextual state or text payload.
        instruction: Evaluation prompt explaining what metric is being graded.
        criteria: Ordered list of level descriptions from lowest (0) to highest (m-1).
                  Must contain between 2 and 10 levels.
        temperature: Post-hoc temperature multiplier (default 1.0).
    """

    question_type: Literal[QuestionType.SCORE] = QuestionType.SCORE
    context: str = Field(..., min_length=1, description="Contextual state payload.")
    instruction: str = Field(..., min_length=1, description="Evaluation prompt or rubric directive.")
    criteria: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Ascending level descriptions from level 0 to level m-1 (2 to 10 levels).",
    )
    temperature: float = Field(
        default=1.0, gt=0.0, le=10.0, description="Temperature scaling factor (> 0.0)."
    )

    @field_validator("criteria")
    @classmethod
    def validate_levels_non_empty(cls, levels: list[str]) -> list[str]:
        """Ensure every level description is a valid non-empty string."""
        for idx, desc in enumerate(levels):
            if not desc.strip():
                raise ValueError(f"Description for level index {idx} cannot be blank.")
        return levels


class ScoreResponse(BaseModel):
    """Response payload returned by a Score evaluation.

    Attributes:
        score: Continuous scalar representing probability-weighted expected value in [0, m-1].
        legend: Mapping from level index strings ("0", "1", ...) back to level descriptions.
        probabilities: Level probabilities mapped to index strings ("0", "1", ...).
        confidence: Ordinal concentration metric based on Mean Absolute Deviation from the mode.
    """

    question_type: Literal[QuestionType.SCORE] = QuestionType.SCORE
    score: float = Field(..., description="Expected value continuous scalar in [0, m-1].")
    legend: dict[str, str] = Field(
        ..., description="Mapping of level index string to description."
    )
    probabilities: dict[str, float] = Field(
        ..., description="Probabilities mapped to each level index string."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Ordinal score confidence in [0.0, 1.0]."
    )

    @model_validator(mode="after")
    def validate_score_and_legend(self) -> "ScoreResponse":
        """Verify score is bounded by the number of levels and probabilities sum to ~1.0."""
        num_levels = len(self.legend)
        if num_levels < 2 or num_levels > 10:
            raise ValueError(f"Legend must contain between 2 and 10 levels, got {num_levels}.")

        if not (0.0 <= self.score <= float(num_levels - 1)):
            raise ValueError(
                f"Score {self.score:.4f} is outside the allowable range [0.0, {num_levels - 1}]."
            )

        prob_sum = sum(self.probabilities.values())
        if not (0.98 <= prob_sum <= 1.02):
            raise ValueError(f"Level probabilities must sum to ~1.0, got {prob_sum:.4f}")

        return self


# ---------------------------------------------------------------------------
# 3. Boolean (Noul) Primitive Contracts
# ---------------------------------------------------------------------------


class BooleanRequest(BaseModel):
    """Request payload for evaluating a binary assertion (True / False).

    Attributes:
        context: Contextual state or text payload.
        instruction: Binary assertion statement (e.g. 'Is this message compliant with policy?').
        criteria: Optional dictionary specifying what defines 'true' and 'false'.
        temperature: Post-hoc temperature multiplier (default 1.0).
    """

    question_type: Literal[QuestionType.BOOLEAN] = QuestionType.BOOLEAN
    context: str = Field(..., min_length=1, description="Contextual state payload.")
    instruction: str = Field(..., min_length=1, description="Binary statement under evaluation.")
    criteria: dict[str, str] | None = Field(
        default=None,
        description="Optional dictionary specifying definition of 'true' and 'false'.",
    )
    temperature: float = Field(
        default=1.0, gt=0.0, le=10.0, description="Temperature scaling factor (> 0.0)."
    )


class BooleanResponse(BaseModel):
    """Response payload returned by a Boolean evaluation.

    Attributes:
        confirmed: Boolean flag (True if probability >= 0.5, else False).
        probability: Calibrated probability P(yes / true) in [0.0, 1.0].
        confidence: Distance from maximum uncertainty: |probability - 0.5| * 2 in [0.0, 1.0].
    """

    question_type: Literal[QuestionType.BOOLEAN] = QuestionType.BOOLEAN
    confirmed: bool = Field(..., description="Thresholded boolean result (True if P >= 0.5).")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated probability of True/Affirmative in [0.0, 1.0]."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Decision certainty metric: |P - 0.5| * 2 in [0.0, 1.0]."
    )


# ---------------------------------------------------------------------------
# 4. Composite Fan-out (Decide) Contracts
# ---------------------------------------------------------------------------

QuestionUnion = Annotated[
    Union[ChoiceRequest, ScoreRequest, BooleanRequest],
    Field(discriminator="question_type"),
]

AnswerUnion = Annotated[
    Union[ChoiceResponse, ScoreResponse, BooleanResponse],
    Field(discriminator="question_type"),
]


class DecideRequest(BaseModel):
    """Speculative fan-out request evaluating multiple questions against a shared context.

    Attributes:
        context: The shared context payload.
        questions: Dictionary mapping arbitrary question identifiers to question specifications.
    """

    context: str = Field(..., min_length=1, description="Shared context string for all questions.")
    questions: dict[str, QuestionUnion] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Map of question IDs to specific primitive requests.",
    )

    @field_validator("questions")
    @classmethod
    def validate_question_keys(cls, questions: dict[str, QuestionUnion]) -> dict[str, QuestionUnion]:
        """Ensure all question IDs are non-blank strings."""
        for key in questions:
            if not key.strip():
                raise ValueError("Question identifier in questions map cannot be blank.")
        return questions


class DecideResponse(BaseModel):
    """Aggregated response containing answers for all submitted questions.

    Attributes:
        answers: Map of question IDs to evaluated answer payloads.
        latency_ms: Execution duration in milliseconds.
    """

    answers: dict[str, AnswerUnion] = Field(
        ..., description="Map of question IDs to evaluated primitive answers."
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Server execution latency in milliseconds."
    )
