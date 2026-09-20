"""Unit tests for Pydantic v2 data contracts and schema validations."""

import pytest
from pydantic import ValidationError
from system_one_engine.core.contracts import (
    BooleanRequest,
    BooleanResponse,
    ChoiceRequest,
    ChoiceResponse,
    DecideRequest,
    DecideResponse,
    QuestionType,
    ScoreRequest,
    ScoreResponse,
)


# ---------------------------------------------------------------------------
# Choice Contract Tests
# ---------------------------------------------------------------------------


def test_choice_request_valid():
    """Verify standard valid ChoiceRequest."""
    req = ChoiceRequest(
        context="My order was damaged during transit.",
        instruction="Route to the appropriate customer department.",
        criteria={
            "refund": "Requesting money back for broken items.",
            "replacement": "Requesting a replacement package.",
            "general_inquiry": "Asking about policies or timelines.",
        },
    )
    assert req.question_type == QuestionType.CHOICE
    assert len(req.criteria) == 3
    assert req.temperature == 1.0


def test_choice_request_criteria_bounds():
    """Verify ChoiceRequest fails with fewer than 2 or more than 64 candidates."""
    # Under minimum (1 choice)
    with pytest.raises(ValidationError):
        ChoiceRequest(
            context="Test context",
            instruction="Test instruction",
            criteria={"only_one": "Description"},
        )

    # Empty candidate key
    with pytest.raises(ValidationError):
        ChoiceRequest(
            context="Test context",
            instruction="Test instruction",
            criteria={"": "Description", "valid": "Valid description"},
        )


def test_choice_response_valid():
    """Verify valid ChoiceResponse creation and validation."""
    resp = ChoiceResponse(
        choice="refund",
        probabilities={"refund": 0.85, "replacement": 0.10, "general_inquiry": 0.05},
        confidence=0.775,
    )
    assert resp.choice == "refund"
    assert resp.probabilities["refund"] == 0.85
    assert resp.confidence == 0.775


def test_choice_response_invalid_sum_or_missing_choice():
    """Verify ChoiceResponse rejects unnormalized distributions or missing choice."""
    # Probabilities do not sum to 1.0
    with pytest.raises(ValidationError):
        ChoiceResponse(
            choice="refund",
            probabilities={"refund": 0.50, "replacement": 0.10},
            confidence=0.5,
        )

    # Choice not in dictionary keys
    with pytest.raises(ValidationError):
        ChoiceResponse(
            choice="unknown_choice",
            probabilities={"refund": 0.80, "replacement": 0.20},
            confidence=0.6,
        )


# ---------------------------------------------------------------------------
# Score Contract Tests
# ---------------------------------------------------------------------------


def test_score_request_valid():
    """Verify valid ScoreRequest with 4 ordered levels."""
    req = ScoreRequest(
        context="Customer is threatening legal action unless refunded immediately.",
        instruction="Rate customer frustration level.",
        criteria=[
            "Calm and polite inquiry.",
            "Slight irritation or confusion.",
            "Clear anger with repeated demands.",
            "Extreme hostility or legal threats.",
        ],
    )
    assert req.question_type == QuestionType.SCORE
    assert len(req.criteria) == 4


def test_score_request_bounds():
    """Verify ScoreRequest requires between 2 and 10 levels."""
    # 1 level (too few)
    with pytest.raises(ValidationError):
        ScoreRequest(
            context="context",
            instruction="instruction",
            criteria=["Only one level"],
        )

    # 11 levels (too many)
    with pytest.raises(ValidationError):
        ScoreRequest(
            context="context",
            instruction="instruction",
            criteria=[f"Level {i}" for i in range(11)],
        )


def test_score_response_valid():
    """Verify valid ScoreResponse validation."""
    resp = ScoreResponse(
        score=2.65,
        legend={"0": "Low", "1": "Medium", "2": "High", "3": "Critical"},
        probabilities={"0": 0.05, "1": 0.10, "2": 0.35, "3": 0.50},
        confidence=0.72,
    )
    assert resp.score == 2.65
    assert len(resp.legend) == 4


def test_score_response_out_of_bounds():
    """Verify ScoreResponse rejects score exceeding maximum level index."""
    with pytest.raises(ValidationError):
        ScoreResponse(
            score=4.5,  # Max allowable is 3.0 for 4 levels
            legend={"0": "Low", "1": "Medium", "2": "High", "3": "Critical"},
            probabilities={"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0},
            confidence=1.0,
        )


# ---------------------------------------------------------------------------
# Boolean Contract Tests
# ---------------------------------------------------------------------------


def test_boolean_request_valid():
    """Verify valid BooleanRequest."""
    req = BooleanRequest(
        context="User query: 'Give me instructions on making explosives.'",
        instruction="Does this request violate safety policies against hazardous weapons?",
    )
    assert req.question_type == QuestionType.BOOLEAN
    assert req.criteria is None


def test_boolean_response_valid():
    """Verify valid BooleanResponse."""
    resp = BooleanResponse(confirmed=True, probability=0.98, confidence=0.96)
    assert resp.confirmed is True
    assert resp.probability == 0.98
    assert resp.confidence == 0.96


# ---------------------------------------------------------------------------
# Decide Composite Contract Tests
# ---------------------------------------------------------------------------


def test_decide_composite_request_and_response():
    """Verify speculative fan-out DecideRequest with mixed question types."""
    req = DecideRequest(
        context="Customer order #9921 delayed by 5 days. Customer writes: 'Where is my order?'",
        questions={
            "intent": ChoiceRequest(
                context="Customer order #9921 delayed by 5 days. Customer writes: 'Where is my order?'",
                instruction="Classify customer intent.",
                criteria={
                    "track_package": "Asking for shipment location.",
                    "cancel_order": "Requesting cancellation.",
                },
            ),
            "frustration": ScoreRequest(
                context="Customer order #9921 delayed by 5 days. Customer writes: 'Where is my order?'",
                instruction="Grade customer frustration.",
                criteria=["Calm", "Mild", "Severe"],
            ),
            "urgent": BooleanRequest(
                context="Customer order #9921 delayed by 5 days. Customer writes: 'Where is my order?'",
                instruction="Is this issue marked urgent?",
            ),
        },
    )
    assert len(req.questions) == 3
    assert req.questions["intent"].question_type == QuestionType.CHOICE
    assert req.questions["frustration"].question_type == QuestionType.SCORE
    assert req.questions["urgent"].question_type == QuestionType.BOOLEAN

    # Simulate corresponding DecideResponse
    resp = DecideResponse(
        answers={
            "intent": ChoiceResponse(
                choice="track_package",
                probabilities={"track_package": 0.95, "cancel_order": 0.05},
                confidence=0.90,
            ),
            "frustration": ScoreResponse(
                score=0.1,
                legend={"0": "Calm", "1": "Mild", "2": "Severe"},
                probabilities={"0": 0.90, "1": 0.10, "2": 0.0},
                confidence=0.85,
            ),
            "urgent": BooleanResponse(
                confirmed=False,
                probability=0.08,
                confidence=0.84,
            ),
        },
        latency_ms=18.4,
    )
    assert resp.latency_ms == 18.4
    assert len(resp.answers) == 3
