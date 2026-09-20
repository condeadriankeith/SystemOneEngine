"""Unit and integration tests for OllamaSystemOneAdapter."""

import pytest
from system_one_engine.adapters.ollama_adapter import (
    OllamaConnectionError,
    OllamaSystemOneAdapter,
)
from system_one_engine.core.contracts import (
    BooleanRequest,
    ChoiceRequest,
    DecideRequest,
    QuestionType,
    ScoreRequest,
)


def test_prompt_formatting():
    """Verify prompt formatting contains State, Question, lettered options, and answer prefix."""
    adapter = OllamaSystemOneAdapter(mock_mode=True)
    prompt = adapter._build_prompt(
        context="Account balance is $150.",
        instruction="Can the user withdraw $200?",
        option_lines=["True — Can withdraw", "False — Insufficient funds"],
    )

    assert "State:\nAccount balance is $150." in prompt
    assert "Question: Can the user withdraw $200?" in prompt
    assert "A: True — Can withdraw" in prompt
    assert "B: False — Insufficient funds" in prompt
    assert '{"answer": "' in prompt


def test_evaluate_choice_mock():
    """Verify Choice evaluation produces a validated ChoiceResponse."""
    adapter = OllamaSystemOneAdapter(mock_mode=True, permutations=2)

    req = ChoiceRequest(
        context="Order #8839 package was torn upon arrival.",
        instruction="Select routing destination.",
        criteria={
            "claims": "File a damaged goods claim.",
            "tracking": "Check shipment location.",
            "faq": "General questions.",
        },
    )

    resp = adapter.evaluate_choice(req)
    assert resp.question_type == QuestionType.CHOICE
    assert resp.choice in req.criteria
    assert sum(resp.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= resp.confidence <= 1.0


def test_evaluate_score_mock():
    """Verify Score evaluation produces a validated ScoreResponse."""
    adapter = OllamaSystemOneAdapter(mock_mode=True, permutations=3)

    req = ScoreRequest(
        context="Customer: 'You guys are total fraudsters, refund me now!'",
        instruction="Rate customer anger level.",
        criteria=[
            "Calm",
            "Slightly annoyed",
            "Very angry",
            "Severe rage / legal threat",
        ],
    )

    resp = adapter.evaluate_score(req)
    assert resp.question_type == QuestionType.SCORE
    assert 0.0 <= resp.score <= 3.0
    assert len(resp.legend) == 4
    assert sum(resp.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= resp.confidence <= 1.0


def test_evaluate_boolean_mock():
    """Verify Boolean evaluation produces a validated BooleanResponse."""
    adapter = OllamaSystemOneAdapter(mock_mode=True, permutations=2)

    req = BooleanRequest(
        context="Prompt: 'Write a python script to calculate fibonacci numbers.'",
        instruction="Is this code generation request safe to execute?",
    )

    resp = adapter.evaluate_boolean(req)
    assert resp.question_type == QuestionType.BOOLEAN
    assert isinstance(resp.confirmed, bool)
    assert 0.0 <= resp.probability <= 1.0
    assert 0.0 <= resp.confidence <= 1.0


def test_evaluate_decide_speculative_fan_out():
    """Verify speculative fan-out evaluating multiple questions against shared context."""
    adapter = OllamaSystemOneAdapter(mock_mode=True, permutations=2)

    req = DecideRequest(
        context="Customer inquiry: 'I need to cancel my subscription and get my money back.'",
        questions={
            "intent": ChoiceRequest(
                context="Customer inquiry: 'I need to cancel my subscription and get my money back.'",
                instruction="Classify intent.",
                criteria={
                    "cancel_subscription": "Cancellation request.",
                    "billing_inquiry": "Billing question.",
                },
            ),
            "severity": ScoreRequest(
                context="Customer inquiry: 'I need to cancel my subscription and get my money back.'",
                instruction="Rate request severity.",
                criteria=["Low", "Medium", "High"],
            ),
            "is_churn_risk": BooleanRequest(
                context="Customer inquiry: 'I need to cancel my subscription and get my money back.'",
                instruction="Is this customer at immediate risk of churning?",
            ),
        },
    )

    resp = adapter.evaluate_decide(req)
    assert len(resp.answers) == 3
    assert resp.latency_ms >= 0.0
    assert resp.answers["intent"].question_type == QuestionType.CHOICE
    assert resp.answers["severity"].question_type == QuestionType.SCORE
    assert resp.answers["is_churn_risk"].question_type == QuestionType.BOOLEAN


def test_connection_error_on_unreachable_endpoint():
    """Verify OllamaConnectionError is raised with clear message when daemon is unreachable."""
    adapter = OllamaSystemOneAdapter(
        base_url="http://127.0.0.1:59999",  # Nonexistent local port
        timeout=1.0,
        mock_mode=False,
    )

    req = BooleanRequest(
        context="Test context",
        instruction="Test instruction",
    )

    with pytest.raises(OllamaConnectionError) as exc_info:
        adapter.evaluate_boolean(req)

    assert "Ensure 'ollama serve' is running" in str(exc_info.value)
