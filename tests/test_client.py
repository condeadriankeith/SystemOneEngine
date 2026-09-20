"""Unit tests for SystemOneClient SDK."""

import pytest
from starlette.testclient import TestClient

from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter
from system_one_engine.client.client import SystemOneClient
from system_one_engine.core.contracts import (
    BooleanRequest,
    BooleanResponse,
    ChoiceRequest,
    ChoiceResponse,
    QuestionType,
    ScoreResponse,
)
from system_one_engine.server.app import create_app


def test_client_in_process_mode():
    """Verify SystemOneClient executing in-process against adapter directly."""
    adapter = OllamaSystemOneAdapter(mock_mode=True)
    client = SystemOneClient(adapter=adapter)

    # 1. Health
    h = client.health()
    assert h["status"] == "healthy"
    assert h["backend"] == "OllamaSystemOneAdapter"

    # 2. Choice
    choice_res = client.choice(
        context="Order damaged in transit.",
        instruction="Route query.",
        criteria={"claims": "File claim", "help": "General help"},
    )
    assert isinstance(choice_res, ChoiceResponse)
    assert choice_res.choice in ["claims", "help"]

    # 3. Score
    score_res = client.score(
        context="Reviewing code PR.",
        instruction="Rate complexity.",
        criteria=["Easy", "Medium", "Hard"],
    )
    assert isinstance(score_res, ScoreResponse)
    assert 0.0 <= score_res.score <= 2.0

    # 4. Boolean
    bool_res = client.boolean(
        context="Is this code safe?",
        instruction="Is this safe?",
    )
    assert isinstance(bool_res, BooleanResponse)

    # 5. Decide fan-out
    decide_res = client.decide(
        context="Shared state context.",
        questions={
            "q1": ChoiceRequest(
                context="Shared state context.",
                instruction="Pick option.",
                criteria={"a": "Opt A", "b": "Opt B"},
            ),
            "q2": BooleanRequest(
                context="Shared state context.",
                instruction="Check true.",
            ),
        },
    )
    assert len(decide_res.answers) == 2
    assert decide_res.answers["q1"].question_type == QuestionType.CHOICE


def test_client_context_manager():
    """Verify client context management cleanly closes session."""
    with SystemOneClient(base_url="http://127.0.0.1:8000") as client:
        assert client.base_url == "http://127.0.0.1:8000"
    assert client._http_client is None
