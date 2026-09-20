"""Unit tests for FastAPI decision microservice endpoints."""

import pytest
from starlette.testclient import TestClient

from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter
from system_one_engine.server.app import create_app


@pytest.fixture
def client():
    """Create test client with mock adapter."""
    mock_adapter = OllamaSystemOneAdapter(mock_mode=True)
    app = create_app(adapter=mock_adapter)
    return TestClient(app)


def test_health_endpoint(client):
    """Verify /health endpoint returns healthy status and backend name."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "OllamaSystemOneAdapter"
    assert data["version"] == "0.1.0"


def test_metrics_endpoint(client):
    """Verify /metrics tracks requests and average latency."""
    # Execute a call first
    client.get("/health")
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    data = metrics_resp.json()
    assert "requests_total" in data
    assert "avg_latency_ms" in data


def test_choice_endpoint(client):
    """Verify POST /choice returns a validated ChoiceResponse."""
    payload = {
        "context": "Customer has not received their package #9941.",
        "instruction": "Route query to correct queue.",
        "criteria": {
            "tracking": "Check package location",
            "refund": "Process money refund",
        },
    }
    resp = client.post("/choice", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["question_type"] == "choice"
    assert data["choice"] in ["tracking", "refund"]
    assert "probabilities" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_boolean_endpoint(client):
    """Verify POST /boolean returns a validated BooleanResponse."""
    payload = {
        "context": "User asks: 'How do I change my account password?'",
        "instruction": "Is this a benign account maintenance request?",
    }
    resp = client.post("/boolean", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["question_type"] == "boolean"
    assert isinstance(data["confirmed"], bool)
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0


def test_score_endpoint(client):
    """Verify POST /score returns a validated ScoreResponse."""
    payload = {
        "context": "Agent was rude and refused to answer my questions.",
        "instruction": "Rate severity of complaint.",
        "criteria": ["Minor", "Moderate", "Major"],
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["question_type"] == "score"
    assert 0.0 <= data["score"] <= 2.0
    assert len(data["legend"]) == 3
    assert 0.0 <= data["confidence"] <= 1.0


def test_decide_endpoint(client):
    """Verify POST /decide evaluates speculative fan-out questions."""
    payload = {
        "context": "Customer requests subscription cancellation due to high pricing.",
        "questions": {
            "intent": {
                "question_type": "choice",
                "context": "Customer requests subscription cancellation due to high pricing.",
                "instruction": "Classify intent.",
                "criteria": {"cancel": "Cancel plan", "upgrade": "Upgrade plan"},
            },
            "at_risk": {
                "question_type": "boolean",
                "context": "Customer requests subscription cancellation due to high pricing.",
                "instruction": "Is customer churning?",
            },
        },
    }
    resp = client.post("/decide", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["answers"]) == 2
    assert data["answers"]["intent"]["question_type"] == "choice"
    assert data["answers"]["at_risk"]["question_type"] == "boolean"
    assert data["latency_ms"] >= 0.0


def test_chat_endpoint(client):
    """Verify POST /chat evaluates conversational message and returns telemetry."""
    payload = {"message": "My order is broken and I want a refund right away!"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert len(data["reply"]) > 0
    assert "telemetry" in data
    telem = data["telemetry"]
    assert "intent" in telem
    assert "intent_confidence" in telem
    assert "sentiment_score" in telem
    assert "needs_escalation" in telem
    assert "system_one_latency_ms" in telem
    assert data["total_latency_ms"] >= 0.0


def test_chat_ui_endpoint(client):
    """Verify GET / and GET /chat-ui return HTML web interface."""
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "text/html" in resp_root.headers["content-type"]
    assert "System One Engine" in resp_root.text

    resp_ui = client.get("/chat-ui")
    assert resp_ui.status_code == 200
    assert "text/html" in resp_ui.headers["content-type"]


def test_chat_stream_endpoint(client):
    """Verify POST /chat/stream yields valid SSE data chunks."""
    import json
    payload = {"message": "Hello stream test"}
    with client.stream("POST", "/chat/stream", json=payload) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                json_str = line[5:].strip()
                if json_str:
                    events.append(json.loads(json_str))
        assert len(events) >= 1
        assert "telemetry" in events[0]
        assert any(e.get("done") for e in events)

