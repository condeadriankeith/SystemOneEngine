"""Automated test suite for System Two Engine."""

import pytest
import httpx

from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter
from system_one_engine.system_two import (
    StreamChunk,
    SystemTwoEngine,
    SystemTwoRequest,
    SystemTwoResponse,
)


@pytest.fixture
def mock_system_one_adapter():
    """System One adapter running in deterministic mock mode."""
    return OllamaSystemOneAdapter(mock_mode=True)


def test_system_two_contracts():
    """Verify System Two Pydantic v2 schemas and validation bounds."""
    req = SystemTwoRequest(message="Write a python binary search function")
    assert req.message == "Write a python binary search function"
    assert req.history == []
    assert req.stream is False

    chunk = StreamChunk(delta="def binary_search", done=False)
    assert chunk.delta == "def binary_search"
    assert chunk.done is False


def test_system_two_steered_prompt_generation(mock_system_one_adapter):
    """Verify prompt composition adapts to System 1 classified intent."""
    engine = SystemTwoEngine(system_one_adapter=mock_system_one_adapter)
    telemetry = engine.evaluate_system_one("write a snake game in python")

    assert telemetry.intent in [
        "code_generation",
        "debugging",
        "code_explanation",
        "refactoring",
        "general_inquiry",
    ]
    assert 0.0 <= telemetry.sentiment_score <= 4.0

    prompt = engine.build_steered_prompt(
        "write a snake game in python", telemetry, history=[]
    )
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "write a snake game in python" in prompt


def test_system_two_offline_fallback(mock_system_one_adapter):
    """Verify that unreachable Ollama port falls back gracefully without crashing."""
    # Point to nonexistent port
    engine = SystemTwoEngine(
        system_one_adapter=mock_system_one_adapter,
        ollama_url="http://127.0.0.1:59999",
        timeout_sec=1.0,
    )
    req = SystemTwoRequest(message="Write code for a quicksort algorithm in python")
    resp = engine.generate(req)

    assert isinstance(resp, SystemTwoResponse)
    assert "System 2 Offline Fallback" in resp.reply
    assert resp.telemetry.intent is not None
    assert resp.total_latency_ms > 0.0


def test_system_two_streaming_structure(mock_system_one_adapter):
    """Verify streaming generator yields initial telemetry on chunk 1."""
    engine = SystemTwoEngine(
        system_one_adapter=mock_system_one_adapter,
        ollama_url="http://127.0.0.1:59999",
        timeout_sec=1.0,
    )
    req = SystemTwoRequest(message="Hello System Two")
    chunks = list(engine.generate_stream(req))

    assert len(chunks) >= 1
    first_chunk = chunks[0]
    assert first_chunk.telemetry is not None
    assert first_chunk.telemetry.system_one_latency_ms >= 0.0


def test_system_two_live_ollama_if_running(mock_system_one_adapter):
    """Test actual live local Ollama Qwen2.5-Coder generation if Ollama service is active."""
    ollama_url = "http://127.0.0.1:11434"
    try:
        ping = httpx.get(f"{ollama_url}/api/tags", timeout=1.0)
        if ping.status_code != 200:
            pytest.skip("Ollama service not running on port 11434")
    except Exception:
        pytest.skip("Ollama service unreachable on port 11434")

    engine = SystemTwoEngine(
        system_one_adapter=mock_system_one_adapter,
        ollama_url=ollama_url,
        model_name="qwen2.5-coder:latest",
        timeout_sec=45.0,
    )
    req = SystemTwoRequest(
        message="write a one-line python lambda that squares an integer x",
        max_tokens=50,
    )
    resp = engine.generate(req)

    assert isinstance(resp, SystemTwoResponse)
    assert len(resp.reply) > 0
    assert "lambda" in resp.reply.lower() or "def" in resp.reply.lower()
    assert resp.model_name == "qwen2.5-coder:latest"
