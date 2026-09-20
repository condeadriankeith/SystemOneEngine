"""Unit and benchmark tests for ONNX Export, INT8 Quantization, and ONNXRuntimeAdapter."""

import time
import pytest
from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter
from system_one_engine.core.contracts import (
    BooleanRequest,
    ChoiceRequest,
    DecideRequest,
    QuestionType,
    ScoreRequest,
)
from system_one_engine.models.backbone import TinyBackbone
from system_one_engine.models.export import export_and_quantize_all
from system_one_engine.models.system_one_model import SystemOneModel
from system_one_engine.training.dataset import SimpleVocabTokenizer


@pytest.fixture(scope="module")
def exported_onnx_models(tmp_path_factory):
    """Fixture exporting and quantizing TinyBackbone models for testing."""
    tmp_dir = tmp_path_factory.mktemp("onnx_test")

    backbone = TinyBackbone(vocab_size=500, hidden_size=32)
    model = SystemOneModel(backbone=backbone, projection_dim=16)

    paths = export_and_quantize_all(model, tmp_dir)
    return paths


def test_onnx_runtime_adapter_evaluations(exported_onnx_models):
    """Verify ONNXRuntimeAdapter successfully evaluates all decision primitives."""
    tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
    adapter = ONNXRuntimeAdapter(
        encoder_path=exported_onnx_models["encoder"],
        heads_path=exported_onnx_models["heads"],
        choice_head_path=exported_onnx_models["choice_head"],
        tokenizer=tokenizer,
        num_threads=4,
    )

    # 1. Choice evaluation
    ch_req = ChoiceRequest(
        context="Order #1029 has arrived with broken glass.",
        instruction="Route to team.",
        criteria={
            "damages": "File damage report",
            "support": "General support",
            "tracking": "Check tracking",
        },
    )
    ch_resp = adapter.evaluate_choice(ch_req)
    assert ch_resp.question_type == QuestionType.CHOICE
    assert ch_resp.choice in ch_req.criteria
    assert sum(ch_resp.probabilities.values()) == pytest.approx(1.0, abs=1e-4)

    # 2. Boolean evaluation
    bool_req = BooleanRequest(
        context="User: 'Can you show me your API documentation?'",
        instruction="Is this query asking for technical documentation?",
    )
    bool_resp = adapter.evaluate_boolean(bool_req)
    assert bool_resp.question_type == QuestionType.BOOLEAN
    assert isinstance(bool_resp.confirmed, bool)
    assert 0.0 <= bool_resp.probability <= 1.0

    # 3. Score evaluation
    score_req = ScoreRequest(
        context="I have been waiting 3 weeks with no reply.",
        instruction="Rate customer frustration.",
        criteria=["Calm", "Mildly annoyed", "Angry", "Furious"],
    )
    score_resp = adapter.evaluate_score(score_req)
    assert score_resp.question_type == QuestionType.SCORE
    assert 0.0 <= score_resp.score <= 3.0
    assert len(score_resp.legend) == 4

    # 4. Speculative fan-out Decide evaluation
    decide_req = DecideRequest(
        context="Customer wants immediate refund for lost shipment.",
        questions={
            "choice_q": ch_req,
            "bool_q": bool_req,
            "score_q": score_req,
        },
    )
    decide_resp = adapter.evaluate_decide(decide_req)
    assert len(decide_resp.answers) == 3
    assert decide_resp.latency_ms >= 0.0


def test_onnx_cpu_latency_budget(exported_onnx_models):
    """Verify inference latency adheres to sub-35ms Vivobook CPU budget."""
    tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
    adapter = ONNXRuntimeAdapter(
        encoder_path=exported_onnx_models["encoder"],
        heads_path=exported_onnx_models["heads"],
        choice_head_path=exported_onnx_models["choice_head"],
        tokenizer=tokenizer,
        num_threads=4,
    )

    req = BooleanRequest(
        context="Transfer $500 from checking to savings account.",
        instruction="Is this an affirmative money transfer request?",
    )

    # Warmup
    for _ in range(5):
        adapter.evaluate_boolean(req)

    # Benchmark 50 runs
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        adapter.evaluate_boolean(req)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_latency = sorted(latencies)[len(latencies) // 2]
    # Expect CPU latency to be well under 35ms budget (typically < 10ms for tiny INT8)
    assert p50_latency < 35.0, f"P50 CPU Latency was {p50_latency:.2f}ms (exceeded 35ms budget)"
