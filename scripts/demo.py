"""System One Engine - Live Interactive Demonstration Script.

Run this script to test the calibrated decision primitives (Choice, Score, Boolean, Decide)
using the local ONNX INT8 runtime:

    uv run python scripts/demo.py
"""

from pathlib import Path
import sys
import time

# Ensure Windows cp1252 terminals do not raise UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter
from system_one_engine.client import SystemOneClient
from system_one_engine.core.contracts import BooleanRequest, ChoiceRequest, ScoreRequest
from system_one_engine.models.backbone import TinyBackbone
from system_one_engine.models.export import export_and_quantize_all
from system_one_engine.models.system_one_model import SystemOneModel
from system_one_engine.training.dataset import SimpleVocabTokenizer


def main() -> None:
    print("=" * 70)
    print("[*] System One Decision Engine - Interactive Live Test")
    print("=" * 70)

    # 1. Ensure ONNX models exist
    onnx_dir = Path("models/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = onnx_dir / "encoder_int8.onnx"

    if not encoder_path.exists():
        print("\n[..] Exporting and quantizing local INT8 ONNX model artifacts...")
        t0 = time.perf_counter()
        backbone = TinyBackbone(vocab_size=500, hidden_size=32)
        model = SystemOneModel(backbone=backbone, projection_dim=16)
        paths = export_and_quantize_all(model, onnx_dir)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[OK] ONNX INT8 artifacts ready in {dt:.1f} ms -> {onnx_dir.resolve()}")
    else:
        paths = {
            "encoder": str(onnx_dir / "encoder_int8.onnx"),
            "heads": str(onnx_dir / "heads_int8.onnx"),
            "choice_head": str(onnx_dir / "choice_head_int8.onnx"),
        }
        print(f"\n[OK] Loaded existing INT8 ONNX models from {onnx_dir.resolve()}")

    # 2. Initialize Adapter & Client SDK
    tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
    adapter = ONNXRuntimeAdapter(
        encoder_path=paths["encoder"],
        heads_path=paths["heads"],
        choice_head_path=paths["choice_head"],
        tokenizer=tokenizer,
        num_threads=4,
    )
    client = SystemOneClient(adapter=adapter)
    print("[OK] SystemOneClient initialized in high-performance in-process mode (4 CPU threads).")

    # 3. Test Primitive 1: Choice[T] (Categorical Selection)
    print("\n" + "-" * 70)
    print("[1/4] TESTING PRIMITIVE: Choice[T] (Categorical Selection)")
    context_choice = "Customer email: 'My package arrived damaged with broken glass.'"
    criteria_choice = {
        "claims": "File an insurance damage claim",
        "billing": "Review payment or subscription",
        "shipping": "Check current package tracking",
    }
    t0 = time.perf_counter()
    choice_resp = client.choice(
        context=context_choice,
        instruction="Route to appropriate support department.",
        criteria=criteria_choice,
    )
    lat_choice = (time.perf_counter() - t0) * 1000.0

    print(f"Context:     \"{context_choice}\"")
    print(f"Selected:    [CHOICE] {choice_resp.choice} (Confidence: {choice_resp.confidence:.1%})")
    print(f"Probabilities:")
    for opt, prob in choice_resp.probabilities.items():
        bar = "#" * int(prob * 20)
        print(f"  - {opt:<10} : {prob:.3f} |{bar:<20}|")
    print(f"Latency:     {lat_choice:.2f} ms")

    # 4. Test Primitive 2: Boolean (Binary Assertion)
    print("\n" + "-" * 70)
    print("[2/4] TESTING PRIMITIVE: Boolean (Binary Assertion)")
    context_bool = "Draft response: 'Your account password has been reset to temporary123.'"
    instruction_bool = "Does this draft contain sensitive credential security concerns?"
    t0 = time.perf_counter()
    bool_resp = client.boolean(context=context_bool, instruction=instruction_bool)
    lat_bool = (time.perf_counter() - t0) * 1000.0

    print(f"Context:     \"{context_bool}\"")
    print(f"Instruction: \"{instruction_bool}\"")
    print(f"Result:      [BOOLEAN] Confirmed: {bool_resp.confirmed} (Probability: {bool_resp.probability:.1%}, Confidence: {bool_resp.confidence:.1%})")
    print(f"Latency:     {lat_bool:.2f} ms")

    # 5. Test Primitive 3: Score (Ordinal Rating & Continuous Expectation)
    print("\n" + "-" * 70)
    print("[3/4] TESTING PRIMITIVE: Score (Continuous Expectation E[S])")
    context_score = "Agent resolved the customer issue on the first call and refunded the fee."
    criteria_score = ["Very Poor", "Poor", "Neutral", "Good", "Outstanding"]
    t0 = time.perf_counter()
    score_resp = client.score(
        context=context_score,
        instruction="Rate customer satisfaction level.",
        criteria=criteria_score,
    )
    lat_score = (time.perf_counter() - t0) * 1000.0

    print(f"Context:     \"{context_score}\"")
    print(f"Expected:    [SCORE] {score_resp.score:.2f} / 4.00 (Confidence: {score_resp.confidence:.1%})")
    print(f"Probabilities:")
    for key, desc in score_resp.legend.items():
        prob = score_resp.probabilities.get(key, 0.0)
        bar = "#" * int(prob * 20)
        print(f"  - [{key}] {desc:<14} : {prob:.3f} |{bar:<20}|")
    print(f"Latency:     {lat_score:.2f} ms")

    # 6. Test Primitive 4: Speculative Fan-out (/decide)
    print("\n" + "-" * 70)
    print("[4/4] TESTING PRIMITIVE: Speculative Fan-out (/decide)")
    decide_context = "User: 'I received the wrong item, need a refund, and someone was rude.'"
    t0 = time.perf_counter()
    decide_resp = client.decide(
        context=decide_context,
        questions={
            "dept": ChoiceRequest(
                context=decide_context,
                instruction="Route to team.",
                criteria=criteria_choice,
            ),
            "is_security_concern": BooleanRequest(
                context=decide_context,
                instruction="Does this mention a security vulnerability?",
            ),
            "satisfaction": ScoreRequest(
                context=decide_context,
                instruction="Rate customer satisfaction level.",
                criteria=criteria_score,
            ),
        },
    )
    lat_decide = (time.perf_counter() - t0) * 1000.0

    print(f"Context:     \"{decide_context}\"")
    print(f"Concurrent Questions Evaluated: {len(decide_resp.answers)}")
    print(f"Reported Engine Latency:        {decide_resp.latency_ms:.2f} ms (Single Context Pass)")
    print(f"Total Wall-Clock Latency:       {lat_decide:.2f} ms")

    print("\n" + "=" * 70)
    print("[SUCCESS] All 4 Decision Primitives Successfully Evaluated!")
    print("=" * 70)
    print("\nOther Ways to Test:")
    print("  1. Run complete automated test suite (62 tests):")
    print("     uv run pytest")
    print("\n  2. Start the local FastAPI microservice:")
    print("     uv run uvicorn system_one_engine.server.app:app --reload")
    print("     Then open in browser: http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    main()
