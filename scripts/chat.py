"""System One Interactive Terminal Chat REPL.

Allows testing the decision engine by typing conversational messages directly:

    uv run python scripts/chat.py
"""

from pathlib import Path
import sys
import time

# Ensure Windows cp1252 consoles do not crash on special characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter
from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter
from system_one_engine.server.chat_handler import ChatRequest, SystemOneChatHandler
from system_one_engine.training.dataset import SimpleVocabTokenizer


def load_best_adapter():
    """Load ONNX INT8 engine if available, otherwise fall back to Ollama adapter."""
    onnx_dir = Path("models/onnx")
    enc_path = onnx_dir / "encoder_int8.onnx"
    heads_path = onnx_dir / "heads_int8.onnx"
    ch_path = onnx_dir / "choice_head_int8.onnx"

    if enc_path.exists() and heads_path.exists() and ch_path.exists():
        tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
        adapter = ONNXRuntimeAdapter(
            encoder_path=str(enc_path),
            heads_path=str(heads_path),
            choice_head_path=str(ch_path),
            tokenizer=tokenizer,
            num_threads=4,
        )
        return adapter, "ONNX INT8 CPU Engine (4 threads)"

    return OllamaSystemOneAdapter(mock_mode=True), "Ollama Fallback Adapter (Mock Mode)"


def main() -> None:
    print("=" * 72)
    print("💬 System One Decision Engine - Interactive Chat REPL")
    print("=" * 72)

    adapter, engine_desc = load_best_adapter()
    print(f"[*] Loaded Backend: {engine_desc}")
    print("[*] Type any message to evaluate intent, urgency, and policy escalation.")
    print("[*] Commands: 'exit', 'quit', or Ctrl+C to terminate.")
    print("=" * 72)

    chat_handler = SystemOneChatHandler(adapter=adapter)

    while True:
        try:
            print()
            user_input = input("You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting System One Chat. Goodbye!")
                break

            # Stream System 1 triage followed by live System 2 generation
            req = ChatRequest(message=user_input)
            telemetry_shown = False
            first_token = True

            for chunk in chat_handler.process_message_stream(req):
                if chunk.telemetry and not telemetry_shown:
                    telemetry_shown = True
                    telem = chunk.telemetry
                    esc_text = "ESCALATION REQUIRED" if telem.needs_escalation else "PASSED (SAFE)"
                    print()
                    print(f"  [⚡ System 1 Decision Reflex | Latency: {telem.system_one_latency_ms:.2f} ms]")
                    print(f"  • Intent:     {telem.intent} (Confidence: {telem.intent_confidence:.1%})")
                    print(f"  • Complexity: {telem.sentiment_score:.2f} / 4.0 ({telem.sentiment_level})")
                    print(f"  • Guardrail:  {esc_text}")
                    print()
                    print("Assistant > ", end="", flush=True)

                if chunk.delta:
                    sys.stdout.write(chunk.delta)
                    sys.stdout.flush()

            print()

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as exc:
            print(f"\n[Error] {exc}")


if __name__ == "__main__":
    main()
