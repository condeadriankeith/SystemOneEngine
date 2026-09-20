"""System Two Generative Engine.

Fuses fast, calibrated non-autoregressive decision triage (System 1)
with local autoregressive Large Language Models (System 2, e.g. Ollama Qwen 2.5-Coder).
"""

import json
import logging
import time
from typing import Any, Iterator
import httpx

from system_one_engine.core.contracts import (
    BooleanRequest,
    ChoiceRequest,
    DecideRequest,
    ScoreRequest,
)
from system_one_engine.system_two.contracts import (
    DecisionTelemetry,
    StreamChunk,
    SystemTwoRequest,
    SystemTwoResponse,
)

logger = logging.getLogger(__name__)

# Specialized System 1 Intent Schema for Software & Agentic Workflows
SYSTEM_TWO_INTENTS = {
    "code_generation": "Writing, implementing, or generating code, complete programs, games, algorithms, or scripts.",
    "debugging": "Analyzing errors, bug fixing, stack traces, troubleshooting broken code, or fixing failures.",
    "code_explanation": "Explaining programming concepts, syntax, libraries, architectural designs, or algorithms.",
    "refactoring": "Optimizing, cleaning up, modernizing, or restructuring existing code without changing behavior.",
    "general_inquiry": "Greetings, general knowledge, casual conversation, questions, or non-coding tasks.",
}

SYSTEM_TWO_COMPLEXITY_LADDER = [
    "Trivial: One-liner syntax, greeting, or micro-snippet.",
    "Simple: Single function, basic script, or utility function.",
    "Intermediate: Standalone application, game (e.g. Snake/Pong), class, or multi-step logic.",
    "Advanced: Complex data structures, concurrency, optimization, or multi-component architecture.",
    "Critical: Production-grade system, security-sensitive module, or high-reliability infrastructure.",
]


class SystemTwoEngine:
    """Deliberative Generative Engine steered by System One Decision Primitives."""

    def __init__(
        self,
        system_one_adapter: Any,
        ollama_url: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5-coder:latest",
        timeout_sec: float = 90.0,
    ) -> None:
        self.system_one = system_one_adapter
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    def evaluate_system_one(self, prompt: str) -> DecisionTelemetry:
        """Run non-autoregressive triage across Intent, Complexity, and Safety in a single pass.

        Args:
            prompt: User message string.

        Returns:
            DecisionTelemetry: Calibrated decisions and latency.
        """
        t0 = time.perf_counter()
        decide_req = DecideRequest(
            context=prompt,
            questions={
                "intent": ChoiceRequest(
                    context=prompt,
                    instruction="Identify the primary computational task requested.",
                    criteria=SYSTEM_TWO_INTENTS,
                ),
                "complexity": ScoreRequest(
                    context=prompt,
                    instruction="Estimate the architectural complexity and depth required.",
                    criteria=SYSTEM_TWO_COMPLEXITY_LADDER,
                ),
                "safety_violation": BooleanRequest(
                    context=prompt,
                    instruction="Does this prompt request destructive system actions (e.g. rm -rf, wiping disks), malware, or credential harvesting?",
                ),
            },
        )

        decide_resp = self.system_one.evaluate_decide(decide_req)
        s1_latency = (time.perf_counter() - t0) * 1000.0

        choice_ans = decide_resp.answers["intent"]
        score_ans = decide_resp.answers["complexity"]
        bool_ans = decide_resp.answers["safety_violation"]

        score_idx = int(round(score_ans.score))
        score_idx = max(0, min(score_idx, len(SYSTEM_TWO_COMPLEXITY_LADDER) - 1))
        level_name = SYSTEM_TWO_COMPLEXITY_LADDER[score_idx].split(":")[0].strip()

        return DecisionTelemetry(
            intent=choice_ans.choice,
            intent_confidence=choice_ans.confidence,
            sentiment_score=score_ans.score,
            sentiment_level=level_name,
            sentiment_confidence=score_ans.confidence,
            needs_escalation=bool_ans.confirmed,
            escalation_confidence=bool_ans.confidence,
            system_one_latency_ms=s1_latency,
        )

    def build_steered_prompt(
        self, prompt: str, telemetry: DecisionTelemetry, history: list[dict[str, str]]
    ) -> str:
        """Construct prompt conditioned on System 1's calibrated decisions.

        Args:
            prompt: User prompt.
            telemetry: System 1 decisions.
            history: Prior turns.

        Returns:
            str: Steered prompt for System 2 LLM.
        """
        intent = telemetry.intent
        complexity = telemetry.sentiment_level

        if intent == "code_generation":
            system_instruction = (
                f"You are an expert staff software engineer. System 1 has classified this task as [CODE_GENERATION] "
                f"with complexity level [{complexity}]. Provide clean, idiomatic, fully functional, and runnable code. "
                f"Begin immediately with the code block. DO NOT output conversational preamble, explanations, or filler. "
                f"DO NOT use placeholder comments like '// TODO' or '...'. Implement the full working solution."
            )
        elif intent == "debugging":
            system_instruction = (
                f"You are a master systems debugger. System 1 classified this task as [DEBUGGING] ({complexity}). "
                f"Isolate the root cause step-by-step and provide the verified fix."
            )
        elif intent == "refactoring":
            system_instruction = (
                f"You are a software architect. System 1 classified this task as [REFACTORING] ({complexity}). "
                f"Improve modularity and maintainability while preserving exact behavior."
            )
        else:
            system_instruction = (
                f"You are an intelligent technical assistant guided by a fast System One decision engine. "
                f"Respond helpfully, accurately, and concisely."
            )

        history_context = ""
        if history:
            for turn in history[-4:]:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                history_context += f"{role}: {content}\n"

        full_prompt = (
            f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
            f"{history_context}"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return full_prompt

    def generate(self, request: SystemTwoRequest) -> SystemTwoResponse:
        """Execute full System 1 triage followed by System 2 deliberate generation.

        Args:
            request: SystemTwoRequest containing prompt and parameters.

        Returns:
            SystemTwoResponse: Generated output with decision telemetry.
        """
        t_start = time.perf_counter()
        telemetry = self.evaluate_system_one(request.message)

        # Safety Guardrail Interception
        if telemetry.needs_escalation and telemetry.escalation_confidence > 0.80:
            refusal_msg = (
                "⚠️ [System 1 Security Guardrail Interception] This prompt was flagged as potentially "
                "destructive or violating safety constraints. Execution was terminated before sending "
                "to System 2."
            )
            return SystemTwoResponse(
                reply=refusal_msg,
                telemetry=telemetry,
                model_name="system_one_guardrail",
                total_latency_ms=(time.perf_counter() - t_start) * 1000.0,
                token_count=0,
            )

        # Build conditioned prompt
        steered_prompt = self.build_steered_prompt(
            request.message, telemetry, request.history
        )

        temperature = request.temperature
        if temperature is None:
            temperature = 0.2 if telemetry.intent in ("code_generation", "debugging") else 0.7

        # Call Ollama
        try:
            resp = httpx.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": steered_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_thread": 8,
                        "num_ctx": 2048,
                        "num_predict": request.max_tokens or 1024,
                    },
                },
                timeout=self.timeout_sec,
            )
            if resp.status_code == 200:
                data = resp.json()
                reply_text = data.get("response", "").strip()
                tokens = data.get("eval_count", None)
                total_ms = (time.perf_counter() - t_start) * 1000.0
                return SystemTwoResponse(
                    reply=reply_text,
                    telemetry=telemetry,
                    model_name=self.model_name,
                    total_latency_ms=total_ms,
                    token_count=tokens,
                )
        except Exception as exc:
            logger.warning("Ollama generation call failed (%s). Using fallback.", exc)

        # Fallback if Ollama is unreachable
        fallback_reply = self._fallback_synthesis(request.message, telemetry)
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return SystemTwoResponse(
            reply=fallback_reply,
            telemetry=telemetry,
            model_name="system_two_offline_fallback",
            total_latency_ms=total_ms,
            token_count=len(fallback_reply.split()),
        )

    def generate_stream(self, request: SystemTwoRequest) -> Iterator[StreamChunk]:
        """Stream System 2 tokens in real time, beginning with immediate System 1 telemetry.

        Args:
            request: SystemTwoRequest.

        Yields:
            StreamChunk: Incremental token deltas.
        """
        telemetry = self.evaluate_system_one(request.message)

        # Emit initial chunk carrying System 1 telemetry
        yield StreamChunk(delta="", done=False, telemetry=telemetry)

        # Guardrail check
        if telemetry.needs_escalation and telemetry.escalation_confidence > 0.80:
            yield StreamChunk(
                delta="⚠️ [System 1 Security Guardrail Interception] Execution terminated for safety.",
                done=True,
                telemetry=telemetry,
            )
            return

        steered_prompt = self.build_steered_prompt(
            request.message, telemetry, request.history
        )
        temperature = request.temperature
        if temperature is None:
            temperature = 0.2 if telemetry.intent in ("code_generation", "debugging") else 0.7

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": steered_prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_thread": 8,
                            "num_ctx": 2048,
                            "num_predict": request.max_tokens or 1024,
                        },
                    },
                ) as response:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        chunk_json = json.loads(line)
                        token = chunk_json.get("response", "")
                        done = chunk_json.get("done", False)
                        if token:
                            yield StreamChunk(delta=token, done=False)
                        if done:
                            yield StreamChunk(delta="", done=True)
                            return
        except Exception as exc:
            logger.warning("Streaming failed (%s). Emitting offline response.", exc)
            fallback = self._fallback_synthesis(request.message, telemetry)
            yield StreamChunk(delta=fallback, done=True)

    def _fallback_synthesis(self, prompt: str, telemetry: DecisionTelemetry) -> str:
        """Deterministic response if local LLM is temporarily unreachable."""
        return (
            f"[System 2 Offline Fallback]\n"
            f"System 1 evaluated your request in {telemetry.system_one_latency_ms:.2f} ms:\n"
            f"- Intent: {telemetry.intent} ({telemetry.intent_confidence:.1%})\n"
            f"- Complexity: {telemetry.sentiment_level} ({telemetry.sentiment_score:.2f}/4.0)\n\n"
            f"To enable live code generation, please start Ollama on {self.ollama_url} with {self.model_name}."
        )
