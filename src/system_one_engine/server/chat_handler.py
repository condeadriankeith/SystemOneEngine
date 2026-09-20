"""Conversational Chat Handler backed by System One Decision Engine.

Orchestrates non-autoregressive triage (Intent, Sentiment Score, Escalation Flag)
and pairs it with responsive, intent-guided conversational generation.
"""

from dataclasses import dataclass
import time
from typing import Any
import httpx
from pydantic import BaseModel, Field

from system_one_engine.core.contracts import (
    BooleanRequest,
    ChoiceRequest,
    DecideRequest,
    ScoreRequest,
)

# Canonical intent criteria for System One conversational routing
CHAT_INTENT_CRITERIA = {
    "billing_refund": "Questions about payments, invoices, refunds, subscription charges, or cancellations.",
    "tech_support": "Software bugs, error messages, login failures, API issues, or technical malfunctions.",
    "order_shipping": "Tracking delivery, shipment delays, missing packages, or courier inquiries.",
    "policy_compliance": "Questions about security policies, terms of service, data privacy, or legal rules.",
    "feedback_complaint": "Negative feedback, customer frustration, compliments, or general service reviews.",
    "general_inquiry": "General information, greetings, conversational chitchat, or casual questions.",
}

# 5-level sentiment/urgency ladder
CHAT_SENTIMENT_CRITERIA = [
    "Delighted / Calm: Relaxed tone, expressing gratitude or casual curiosity.",
    "Inquiring / Neutral: Standard business inquiry with no notable emotional friction.",
    "Mildly Annoyed: Experiencing minor delays or inconveniences.",
    "Frustrated / Angry: Serious dissatisfaction, demanding fixes or complaining.",
    "Critical / Urgent: Immediate escalation, severe financial or security hazard.",
]


class ChatRequest(BaseModel):
    """User incoming chat message payload."""

    message: str = Field(..., min_length=1, description="User's input message.")
    history: list[dict[str, str]] = Field(
        default_factory=list, description="Optional prior conversation turns."
    )


from system_one_engine.system_two.contracts import DecisionTelemetry


class ChatResponse(BaseModel):
    """Conversational assistant response accompanied by System One telemetry."""

    reply: str
    telemetry: DecisionTelemetry
    total_latency_ms: float


class SystemOneChatHandler:
    """Conversational coordinator marrying System One triage with System Two generation."""

    def __init__(
        self,
        adapter: Any,
        ollama_url: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5-coder:latest",
    ) -> None:
        self.adapter = adapter
        from system_one_engine.system_two.engine import SystemTwoEngine
        self.system_two = SystemTwoEngine(
            system_one_adapter=adapter,
            ollama_url=ollama_url,
            model_name=model_name,
            timeout_sec=90.0,
        )

    def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process a user message through System One triage and System Two generation.

        Args:
            request: ChatRequest containing the user message and history.

        Returns:
            ChatResponse: Tailored reply and calibrated System One telemetry.
        """
        from system_one_engine.system_two.contracts import SystemTwoRequest

        s2_req = SystemTwoRequest(message=request.message, history=request.history)
        s2_resp = self.system_two.generate(s2_req)

        return ChatResponse(
            reply=s2_resp.reply,
            telemetry=s2_resp.telemetry,
            total_latency_ms=s2_resp.total_latency_ms,
        )

    def process_message_stream(self, request: ChatRequest):
        """Stream conversational tokens with initial System One decision telemetry.

        Args:
            request: ChatRequest.

        Yields:
            StreamChunk: Incremental token deltas and telemetry.
        """
        from system_one_engine.system_two.contracts import SystemTwoRequest

        s2_req = SystemTwoRequest(
            message=request.message, history=request.history, stream=True
        )
        yield from self.system_two.generate_stream(s2_req)

