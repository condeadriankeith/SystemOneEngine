"""Typed Python Client SDK for System One Decision Engine.

Supports:
- Remote HTTP microservice calls via httpx.
- In-process zero-copy direct execution via loaded backend adapters.
- Strict Pydantic v2 return models: ChoiceResponse, ScoreResponse, BooleanResponse, DecideResponse.
- Synchronous and asynchronous interfaces.
"""

from typing import Any
import httpx

from system_one_engine.core.contracts import (
    BooleanRequest,
    BooleanResponse,
    ChoiceRequest,
    ChoiceResponse,
    DecideRequest,
    DecideResponse,
    ScoreRequest,
    ScoreResponse,
)


class SystemOneClient:
    """Client SDK for evaluating calibrated System One decisions.

    Attributes:
        base_url: Optional URL of remote FastAPI microservice (e.g. 'http://127.0.0.1:8000').
        adapter: Optional in-process adapter (ONNXRuntimeAdapter or OllamaSystemOneAdapter).
        timeout: Network timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        adapter: Any | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Initialize client in either HTTP mode or in-process mode."""
        if base_url is None and adapter is None:
            base_url = "http://127.0.0.1:8000"

        self.base_url = base_url.rstrip("/") if base_url else None
        self.adapter = adapter
        self.timeout = timeout
        self._http_client: httpx.Client | None = None

    @property
    def http_client(self) -> httpx.Client:
        """Lazily initialize persistent HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.Client(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._http_client

    def close(self) -> None:
        """Close underlying HTTP connections."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> "SystemOneClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        """Check service health."""
        if self.adapter is not None:
            return {"status": "healthy", "backend": self.adapter.__class__.__name__}

        resp = self.http_client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def choice(
        self,
        context: str,
        instruction: str,
        criteria: dict[str, str],
        temperature: float = 1.0,
    ) -> ChoiceResponse:
        """Evaluate a categorical Choice decision."""
        req = ChoiceRequest(
            context=context,
            instruction=instruction,
            criteria=criteria,
            temperature=temperature,
        )

        if self.adapter is not None:
            return self.adapter.evaluate_choice(req)

        resp = self.http_client.post("/choice", json=req.model_dump())
        resp.raise_for_status()
        return ChoiceResponse.model_validate(resp.json())

    def score(
        self,
        context: str,
        instruction: str,
        criteria: list[str],
        temperature: float = 1.0,
    ) -> ScoreResponse:
        """Evaluate an ordinal position on an ascending Score ladder."""
        req = ScoreRequest(
            context=context,
            instruction=instruction,
            criteria=criteria,
            temperature=temperature,
        )

        if self.adapter is not None:
            return self.adapter.evaluate_score(req)

        resp = self.http_client.post("/score", json=req.model_dump())
        resp.raise_for_status()
        return ScoreResponse.model_validate(resp.json())

    def boolean(
        self,
        context: str,
        instruction: str,
        criteria: dict[str, str] | None = None,
        temperature: float = 1.0,
    ) -> BooleanResponse:
        """Evaluate a binary Boolean assertion under uncertainty."""
        req = BooleanRequest(
            context=context,
            instruction=instruction,
            criteria=criteria,
            temperature=temperature,
        )

        if self.adapter is not None:
            return self.adapter.evaluate_boolean(req)

        resp = self.http_client.post("/boolean", json=req.model_dump())
        resp.raise_for_status()
        return BooleanResponse.model_validate(resp.json())

    def decide(
        self,
        context: str,
        questions: dict[str, ChoiceRequest | ScoreRequest | BooleanRequest],
    ) -> DecideResponse:
        """Evaluate speculative fan-out questions against shared context."""
        req = DecideRequest(context=context, questions=questions)

        if self.adapter is not None:
            return self.adapter.evaluate_decide(req)

        resp = self.http_client.post("/decide", json=req.model_dump())
        resp.raise_for_status()
        return DecideResponse.model_validate(resp.json())
