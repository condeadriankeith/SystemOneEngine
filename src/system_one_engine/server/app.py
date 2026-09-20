"""FastAPI REST Microservice for System One Decision Engine.

Exposes high-speed, non-autoregressive decision endpoints:
- POST /choice: Categorical selection.
- POST /score: Ordinal ladder evaluation.
- POST /boolean: Binary assertion under doubt.
- POST /decide: High-throughput speculative fan-out.
- GET /health: Health status and backend engine inspection.
- GET /metrics: Request counts and latency diagnostics.
"""

from pathlib import Path
import time
from typing import Any
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter
from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter
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
from system_one_engine.server.chat_handler import (
    ChatRequest,
    ChatResponse,
    SystemOneChatHandler,
)
from system_one_engine.server.chat_ui import get_chat_html
from system_one_engine.training.dataset import SimpleVocabTokenizer


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str
    backend: str
    version: str = "0.1.0"


class MetricsResponse(BaseModel):
    """Server execution metrics."""

    requests_total: int
    avg_latency_ms: float
    backend: str


class ServerMetrics:
    """Thread-safe metrics accumulator for tracking latency and request volume."""

    def __init__(self) -> None:
        self.requests_total: int = 0
        self.total_latency_ms: float = 0.0

    def record(self, latency_ms: float) -> None:
        self.requests_total += 1
        self.total_latency_ms += latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total


def create_app(adapter: Any | None = None) -> FastAPI:
    """Application factory creating the FastAPI decision service.

    Args:
        adapter: Pre-configured adapter instance (ONNXRuntimeAdapter or OllamaSystemOneAdapter).
                 Defaults to local ONNXRuntimeAdapter if models/onnx exists, else OllamaSystemOneAdapter(mock_mode=True).

    Returns:
        FastAPI: Configured web application.
    """
    app = FastAPI(
        title="System One Decision Engine",
        version="0.1.0",
        description="Fast, calibrated non-autoregressive decision engine for software systems and autonomous agents.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if adapter is None:
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
        else:
            adapter = OllamaSystemOneAdapter(mock_mode=True)

    app.state.adapter = adapter
    app.state.metrics = ServerMetrics()
    app.state.chat_handler = SystemOneChatHandler(adapter=adapter)

    def get_adapter(request: Request) -> Any:
        return request.app.state.adapter

    def get_metrics(request: Request) -> ServerMetrics:
        return request.app.state.metrics

    def get_chat_handler(request: Request) -> SystemOneChatHandler:
        return request.app.state.chat_handler

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/chat-ui", response_class=HTMLResponse, tags=["Chat"])
    async def chat_ui() -> HTMLResponse:
        """Interactive Web Chat interface for System One Engine."""
        return HTMLResponse(get_chat_html())

    @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def handle_chat(
        payload: ChatRequest,
        chat_handler: SystemOneChatHandler = Depends(get_chat_handler),
        metrics: ServerMetrics = Depends(get_metrics),
    ) -> ChatResponse:
        """Conversational chat turn evaluated with System One decision primitives."""
        t0 = time.perf_counter()
        try:
            res = chat_handler.process_message(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat processing failed: {exc}",
            ) from exc
        finally:
            metrics.record((time.perf_counter() - t0) * 1000.0)
        return res

    @app.post("/chat/stream", tags=["Chat"])
    async def handle_chat_stream(
        payload: ChatRequest,
        chat_handler: SystemOneChatHandler = Depends(get_chat_handler),
    ):
        """Stream conversational tokens with Server-Sent Events."""
        import json
        from fastapi.responses import StreamingResponse

        def event_generator():
            for chunk in chat_handler.process_message_stream(payload):
                data = chunk.model_dump()
                yield f"data: {json.dumps(data)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
    async def health_check(adapter: Any = Depends(get_adapter)) -> HealthResponse:
        """Inspect health status and active backend engine."""
        backend_name = adapter.__class__.__name__
        return HealthResponse(status="healthy", backend=backend_name)

    @app.get("/metrics", response_model=MetricsResponse, tags=["Diagnostics"])
    async def server_metrics(
        metrics: ServerMetrics = Depends(get_metrics),
        adapter: Any = Depends(get_adapter),
    ) -> MetricsResponse:
        """Inspect latency metrics and request counters."""
        return MetricsResponse(
            requests_total=metrics.requests_total,
            avg_latency_ms=round(metrics.avg_latency_ms, 2),
            backend=adapter.__class__.__name__,
        )

    @app.post("/choice", response_model=ChoiceResponse, tags=["Decisions"])
    async def evaluate_choice(
        payload: ChoiceRequest,
        adapter: Any = Depends(get_adapter),
        metrics: ServerMetrics = Depends(get_metrics),
    ) -> ChoiceResponse:
        """Evaluate an unordered categorical Choice decision."""
        t0 = time.perf_counter()
        try:
            result = adapter.evaluate_choice(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Choice evaluation failed: {exc}",
            ) from exc
        finally:
            metrics.record((time.perf_counter() - t0) * 1000.0)
        return result

    @app.post("/boolean", response_model=BooleanResponse, tags=["Decisions"])
    async def evaluate_boolean(
        payload: BooleanRequest,
        adapter: Any = Depends(get_adapter),
        metrics: ServerMetrics = Depends(get_metrics),
    ) -> BooleanResponse:
        """Evaluate a binary Boolean condition under doubt."""
        t0 = time.perf_counter()
        try:
            result = adapter.evaluate_boolean(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Boolean evaluation failed: {exc}",
            ) from exc
        finally:
            metrics.record((time.perf_counter() - t0) * 1000.0)
        return result

    @app.post("/score", response_model=ScoreResponse, tags=["Decisions"])
    async def evaluate_score(
        payload: ScoreRequest,
        adapter: Any = Depends(get_adapter),
        metrics: ServerMetrics = Depends(get_metrics),
    ) -> ScoreResponse:
        """Evaluate an ordinal position on an ascending ladder."""
        t0 = time.perf_counter()
        try:
            result = adapter.evaluate_score(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Score evaluation failed: {exc}",
            ) from exc
        finally:
            metrics.record((time.perf_counter() - t0) * 1000.0)
        return result

    @app.post("/decide", response_model=DecideResponse, tags=["Decisions"])
    async def evaluate_decide(
        payload: DecideRequest,
        adapter: Any = Depends(get_adapter),
        metrics: ServerMetrics = Depends(get_metrics),
    ) -> DecideResponse:
        """Speculative fan-out evaluating multiple heterogeneous questions over a shared context."""
        t0 = time.perf_counter()
        try:
            result = adapter.evaluate_decide(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Decide fan-out failed: {exc}",
            ) from exc
        finally:
            metrics.record((time.perf_counter() - t0) * 1000.0)
        return result

    return app


# Default application instance
app = create_app()
