"""SystemOneEngine - Server module.

FastAPI REST microservice providing high-performance decision endpoints.
"""

from system_one_engine.server.app import (
    HealthResponse,
    MetricsResponse,
    ServerMetrics,
    app,
    create_app,
)

__all__ = [
    "app",
    "create_app",
    "HealthResponse",
    "MetricsResponse",
    "ServerMetrics",
]
