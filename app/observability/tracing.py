from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("otel")


def setup_tracing(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.enable_tracing:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("otel_fastapi_instrumented")
    except Exception as exc:
        logger.warning("otel_setup_failed", error=str(exc))
