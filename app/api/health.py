from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from app import __version__
from app.core.config import get_settings
from app.models.schemas import StatusResponse
from app.rag.knowledge_base import seed_qdrant_if_possible

router = APIRouter(tags=["system"])

REQUESTS = Counter("airvision_requests_total", "Total API requests", ["route"])


@router.get("/health", response_model=StatusResponse)
async def health():
    settings = get_settings()
    rag_status = seed_qdrant_if_possible()
    services = {
        "api": "ok",
        "rag": rag_status,
        "forecaster": "ok",
        "redis": "optional",
        "postgres": "optional",
    }
    return StatusResponse(
        status="ok",
        version=__version__,
        llm_configured=settings.llm_configured,
        services=services,
    )


@router.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
