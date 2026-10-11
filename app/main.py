from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import agent, aqi, auth, forecast, health, routes_api
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.ml.forecaster import get_forecaster
from app.observability.tracing import setup_tracing
from app.rag.knowledge_base import seed_qdrant_if_possible

logger = get_logger("main")


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    settings = get_settings()
    get_forecaster()
    rag = seed_qdrant_if_possible()
    logger.info("startup_complete", version=__version__, rag=rag, env=settings.app_env)
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AirVision Intelligence API",
        description=(
            "AirVision Intelligence API — air-quality forecasts, geospatial routing, "
            "and health-aware environmental decision support."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    # Allow both localhost and 127.0.0.1 — browsers treat them as different origins.
    origins = settings.cors_origin_list or []
    for host in ("http://localhost:3000", "http://127.0.0.1:3000"):
        if host not in origins:
            origins.append(host)
    if not settings.is_production and "*" not in origins:
        # Dev-friendly fallback if Origin header is unexpected
        allow_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    else:
        allow_origin_regex = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_origin_regex=allow_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    setup_tracing(app)

    app.include_router(health.router)
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(aqi.router, prefix="/api/v1")
    app.include_router(forecast.router, prefix="/api/v1")
    app.include_router(routes_api.router, prefix="/api/v1")
    app.include_router(agent.router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {
            "name": "AirVision Intelligence",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
