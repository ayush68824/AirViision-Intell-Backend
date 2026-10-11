from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_secret_key: str = "dev-secret-change-me"
    api_base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000"

    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 14

    database_url: str = "postgresql+asyncpg://airvision:airvision@localhost:5432/airvision"
    database_url_sync: str = "postgresql://airvision:airvision@localhost:5432/airvision"
    redis_url: str = "redis://localhost:6379/0"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "airvision_scientific"

    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_reasoning_model: str = "gpt-4o"
    llm_fast_model: str = "gpt-4o-mini"
    llm_vision_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    openweather_api_key: str = ""
    waqi_api_key: str = ""
    aqicn_token: str = ""

    rate_limit_per_minute: int = 60
    rate_limit_agent_per_minute: int = 20

    otel_exporter_otlp_endpoint: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "airvision-intelligence"
    enable_tracing: bool = False

    use_live_osm: bool = True
    use_mock_aqi_fallback: bool = True
    enable_vision_agent: bool = True
    enable_voice: bool = False

    artifacts_dir: str = "artifacts"
    default_city: str = "Delhi"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
