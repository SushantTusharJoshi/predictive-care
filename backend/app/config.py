"""Centralized configuration via environment variables."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://pc_user:pc_local_dev_2024@localhost:5432/predictive_care"
    database_url_sync: str = "postgresql://pc_user:pc_local_dev_2024@localhost:5432/predictive_care"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    jwt_secret: str = "dev-secret-change-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    groq_api_key: str = ""
    frontend_url: str = "http://localhost:3000"
    environment: str = "development"
    log_level: str = "info"
    rate_limit_per_minute: int = 60
    model_dir: str = "data/models"
    hipaa_audit_enabled: bool = True
    session_timeout_minutes: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
