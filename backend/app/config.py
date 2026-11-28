"""Centralized configuration via environment variables."""
from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", protected_namespaces=("settings_",))

    # No password in code: set DATABASE_URL / DATABASE_URL_SYNC in .env or the host's variables
    database_url: str = "postgresql+asyncpg://pc_user@localhost:5432/predictive_care"
    database_url_sync: str = "postgresql://pc_user@localhost:5432/predictive_care"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    jwt_secret: str = "dev-secret-change-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    groq_api_key: str = ""
    frontend_url: str = "http://localhost:3000"
    environment: str = "development"
    # Demo accounts (password = username). Unset means on everywhere except production.
    demo_logins: bool | None = None
    # Demo accounts act with their full role on the synthetic demo; true makes them view-only for actions.
    demo_read_only: bool = False
    # JSON object: {"username": {"password_hash": "<bcrypt>", "role": "admin", "name": "..."}}
    auth_users: str = ""
    log_level: str = "info"
    rate_limit_per_minute: int = 60
    model_dir: str = "data/models"
    # Pin a model version (models-<version>.pkl); empty = the one named in current.json
    model_version: str = ""
    # Machine token for /ops/drift and /ops/retrain (Authorization: Bearer). Unset = those routes return 503.
    ops_token: str = ""
    # Newest N patients in the live drift sample
    drift_max_patients: int = 20000
    hipaa_audit_enabled: bool = True
    session_timeout_minutes: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
