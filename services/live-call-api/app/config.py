"""Service-level settings, read from environment variables.

Kept separate from config/risk_formula.yaml on purpose: this file is about
*where the service runs* (log level, which config file to load); the YAML
file is about *how it scores* (the formula). Mixing the two would make it
harder to reason about which changes affect reproducibility of a score.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    risk_config_path: str = "config/risk_formula.yaml"
    history_db_path: str = "data/sessions.db"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
