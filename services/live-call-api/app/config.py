"""Service-level settings, read from environment variables.

Kept separate from config/risk_formula.yaml on purpose: this file is about
*where the service runs* (log level, which config file to load); the YAML
file is about *how it scores* (the formula). Mixing the two would make it
harder to reason about which changes affect reproducibility of a score.
"""
from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    risk_config_path: str = "config/risk_formula.yaml"
    history_db_path: str = "data/sessions.db"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    # Empty by default: this app's UI/API/WebSocket code all uses
    # root-relative paths (fetch("/v1/..."), WebSocket to
    # `${location.host}/v1/stream/...`), which only resolve correctly when
    # the app is served at its origin's root. Set this (e.g. "/satya-vani")
    # ONLY when Traefik (or another reverse proxy) path-routes this app
    # under a prefix on a shared domain — see
    # deploy/portainer/satya-vani.yml's own note. app/main.py's "/" route
    # injects this value into index.html as `window.__BASE_PATH__`, and
    # app/ui/app.js prefixes every request with it; the reverse proxy must
    # still strip the prefix back off before forwarding, since this
    # container itself never learns to serve anything but root-relative
    # paths.
    base_path: str = ""

    @field_validator("base_path")
    @classmethod
    def _no_trailing_slash(cls, value: str) -> str:
        """A trailing slash would double up with app.js's own leading
        "/" on every path it builds (BASE_PATH + "/v1/...") -- e.g.
        "/satya-vani/" + "/v1/config" would build "/satya-vani//v1/config".
        Stripped here once rather than trusted at every call site."""
        return value.rstrip("/")


settings = Settings()
