"""Application configuration loaded from environment."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = Field(default="anthropic", alias="LLM_PROVIDER")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")
    anthropic_fast_model: str = Field(default="claude-haiku-4-5-20251001", alias="ANTHROPIC_FAST_MODEL")

    xai_api_key: str = Field(default="", alias="XAI_API_KEY")
    xai_model: str = Field(default="grok-3", alias="XAI_MODEL")
    xai_fast_model: str = Field(default="grok-3-mini", alias="XAI_FAST_MODEL")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_fast_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_FAST_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    gemini_fast_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_FAST_MODEL")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai/", alias="GEMINI_BASE_URL")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_fast_model: str = Field(default="gpt-4o-mini", alias="OPENAI_FAST_MODEL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    shared_daily_quota: int = Field(default=5, alias="SHARED_DAILY_QUOTA")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    max_upload_mb: int = Field(default=10, alias="MAX_UPLOAD_MB")
    request_timeout_s: int = Field(default=180, alias="REQUEST_TIMEOUT_S")
    cors_origins_raw: str = Field(default="*", alias="CORS_ORIGINS")
    allowed_hosts_raw: str = Field(default="*", alias="ALLOWED_HOSTS")
    domain: str = Field(default="", alias="DOMAIN")

    @property
    def cors_origins(self) -> List[str]:
        if self.cors_origins_raw.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def allowed_hosts(self) -> List[str]:
        if self.allowed_hosts_raw.strip() == "*":
            return ["*"]
        return [h.strip() for h in self.allowed_hosts_raw.split(",") if h.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
