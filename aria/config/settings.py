from __future__ import annotations
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq ────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # ── Ollama ───────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")

    # ── Agent LLM routing ────────────────────────────────────────
    orchestrator_provider: Literal["groq", "ollama"] = Field(
        default="groq", alias="ORCHESTRATOR_PROVIDER"
    )
    executor_provider: Literal["groq", "ollama"] = Field(
        default="ollama", alias="EXECUTOR_PROVIDER"
    )

    # ── Embeddings ───────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )

    # ── Paths ────────────────────────────────────────────────────
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    # ── Agent behaviour ──────────────────────────────────────────
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    executor_max_turns: int = Field(default=10, alias="EXECUTOR_MAX_TURNS")
    anomaly_drift_threshold: float = Field(
        default=0.45, alias="ANOMALY_DRIFT_THRESHOLD"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
