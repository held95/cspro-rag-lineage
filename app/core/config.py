from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KeyRecord(BaseModel):
    """What an API key resolves to. The tenant ALWAYS comes from here, never from a request."""

    tenant_id: str
    subject: str
    groups: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Configuration from environment variables prefixed ``CSPRO_`` (or a local ``.env.local``)."""

    model_config = SettingsConfigDict(env_prefix="CSPRO_", env_file=".env.local", extra="ignore")

    ledger_path: str = "var/ledger.db"
    vector_path: str | None = "var/qdrant"  # None -> in-memory (tests)
    collection: str = "chunks"

    llm_backend: str = "ollama"  # "ollama" | "fake"
    embed_backend: str = "ollama"  # "ollama" | "hash"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    embed_model: str = "nomic-embed-text"
    llm_connect_timeout: float = 5.0
    llm_read_timeout: float = 120.0

    min_score: float = 0.45
    max_body_bytes: int = 256 * 1024

    # sha256(api_key) -> KeyRecord. Populated by `python -m scripts.seed_demo` into .env.local.
    api_keys: dict[str, KeyRecord] = Field(default_factory=dict)
