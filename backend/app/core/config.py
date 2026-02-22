from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Pinecone ──────────────────────────────────────────────────────────────
    PINECODE_API_KEY:    str = ""
    PINECONE_INDEX_NAME: str = "cumi-abrasives-index"
    PINECONE_EMBED_MODEL:  str = "llama-text-embed-v2-1024"
    PINECONE_RERANK_MODEL: str = "bge-reranker-v2-m3"
    PINECONE_EMBED_DIM:    int = 1024
    SEARCH_TOP_N:          int = 30    # candidates fetched before reranking
    SEARCH_TOP_K:          int = 5     # final results returned to user

    # ── Ollama / LLM ─────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL:    str = "llama3.2"

    # ── App ───────────────────────────────────────────────────────────────────
    APP_TITLE:   str = "CUMI Abrasives AI Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG:       bool = False


settings = Settings()