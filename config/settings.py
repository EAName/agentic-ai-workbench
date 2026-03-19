"""
Settings management using pydantic-settings.
All configuration flows through here -- no scattered env reads.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    default_provider: str = Field(default="anthropic", alias="DEFAULT_PROVIDER")
    default_model: str = Field(
        default="claude-sonnet-4-20250514", alias="DEFAULT_MODEL"
    )
    default_max_tokens: int = Field(default=4096, alias="DEFAULT_MAX_TOKENS")
    default_temperature: float = Field(default=0.7, alias="DEFAULT_TEMPERATURE")

    model_config = {"env_file": ".env", "extra": "ignore"}


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    chroma_persist_dir: str = Field(
        default="./data/embeddings", alias="CHROMA_PERSIST_DIR"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    model_config = {"env_file": ".env", "extra": "ignore"}


class LogSettings(BaseSettings):
    """Logging configuration."""

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    model_config = {"env_file": ".env", "extra": "ignore"}


class PlatformSettings(BaseSettings):
    """API platform configuration."""

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = {"env_file": ".env", "extra": "ignore"}


class Settings:
    """Aggregated settings -- single access point for all config."""

    def __init__(self):
        self.llm = LLMSettings()
        self.vector_store = VectorStoreSettings()
        self.logging = LogSettings()
        self.platform = PlatformSettings()


# Singleton
settings = Settings()
