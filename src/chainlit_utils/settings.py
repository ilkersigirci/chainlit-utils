"""Configuration for Chainlit utilities."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration shared by the reusable Chainlit helpers."""

    model_config = SettingsConfigDict(
        env_prefix="CHAINLIT_UTILS_",
        extra="ignore",
    )

    MIGRATIONS_TABLE: str = Field(
        default="_chainlit_utils_schema_migrations",
        description="PostgreSQL table that records applied Chainlit migrations.",
    )
    MODEL_CONTEXT_EXCLUDED_KEY: str = Field(
        default="chainlit_utils.exclude_from_model_context",
        description=(
            "Persisted message metadata key that excludes UI-only messages from "
            "model context."
        ),
    )


settings = Settings()
