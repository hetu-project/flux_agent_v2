"""Configuration settings for the application."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # LLM Provider Settings
    llm_provider: str = "openai"  # "openai" or "aihubmix"
    
    # OpenAI / AIHubMix
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-4-turbo-preview"
    
    # AIHubMix specific (optional)
    aihubmix_api_key: str = ""
    aihubmix_base_url: str = "https://aihubmix.com"
    
    # Twitter API
    twitter_bearer_token: str = ""
    
    # Application
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

