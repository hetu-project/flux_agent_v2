"""Configuration settings for the application."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # PostgreSQL Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hetu_agent"
    postgres_user: str = "hetu_user"
    postgres_password: str = "hetu_password"
    
    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL database URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def postgres_url_async(self) -> str:
        """Get PostgreSQL async database URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # AIHubMix Settings (从环境变量读取)
    aihubmix_api_key: str = ""
    aihubmix_base_url: str = "https://aihubmix.com/v1"
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "grok-4-fast-non-reasoning"
    
    # OpenRouter Settings
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-chat-v3.1"
    
    # RapidAPI for Twitter (从环境变量读取)
    rapid_api_key: str = ""
    
    # Linkol API (从环境变量读取)
    linkol_url: str = "https://api.linkol.ai"
    linkol_api_key: str = ""
    
    # MCP Server (从环境变量读取)
    mcp_url: str = "http://localhost:6001"
    mcp_endpoint: str = "/mcp"
    
    # Task API (从环境变量读取)
    task_api_url: str = "http://144.91.78.212:8000"
    
    # Application
    log_level: str = "INFO"
    log_file: Optional[str] = Field(None, description="Optional log file path (logs/app.log)")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="allow",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

