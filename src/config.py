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
    
    # AIHubMix Settings (从环境变量读取)
    aihubmix_api_key: str = ""
    aihubmix_base_url: str = "https://aihubmix.com/v1"
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "DeepSeek-V3.2-Exp"
    
    # Twitter API (从环境变量读取)
    twitter_bearer_token: str = ""
    
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

