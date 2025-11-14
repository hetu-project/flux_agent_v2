"""Task data models."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """Task model."""
    task_id: int = Field(..., description="Task ID")
    twitter_name: str = Field(..., description="Twitter username")
    type: str = Field(..., description="Task type (e.g., twitter_retweet, twitter_post)")
    url: str = Field(..., description="Task URL")
    creator_wallet: str = Field(..., description="Creator wallet address")
    flux_task_id: str = Field(..., description="Flux task ID")
    telegram_channel: Optional[str] = Field(None, description="Telegram channel")
    created_time: datetime = Field(..., description="Task creation time")
    valid_until: datetime = Field(..., description="Task validity end time")
    status: str = Field(..., description="Task status (e.g., active)")

