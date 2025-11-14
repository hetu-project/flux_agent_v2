"""Task schemas for API requests and responses."""

from typing import List
from pydantic import BaseModel, Field


class TaskListRequest(BaseModel):
    """Request schema for listing tasks."""
    limit: int = Field(10, ge=1, le=100, description="Maximum number of tasks to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class TaskListResponse(BaseModel):
    """Response schema for listing tasks."""
    tasks: List[dict] = Field(..., description="List of tasks")
    total_count: int = Field(..., description="Total number of tasks")
    twitter_retweet_count: int = Field(..., description="Number of Twitter retweet tasks")
    twitter_post_count: int = Field(..., description="Number of Twitter post tasks")
    telegram_task_count: int = Field(..., description="Number of Telegram tasks")
    limit: int = Field(..., description="Limit used in request")
    offset: int = Field(..., description="Offset used in request")
    has_more: bool = Field(..., description="Whether there are more tasks")

