"""Chat schemas for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat with RAG agent."""
    query: str = Field(..., description="User's question", min_length=1)
    project: Optional[str] = Field(None, description="Project name to filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")


class Source(BaseModel):
    """Source document information."""
    text: str
    author: str
    created_at: str
    score: float


class ChatResponse(BaseModel):
    """Response schema for chat."""
    answer: str
    sources: List[Source]
    num_sources: int

