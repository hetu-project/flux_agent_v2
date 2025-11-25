"""Chat schemas for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user, assistant, system, etc.)")
    content: str = Field(..., description="Message content", min_length=1)


class ChatRequest(BaseModel):
    """Request schema for chat with RAG agent (OpenAI-compatible format)."""
    model: Optional[str] = Field(None, description="Model name (ignored, kept for compatibility)")
    messages: List[Message] = Field(..., description="List of chat messages", min_length=1)
    project: Optional[str] = Field(None, description="Project name to filter (optional)")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    session_id: Optional[str] = Field(None, description="Session ID for maintaining conversation context (optional)")
    user_id: Optional[str] = Field(None, description="User ID for user-specific context management (optional)")
    
    def get_user_query(self) -> str:
        """
        Extract user query from messages.
        Returns the content of the last message.
        """
        if not self.messages:
            raise ValueError("Messages list is empty")
        return self.messages[-1].content


class Source(BaseModel):
    """Source document information."""
    text: str
    author: str
    created_at: str
    score: float


class ChatMessage(BaseModel):
    """Chat message in response."""
    role: str = Field(default="assistant", description="Message role (hardcoded as assistant)")
    content: str = Field(..., description="Message content")


class Choice(BaseModel):
    """Choice in response."""
    message: ChatMessage = Field(..., description="Chat message")


class ChatResponse(BaseModel):
    """Response schema for chat (OpenAI-compatible format)."""
    choices: List[Choice] = Field(..., description="List of choices")
    session_id: Optional[str] = Field(None, description="Session ID for maintaining conversation context (returned for subsequent requests)")

