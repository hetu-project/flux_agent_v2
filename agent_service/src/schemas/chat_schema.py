"""Chat schemas for API requests and responses."""

from typing import List, Optional, Dict, Any
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


class GetChatHistoryRequest(BaseModel):
    """Request schema for getting chat history."""
    user_id: str = Field(..., description="User ID (client_id) for filtering chat history")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of messages to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    agent_name: Optional[str] = Field(None, description="Optional agent name to filter by (e.g., 'company')")


class ChatHistoryMessage(BaseModel):
    """Chat history message model."""
    id: int = Field(..., description="Message ID")
    conversation_id: int = Field(..., description="Conversation ID")
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    sequence: int = Field(..., description="Message sequence number")
    created_at: str = Field(..., description="Message creation timestamp")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatHistoryConversation(BaseModel):
    """Chat history conversation model."""
    id: int = Field(..., description="Conversation ID")
    agent_id: int = Field(..., description="Agent ID")
    agent_name: Optional[str] = Field(None, description="Agent name")
    title: Optional[str] = Field(None, description="Conversation title")
    status: str = Field(..., description="Conversation status")
    created_at: str = Field(..., description="Conversation creation timestamp")
    updated_at: str = Field(..., description="Conversation last update timestamp")
    messages: List[ChatHistoryMessage] = Field(default_factory=list, description="Messages in this conversation")


class GetChatHistoryResponse(BaseModel):
    """Response schema for getting chat history."""
    conversations: List[ChatHistoryConversation] = Field(..., description="List of conversations with messages")
    total: int = Field(..., description="Total number of conversations")
    limit: int = Field(..., description="Limit used in request")
    offset: int = Field(..., description="Offset used in request")

