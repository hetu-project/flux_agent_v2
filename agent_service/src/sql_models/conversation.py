"""Conversation SQLAlchemy model for storing conversation sessions."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.services.database import Base


class Conversation(Base):
    """Conversation model for storing conversation sessions."""
    
    __tablename__ = "conversations"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    
    # Foreign keys
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True, comment="Associated agent ID")
    user_id = Column(String(255), nullable=True, index=True, comment="User ID for multi-user support")
    
    # Conversation metadata
    title = Column(String(500), nullable=True, comment="Conversation title/summary (optional)")
    status = Column(String(50), default="active", nullable=False, index=True, comment="Conversation status: active, expired, deleted")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True, comment="Creation timestamp")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="Last update timestamp")
    expired_at = Column(DateTime(timezone=True), nullable=True, index=True, comment="Expiration timestamp (optional)")
    
    # Relationships
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.sequence")
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_conversation_user_agent", "user_id", "agent_id"),
        Index("idx_conversation_user_status", "user_id", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id='{self.user_id}', agent_id={self.agent_id}, status='{self.status}')>"

