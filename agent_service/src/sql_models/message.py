"""Message SQLAlchemy model for storing chat messages."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.services.database import Base


class Message(Base):
    """Message model for storing chat messages in conversations."""
    
    __tablename__ = "messages"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    
    # Foreign key
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True, comment="Associated conversation ID")
    
    # Message content
    role = Column(String(50), nullable=False, index=True, comment="Message role: user, assistant, system")
    content = Column(Text, nullable=False, comment="Message content")
    sequence = Column(Integer, nullable=False, default=0, index=True, comment="Message sequence number within conversation")
    
    # Optional metadata (JSON field for flexibility)
    extra_metadata = Column(JSON, nullable=True, comment="Additional metadata (e.g., token_count, model_name, etc.)")
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True, comment="Creation timestamp")
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_message_conversation_sequence", "conversation_id", "sequence"),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role='{self.role}', sequence={self.sequence})>"

