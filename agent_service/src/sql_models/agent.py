"""Agent SQLAlchemy model for storing agent information."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.services.database import Base


class Agent(Base):
    """Agent model for storing agent information."""
    
    __tablename__ = "agents"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    
    # Agent information
    name = Column(String(255), nullable=False, unique=True, index=True, comment="Agent name")
    description = Column(Text, nullable=True, comment="Agent description")
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, comment="Whether the agent is active")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Creation timestamp")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="Last update timestamp")
    
    # Relationships
    conversations = relationship("Conversation", back_populates="agent", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Agent(id={self.id}, name='{self.name}')>"

