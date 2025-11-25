"""User SQLAlchemy model for storing end-user information."""

from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.services.database import Base


class User(Base):
    """User model for storing client user details and profile metadata."""

    __tablename__ = "users"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)

    # Identifiers
    client_id = Column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="External client identifier",
    )

    # Profile data (JSON for flexibility)
    profile = Column(JSON, nullable=True, comment="User profile metadata")

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Creation timestamp",
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp",
    )

    # Relationships
    conversations = relationship("Conversation", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, client_id='{self.client_id}')>"

