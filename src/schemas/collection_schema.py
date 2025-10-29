"""Collection schemas for API requests and responses."""

from typing import Optional
from pydantic import BaseModel, Field


class CollectionCreateRequest(BaseModel):
    """Request schema for creating a collection."""
    name: str = Field(..., description="Collection name")
    vector_size: int = Field(1536, ge=1, description="Vector dimension size")


class CollectionInfo(BaseModel):
    """Collection information schema."""
    name: str
    points_count: int
    vectors_count: int
    indexed_vectors_count: Optional[int] = None

