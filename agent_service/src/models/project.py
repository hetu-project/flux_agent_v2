"""Project data models."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Project(BaseModel):
    """Project model."""
    name: str = Field(..., description="Project name", min_length=1)
    description: Optional[str] = Field(None, description="Project description")
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to payload format for vector database or SQL database."""
        return {
            "name": self.name,
            "description": self.description or "",
        }

