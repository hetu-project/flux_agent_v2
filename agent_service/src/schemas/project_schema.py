"""Project schemas for API requests and responses."""

from typing import Optional
from pydantic import BaseModel, Field

from src.models.project import Project


class ProjectCreateRequest(BaseModel):
    """Request schema for creating a project."""
    name: str = Field(..., description="Project name", min_length=1)
    description: Optional[str] = Field(None, description="Project description")


class ProjectUpdateRequest(BaseModel):
    """Request schema for updating a project."""
    description: Optional[str] = Field(None, description="Project description")


class ProjectResponse(BaseModel):
    """Response schema for project."""
    name: str
    description: Optional[str] = None

    @classmethod
    def from_model(cls, project: Project) -> "ProjectResponse":
        """Create from Project model."""
        return cls(
            name=project.name,
            description=project.description
        )


class ProjectListResponse(BaseModel):
    """Response schema for project list."""
    projects: list[ProjectResponse]
    total: int


class ProjectSearchRequest(BaseModel):
    """Request schema for searching projects."""
    query: str = Field(..., description="Search query (will be vectorized for similarity search)")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class ProjectSearchResult(BaseModel):
    """Single project search result."""
    name: str
    description: Optional[str] = None
    score: float = Field(..., description="Similarity score")


class ProjectSearchResponse(BaseModel):
    """Response schema for project search."""
    results: list[ProjectSearchResult]
    total: int

