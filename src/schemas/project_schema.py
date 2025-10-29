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

