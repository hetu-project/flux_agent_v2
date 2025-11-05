"""Project API routes."""

from fastapi import APIRouter, HTTPException, Depends
from src.repositories.project_repository import ProjectRepository
from src.schemas.project_schema import (
    ProjectCreateRequest,
    ProjectUpdateRequest,
    ProjectResponse,
    ProjectListResponse,
    ProjectSearchRequest,
    ProjectSearchResponse,
    ProjectSearchResult,
)
from src.models.project import Project
from src.api.dependencies import get_project_repo


router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    request: ProjectCreateRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
):
    """
    Create a new project.
    """
    try:
        project = Project(
            name=request.name,
            description=request.description
        )
        
        created_project = project_repo.create(project)
        return ProjectResponse.from_model(created_project)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=ProjectListResponse)
async def list_projects(project_repo: ProjectRepository = Depends(get_project_repo)):
    """
    Get all projects.
    """
    try:
        projects = project_repo.get_all()
        project_responses = [ProjectResponse.from_model(p) for p in projects]
        
        return ProjectListResponse(
            projects=project_responses,
            total=len(project_responses)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_name}", response_model=ProjectResponse)
async def get_project(
    project_name: str,
    project_repo: ProjectRepository = Depends(get_project_repo),
):
    """
    Get project by name.
    """
    try:
        project = project_repo.get_by_name(project_name)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
        
        return ProjectResponse.from_model(project)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{project_name}", response_model=ProjectResponse)
async def update_project(
    project_name: str,
    request: ProjectUpdateRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
):
    """
    Update project description.
    """
    try:
        updated_project = project_repo.update(
            name=project_name,
            description=request.description
        )
        
        if not updated_project:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
        
        return ProjectResponse.from_model(updated_project)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_name}", status_code=204)
async def delete_project(
    project_name: str,
    project_repo: ProjectRepository = Depends(get_project_repo),
):
    """
    Delete a project.
    """
    try:
        success = project_repo.delete(project_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=ProjectSearchResponse)
async def search_projects(
    request: ProjectSearchRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
):
    """
    Search projects by description using vector similarity.
    """
    try:
        results = project_repo.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # Format results
        search_results = [
            ProjectSearchResult(
                name=r["name"],
                description=r.get("description"),
                score=r["score"]
            )
            for r in results
        ]
        
        return ProjectSearchResponse(
            results=search_results,
            total=len(search_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

