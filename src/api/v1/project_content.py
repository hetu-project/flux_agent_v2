"""API routes for ProjectContent operations."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from src.repositories.project_content_repository import ProjectContentRepository
from src.models.project_content import ProjectContent
from src.schemas.project_content_schema import (
    CreateProjectContentRequest,
    CreateProjectContentResponse,
    ProjectContentSearchRequest,
    ProjectContentSearchResponse,
    ProjectContentSearchResult,
    ProjectContentResponse,
)
from src.api.dependencies import get_project_content_repo
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/project-content", tags=["project-content"])


@router.post("", response_model=CreateProjectContentResponse, status_code=201)
async def create_project_content(
    request: CreateProjectContentRequest,
    repo: ProjectContentRepository = Depends(get_project_content_repo),
):
    """
    Create a new project content item (tweet, paper, document, etc.).
    
    The content will be automatically vectorized and stored in the project_content collection.
    """
    logger.info(f"Creating content: {request.content_type} for project '{request.project_name}'")
    
    try:
        # Convert request to model
        content = ProjectContent(
            content_id=request.content_id,
            project_name=request.project_name,
            content_type=request.content_type,
            content=request.content,
            title=request.title,
            author=request.author,
            source_url=request.source_url,
            created_at=request.created_at,
            timestamp=request.timestamp,
            tweet_id=request.tweet_id,
            author_id=request.author_id,
            likes=request.likes,
            retweets=request.retweets,
            replies=request.replies,
            paper_id=request.paper_id,
            arxiv_id=request.arxiv_id,
        )
        
        # Create in repository (embedding will be generated automatically)
        created_content = repo.create(content)
        
        logger.info(f"Content '{created_content.content_id}' created successfully")
        
        return CreateProjectContentResponse(
            status="success",
            content_id=created_content.content_id,
            project_name=created_content.project_name,
            content_type=created_content.content_type,
            message=f"Content '{created_content.content_id}' created successfully"
        )
    except ValueError as e:
        logger.error(f"Validation error creating content: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create content: {str(e)}")


@router.get("/{content_id}", response_model=ProjectContentResponse)
async def get_project_content(
    content_id: str,
    repo: ProjectContentRepository = Depends(get_project_content_repo),
):
    """Get a project content item by ID."""
    logger.debug(f"Getting content by ID: {content_id}")
    
    content = repo.get_by_id(content_id)
    if not content:
        raise HTTPException(status_code=404, detail=f"Content '{content_id}' not found")
    
    return ProjectContentResponse(
        content_id=content.content_id,
        project_name=content.project_name,
        content_type=content.content_type,
        content=content.content,
        title=content.title,
        author=content.author,
        source_url=content.source_url,
        created_at=content.created_at.isoformat() if content.created_at else None,
        timestamp=content.timestamp,
        tweet_id=content.tweet_id,
        author_id=content.author_id,
        likes=content.likes,
        retweets=content.retweets,
        replies=content.replies,
        paper_id=content.paper_id,
        arxiv_id=content.arxiv_id,
    )


@router.post("/search", response_model=ProjectContentSearchResponse)
async def search_project_content(
    request: ProjectContentSearchRequest,
    repo: ProjectContentRepository = Depends(get_project_content_repo),
):
    """
    Search project content by vector similarity.
    
    Supports filtering by project_name and content_type.
    Can search across all content types or filter by specific types.
    """
    logger.info(f"Searching content: query='{request.query[:50]}...', project={request.project_name}, type={request.content_type}")
    
    try:
        # Search in repository
        results = repo.search(
            query=request.query,
            project_name=request.project_name,
            content_type=request.content_type,
            content_types=request.content_types,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(ProjectContentSearchResult(
                content_id=result.get("content_id", ""),
                project_name=result.get("project_name", ""),
                content_type=result.get("content_type", "document"),
                content=result.get("content", ""),
                title=result.get("title"),
                author=result.get("author"),
                source_url=result.get("source_url"),
                created_at=result.get("created_at"),
                timestamp=result.get("timestamp"),
                score=result.get("score", 0.0),
                tweet_id=result.get("tweet_id"),
                likes=result.get("likes"),
                retweets=result.get("retweets"),
                paper_id=result.get("paper_id"),
                arxiv_id=result.get("arxiv_id"),
            ))
        
        logger.info(f"Search returned {len(formatted_results)} results")
        
        return ProjectContentSearchResponse(
            results=formatted_results,
            total=len(formatted_results)
        )
    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search content: {str(e)}")


@router.delete("/{content_id}", status_code=204)
async def delete_project_content(
    content_id: str,
    repo: ProjectContentRepository = Depends(get_project_content_repo),
):
    """Delete a project content item by ID."""
    logger.info(f"Deleting content: {content_id}")
    
    success = repo.delete(content_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Content '{content_id}' not found")
    
    return None

