"""ProjectContent schemas for API requests and responses."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


ContentType = Literal["tweet", "paper", "document", "blog", "project_description"]


class CreateProjectContentRequest(BaseModel):
    """Request schema for creating project content."""
    content_id: str = Field(..., description="Unique content ID")
    project_name: str = Field(..., description="Associated project name")
    content_type: ContentType = Field(..., description="Type of content (tweet, paper, etc.)")
    content: str = Field(..., description="Main content text", min_length=1)
    
    # Optional metadata fields
    title: Optional[str] = Field(None, description="Title (for papers, documents, blogs)")
    author: Optional[str] = Field(None, description="Author (for tweets, papers, blogs)")
    source_url: Optional[str] = Field(None, description="Source URL")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    timestamp: Optional[int] = Field(None, description="Unix timestamp (seconds since epoch)")
    
    # Tweet-specific fields
    tweet_id: Optional[str] = Field(None, description="Tweet ID (for tweets)")
    author_id: Optional[str] = Field(None, description="Author ID (for tweets)")
    likes: Optional[int] = Field(None, ge=0, description="Like count (for tweets)")
    retweets: Optional[int] = Field(None, ge=0, description="Retweet count (for tweets)")
    replies: Optional[int] = Field(None, ge=0, description="Reply count (for tweets)")
    
    # Paper-specific fields
    paper_id: Optional[str] = Field(None, description="Paper ID (for papers)")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID (for papers)")


class CreateProjectContentResponse(BaseModel):
    """Response schema for creating project content."""
    status: str = "success"
    content_id: str
    project_name: str
    content_type: ContentType
    message: str


class ProjectContentSearchRequest(BaseModel):
    """Request schema for searching project content."""
    query: str = Field(..., description="Search query (will be vectorized)", min_length=1)
    project_name: Optional[str] = Field(None, description="Project name to filter")
    content_type: Optional[ContentType] = Field(None, description="Content type to filter")
    content_types: Optional[List[ContentType]] = Field(None, description="List of content types to filter")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class ProjectContentSearchResult(BaseModel):
    """Single project content search result."""
    content_id: str
    project_name: str
    content_type: ContentType
    content: str
    title: Optional[str] = None
    author: Optional[str] = None
    source_url: Optional[str] = None
    created_at: Optional[str] = None
    timestamp: Optional[int] = None
    score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    
    # Type-specific fields
    tweet_id: Optional[str] = None
    likes: Optional[int] = None
    retweets: Optional[int] = None
    paper_id: Optional[str] = None
    arxiv_id: Optional[str] = None


class ProjectContentSearchResponse(BaseModel):
    """Response schema for searching project content."""
    results: List[ProjectContentSearchResult]
    total: int


class ProjectContentResponse(BaseModel):
    """Response schema for getting a single project content."""
    content_id: str
    project_name: str
    content_type: ContentType
    content: str
    title: Optional[str] = None
    author: Optional[str] = None
    source_url: Optional[str] = None
    created_at: Optional[str] = None
    timestamp: Optional[int] = None
    
    # Type-specific fields
    tweet_id: Optional[str] = None
    author_id: Optional[str] = None
    likes: Optional[int] = None
    retweets: Optional[int] = None
    replies: Optional[int] = None
    paper_id: Optional[str] = None
    arxiv_id: Optional[str] = None

