"""Project content data models for unified storage of tweets, papers, etc."""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


ContentType = Literal["tweet", "paper", "document", "blog", "project_description"]


class ProjectContent(BaseModel):
    """Unified model for project-related content (tweets, papers, etc.)."""
    
    content_id: str = Field(..., description="Unique content ID")
    project_name: str = Field(..., description="Associated project name")
    content_type: ContentType = Field(..., description="Type of content (tweet, paper, etc.)")
    content: str = Field(..., description="Main content text")
    
    # Optional metadata fields (flexible based on content type)
    title: Optional[str] = Field(None, description="Title (for papers, documents, blogs)")
    author: Optional[str] = Field(None, description="Author (for tweets, papers, blogs)")
    source_url: Optional[str] = Field(None, description="Source URL")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    timestamp: Optional[int] = Field(None, description="Unix timestamp (seconds since epoch)")
    
    # Tweet-specific fields
    tweet_id: Optional[str] = Field(None, description="Tweet ID (for tweets)")
    author_id: Optional[str] = Field(None, description="Author ID (for tweets)")
    likes: Optional[int] = Field(None, description="Like count (for tweets)")
    retweets: Optional[int] = Field(None, description="Retweet count (for tweets)")
    replies: Optional[int] = Field(None, description="Reply count (for tweets)")
    
    # Paper-specific fields
    paper_id: Optional[str] = Field(None, description="Paper ID (for papers)")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID (for papers)")
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional flexible metadata")
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        payload = {
            "content_id": self.content_id,
            "project_name": self.project_name,
            "content_type": self.content_type,
            "content": self.content,
        }
        
        # Add optional fields if present
        if self.title:
            payload["title"] = self.title
        if self.author:
            payload["author"] = self.author
        if self.source_url:
            payload["source_url"] = self.source_url
        if self.created_at:
            payload["created_at"] = self.created_at.isoformat()
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        
        # Add type-specific fields
        if self.content_type == "tweet":
            if self.tweet_id:
                payload["tweet_id"] = self.tweet_id
            if self.author_id:
                payload["author_id"] = self.author_id
            if self.likes is not None:
                payload["likes"] = self.likes
            if self.retweets is not None:
                payload["retweets"] = self.retweets
            if self.replies is not None:
                payload["replies"] = self.replies
        
        elif self.content_type == "paper":
            if self.paper_id:
                payload["paper_id"] = self.paper_id
            if self.arxiv_id:
                payload["arxiv_id"] = self.arxiv_id
        
        # Add flexible metadata
        if self.metadata:
            payload.update(self.metadata)
        
        return payload
    
    @classmethod
    def from_payload(cls, content_id: str, payload: Dict[str, Any]) -> "ProjectContent":
        """Create ProjectContent from Qdrant payload."""
        return cls(
            content_id=content_id,
            project_name=payload.get("project_name", ""),
            content_type=payload.get("content_type", "document"),  # Default to document
            content=payload.get("content", ""),
            title=payload.get("title"),
            author=payload.get("author"),
            source_url=payload.get("source_url"),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else None,
            timestamp=payload.get("timestamp"),
            tweet_id=payload.get("tweet_id"),
            author_id=payload.get("author_id"),
            likes=payload.get("likes"),
            retweets=payload.get("retweets"),
            replies=payload.get("replies"),
            paper_id=payload.get("paper_id"),
            arxiv_id=payload.get("arxiv_id"),
            metadata={k: v for k, v in payload.items() 
                     if k not in ["content_id", "project_name", "content_type", "content", 
                                  "title", "author", "source_url", "created_at", "timestamp",
                                  "tweet_id", "author_id", "likes", "retweets", "replies",
                                  "paper_id", "arxiv_id"]}
        )

