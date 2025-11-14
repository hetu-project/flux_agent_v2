"""Tweet schemas for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class CollectTweetsRequest(BaseModel):
    """Request schema for collecting tweets."""
    project_name: str = Field(..., description="Project name")
    user_id: str = Field(..., description="Twitter user ID (numeric string)")
    max_tweets: int = Field(10, ge=1, le=100, description="Maximum number of tweets to collect")


class CollectTweetsResponse(BaseModel):
    """Response schema for collecting tweets."""
    status: str = "success"
    project: str
    tweets_collected: int
    message: str


class TweetSearchRequest(BaseModel):
    """Request schema for searching tweets."""
    query: str = Field(..., description="Search query (will be vectorized)")
    project: Optional[str] = Field(None, description="Project name to filter")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class TweetSearchResult(BaseModel):
    """Single tweet search result."""
    id: str
    text: str
    author: str
    created_at: str
    project: Optional[str] = None
    score: float = Field(..., description="Similarity score")


class TweetSearchResponse(BaseModel):
    """Response schema for searching tweets."""
    results: List[TweetSearchResult]
    total: int


class GetTweetsByProjectRequest(BaseModel):
    """Request schema for getting tweets by project."""
    project_name: str = Field(..., description="Project name")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of tweets to return")
    offset: Optional[int] = Field(None, ge=0, description="Offset for pagination")


class TweetItem(BaseModel):
    """Single tweet item."""
    id: str
    tweet_id: str
    text: str
    author: str
    author_id: Optional[str] = None
    created_at: str
    project: Optional[str] = None
    likes: int = 0
    retweets: int = 0
    replies: int = 0


class GetTweetsByProjectResponse(BaseModel):
    """Response schema for getting tweets by project."""
    project_name: str
    tweets: List[TweetItem]
    total: int

