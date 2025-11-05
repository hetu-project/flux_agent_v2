"""Tweet data models."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TweetMeta(BaseModel):
    """Tweet metadata."""
    author: str = Field(..., description="Twitter author username")
    created_at: datetime = Field(..., description="Tweet creation time")
    author_id: Optional[str] = Field(None, description="Twitter author ID")
    project: Optional[str] = Field(None, description="Project name")
    

class TweetMetrics(BaseModel):
    """Tweet engagement metrics."""
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0


class Tweet(BaseModel):
    """Tweet model."""
    id: str = Field(..., description="Tweet ID")
    text: str = Field(..., description="Tweet content")
    meta: TweetMeta = Field(..., description="Tweet metadata")
    metrics: TweetMetrics = Field(..., description="Engagement metrics")
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "tweet_id": self.id,
            "text": self.text,
            "author": self.meta.author,
            "author_id": self.meta.author_id,
            "created_at": self.meta.created_at.isoformat(),
            "project": self.meta.project,
            "likes": self.metrics.like_count,
            "retweets": self.metrics.retweet_count,
            "replies": self.metrics.reply_count,
        }

