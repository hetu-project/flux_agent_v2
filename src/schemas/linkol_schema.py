"""Linkol API schemas for requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class KOLPriceResponseData(BaseModel):
    """KOL price data."""
    price: float = Field(..., description="KOL price")


class KOLPriceResponse(BaseModel):
    """Response schema for KOL price calculation."""
    code: int = Field(..., description="Status code: 200 for success, 400 for missing parameters")
    msg: str = Field(..., description="Message: 'ok' on success")
    data: KOLPriceResponseData = Field(..., description="Price data")


class KOLInfo(BaseModel):
    """KOL information."""
    id: Optional[int] = Field(None, description="KOL ID")
    screen_name: Optional[str] = Field(None, description="Screen name (@username)")
    name: Optional[str] = Field(None, description="Display name")
    x_user_id: Optional[str] = Field(None, description="X (Twitter) user ID")
    description: Optional[str] = Field(None, description="Profile description")
    profile_image_url: Optional[str] = Field(None, description="Profile image URL")
    profile_banner_url: Optional[str] = Field(None, description="Profile banner URL")
    x_created_at: Optional[str] = Field(None, description="X account creation time")
    location: Optional[str] = Field(None, description="Location")
    followers_count: Optional[int] = Field(None, description="Followers count")
    total_tweet_count: Optional[int] = Field(None, description="Total tweet count")
    like_count: Optional[int] = Field(None, description="Total like count")
    search_count: Optional[int] = Field(None, description="Search count")
    last_search: Optional[str] = Field(None, description="Last search time")
    created_at: Optional[str] = Field(None, description="Data creation time")


class HotKOLsResponseData(BaseModel):
    """Hot KOLs list data."""
    total: int = Field(..., description="Total count")
    current_page: int = Field(..., description="Current page number")
    page_range: List[int] = Field(..., description="Page range")
    list: List[KOLInfo] = Field(..., description="KOL list")


class HotKOLsResponse(BaseModel):
    """Response schema for hot KOLs."""
    code: int = Field(..., description="Status code")
    msg: str = Field(..., description="Message")
    data: HotKOLsResponseData = Field(..., description="KOL list data")

