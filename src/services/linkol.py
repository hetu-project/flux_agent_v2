"""Linkol API service for KOL price and hot KOLs."""

from typing import Optional, List, Dict, Any
import httpx
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class LinkolService:
    """Service for interacting with Linkol API."""
    
    def __init__(self):
        self.base_url = settings.linkol_url.rstrip('/')
        self.api_key = settings.linkol_api_key
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def get_kol_price(
        self,
        screen_name: str,
        page: Optional[int] = None,
        size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get KOL price calculation.
        
        This endpoint calculates the price based on the KOL's last 20 original tweets.
        
        Args:
            screen_name: KOL screen name (without @, e.g., "vis_eth")
            page: Page number (optional)
            size: Maximum items per page (optional)
            
        Returns:
            Response dict with format:
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "price": 1115.07
                }
            }
        """
        url = f"{self.base_url}/open/api/v1/kol/price/"
        
        params = {
            "screen_name": screen_name
        }
        if page is not None:
            params["page"] = page
        if size is not None:
            params["size"] = size
        
        logger.debug(f"Requesting KOL price: {url}, params={params}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers=self.headers,
                    params=params if params else None
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"KOL price retrieved successfully: {result.get('data', {}).get('price')}")
                return result
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting KOL price: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error getting KOL price: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting KOL price: {e}")
            raise
    
    async def get_hot_kols(self) -> Dict[str, Any]:
        """
        Get hot/popular KOLs list.
        
        Returns:
            Response dict with format:
            {
                "code": 200,
                "msg": "ok",
                "data": {
                    "total": 1,
                    "current_page": 1,
                    "page_range": [1],
                    "list": [
                        {
                            "id": 1,
                            "screen_name": "vis_eth",
                            "name": "Vis.eth",
                            "x_user_id": "4449061",
                            "description": "...",
                            "profile_image_url": "...",
                            "profile_banner_url": "...",
                            "x_created_at": "2007-04-13 03:54:24",
                            "location": "",
                            "followers_count": 8784,
                            "total_tweet_count": 813,
                            "like_count": 1622,
                            "search_count": 2,
                            "last_search": "2025-08-13 07:39:31",
                            "created_at": "2025-08-13 07:36:07"
                        }
                    ]
                }
            }
        """
        url = f"{self.base_url}/open/api/v1/hot/kols/"
        
        logger.debug(f"Requesting hot KOLs: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers=self.headers
                )
                response.raise_for_status()
                result = response.json()
                
                total = result.get('data', {}).get('total', 0)
                logger.info(f"Hot KOLs retrieved successfully: {total} KOLs")
                return result
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting hot KOLs: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error getting hot KOLs: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting hot KOLs: {e}")
            raise

