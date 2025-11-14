"""Task API service for fetching tasks from external API."""

from typing import Set
import httpx
from src.config import get_settings
from src.schemas.task_schema import TaskListRequest, TaskListResponse
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class TaskService:
    """Service for interacting with Task API."""
    
    def __init__(self):
        self.base_url = settings.task_api_url.rstrip('/')
        self.api_endpoint = "/api/v2/task/list"
        self.full_url = f"{self.base_url}{self.api_endpoint}"
    
    async def get_task_list(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> TaskListResponse:
        """
        Get list of tasks from the external API.
        
        Args:
            limit: Maximum number of tasks to return (default: 10)
            offset: Offset for pagination (default: 0)
            
        Returns:
            TaskListResponse containing tasks and metadata
            
        Raises:
            httpx.HTTPStatusError: If API request fails
            ValueError: If response parsing fails
        """
        request_data = {
            "limit": limit,
            "offset": offset
        }
        
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Fetching tasks from {self.full_url} with limit={limit}, offset={offset}")
                response = await client.post(
                    self.full_url,
                    json=request_data,
                    timeout=30.0
                )
                response.raise_for_status()
                response_data = response.json()
                
                logger.info(f"Successfully fetched {len(response_data.get('tasks', []))} tasks")
                return TaskListResponse(**response_data)
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching tasks: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error fetching tasks: {e}", exc_info=True)
                raise
    
    async def get_all_twitter_names(self, limit: int = 100) -> Set[str]:
        """
        Get all unique Twitter names from all tasks by paginating through all pages.
        
        Args:
            limit: Number of tasks to fetch per page (default: 100)
            
        Returns:
            Set of all non-empty twitter_name values from all tasks
            
        Raises:
            httpx.HTTPStatusError: If API request fails
            ValueError: If response parsing fails
        """
        twitter_names: Set[str] = set()
        offset = 0
        
        logger.info(f"Starting to fetch all Twitter names with limit={limit}")
        
        while True:
            try:
                # Fetch a page of tasks
                response = await self.get_task_list(limit=limit, offset=offset)
                
                # Extract twitter_name from each task
                for task in response.tasks:
                    twitter_name = task.get('twitter_name', '')
                    # Only add non-empty twitter_name
                    if twitter_name and twitter_name.strip():
                        twitter_names.add(twitter_name.strip())
                
                logger.info(
                    f"Fetched page: offset={offset}, tasks={len(response.tasks)}, "
                    f"unique twitter_names so far: {len(twitter_names)}"
                )
                
                # Check if there are more pages
                if not response.has_more:
                    logger.info(f"Reached end of pagination. Total unique twitter_names: {len(twitter_names)}")
                    break
                
                # Move to next page
                offset += limit
                
            except Exception as e:
                logger.error(f"Error fetching Twitter names at offset {offset}: {e}", exc_info=True)
                raise
        
        return twitter_names

