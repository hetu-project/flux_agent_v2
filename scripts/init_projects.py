"""Initialize projects database from external API."""

import asyncio
import httpx
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repositories.project_repository import ProjectRepository
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.models.project import Project
from src.utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# API endpoint
API_URL = "http://144.91.78.212:8000/api/v2/project/list"


async def fetch_projects(limit: int = 100, offset: int = 0) -> dict:
    """
    Fetch projects from external API.
    
    Args:
        limit: Number of projects to fetch per request
        offset: Offset for pagination
        
    Returns:
        API response as dictionary
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                API_URL,
                json={"limit": limit, "offset": offset}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching projects: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching projects: {e}")
            raise


async def init_projects(limit: int = 100, skip_existing: bool = True):
    """
    Initialize projects database from external API.
    
    Args:
        limit: Number of projects to fetch per request
        skip_existing: If True, skip projects that already exist
    """
    logger.info("Starting project database initialization...")
    
    # Initialize services
    logger.info("Initializing services...")
    qdrant_service = QdrantService()
    embedding_service = EmbeddingService()
    project_repo = ProjectRepository(
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
        collection_name="projects"
    )
    
    offset = 0
    total_created = 0
    total_skipped = 0
    total_errors = 0
    
    try:
        while True:
            logger.info(f"Fetching projects (offset={offset}, limit={limit})...")
            
            # Fetch projects from API
            response_data = await fetch_projects(limit=limit, offset=offset)
            
            if not response_data.get("success"):
                logger.error(f"API returned unsuccessful response: {response_data.get('message')}")
                break
            
            projects_data = response_data.get("projects", [])
            total_count = response_data.get("total_count", 0)
            
            if not projects_data:
                logger.info("No more projects to fetch")
                break
            
            logger.info(f"Received {len(projects_data)} projects (total: {total_count})")
            
            # Process each project
            for project_data in projects_data:
                project_name = project_data.get("name")
                project_description = project_data.get("description")
                
                if not project_name:
                    logger.warning(f"Skipping project with no name: {project_data}")
                    total_errors += 1
                    continue
                
                # Check if project already exists
                if skip_existing:
                    existing = project_repo.get_by_name(project_name)
                    if existing:
                        logger.debug(f"Project '{project_name}' already exists, skipping")
                        total_skipped += 1
                        continue
                
                # Create project
                try:
                    project = Project(
                        name=project_name,
                        description=project_description or ""
                    )
                    
                    project_repo.create(project)
                    total_created += 1
                    logger.info(f"Created project: {project_name}")
                    
                except ValueError as e:
                    # Project already exists
                    logger.warning(f"Project '{project_name}' already exists: {e}")
                    total_skipped += 1
                except Exception as e:
                    logger.error(f"Error creating project '{project_name}': {e}")
                    total_errors += 1
            
            # Check if we've fetched all projects
            offset += len(projects_data)
            if offset >= total_count:
                logger.info("All projects fetched")
                break
            
            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.5)
        
        logger.info("=" * 50)
        logger.info("Project database initialization completed!")
        logger.info(f"Total created: {total_created}")
        logger.info(f"Total skipped: {total_skipped}")
        logger.info(f"Total errors: {total_errors}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}", exc_info=True)
        raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize projects database from external API")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of projects to fetch per request (default: 100)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip projects that already exist (default: True)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Do not skip existing projects (will update them)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(init_projects(
        limit=args.limit,
        skip_existing=args.skip_existing
    ))


if __name__ == "__main__":
    main()

