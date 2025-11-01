"""Script to add Parallel Universe project to database."""

import asyncio
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


async def add_parallel_universe():
    """Add Parallel Universe project to database."""
    logger.info("Adding Parallel Universe project...")
    
    # Initialize services
    logger.info("Initializing services...")
    qdrant_service = QdrantService()
    embedding_service = EmbeddingService()
    project_repo = ProjectRepository(
        qdrant_service=qdrant_service,
        embedding_service=embedding_service,
        collection_name="projects"
    )
    
    project_name = "Parallel Universe"
    project_description = (
        "Parallel Universe is a parallel network ecosystem launched by HETU, built with the FLUX points system. "
        "Users can complete tasks in the parallel network to earn FLUX points, which can be exchanged for various tokens or NFTs within the ecosystem. "
        "It aims to deeply integrate the mainnet and parallel network ecosystems to create scale effects."
    )
    
    # Check if project already exists
    try:
        existing = project_repo.get_by_name(project_name)
        if existing:
            logger.warning(f"Project '{project_name}' already exists!")
            logger.info(f"Existing description: {existing.description}")
            logger.info("Skipping creation. Use update if you want to modify it.")
            return
    except Exception:
        # Project doesn't exist, continue
        pass
    
    # Create project
    try:
        project = Project(
            name=project_name,
            description=project_description
        )
        
        project_repo.create(project)
        logger.info(f"✅ Successfully created project: {project_name}")
        logger.info(f"Description: {project_description[:100]}...")
        
    except ValueError as e:
        logger.error(f"❌ Error: Project '{project_name}' already exists: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating project '{project_name}': {e}")
        raise


def main():
    """Main entry point."""
    asyncio.run(add_parallel_universe())


if __name__ == "__main__":
    main()

