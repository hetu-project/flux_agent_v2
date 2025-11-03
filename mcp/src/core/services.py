"""Service initialization and management."""

from typing import Optional, Tuple
from config import get_settings
from services.linkol import LinkolService
from services.qdrant_client import QdrantService
from services.embedding import EmbeddingService
from repositories.project_content_repository import ProjectContentRepository
from utils.logger import get_logger

logger = get_logger(__name__)

# Global service instances (singleton pattern)
_qdrant_service: Optional[QdrantService] = None
_embedding_service: Optional[EmbeddingService] = None
_linkol_service: Optional[LinkolService] = None
_project_content_repo: Optional[ProjectContentRepository] = None


def init_services():
    """Initialize all services (called once at startup)."""
    global _qdrant_service, _embedding_service, _linkol_service, _project_content_repo
    
    if _qdrant_service is None:
        logger.info("Initializing services...")
        _qdrant_service = QdrantService()
        _embedding_service = EmbeddingService()
        _linkol_service = LinkolService()
        _project_content_repo = ProjectContentRepository(
            qdrant_service=_qdrant_service,
            embedding_service=_embedding_service,
            collection_name="project_content"
        )
        logger.info("Services initialized successfully")
    
    return _linkol_service, _project_content_repo, _qdrant_service


def get_linkol_service() -> LinkolService:
    """Get LinkolService instance."""
    if _linkol_service is None:
        init_services()
    return _linkol_service


def get_project_content_repo() -> ProjectContentRepository:
    """Get ProjectContentRepository instance."""
    if _project_content_repo is None:
        init_services()
    return _project_content_repo


def get_qdrant_service() -> QdrantService:
    """Get QdrantService instance."""
    if _qdrant_service is None:
        init_services()
    return _qdrant_service


def get_embedding_service() -> EmbeddingService:
    """Get EmbeddingService instance."""
    if _embedding_service is None:
        init_services()
    return _embedding_service


def get_all_services() -> Tuple[LinkolService, ProjectContentRepository, QdrantService]:
    """Get all service instances."""
    return init_services()
