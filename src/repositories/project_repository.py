"""Repository layer for Project CRUD operations."""

from typing import List, Optional, Dict, Any
import hashlib
from src.models.project import Project
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProjectRepository:
    """
    Repository for Project CRUD operations.
    
    Stores projects in Qdrant vector database with vector embeddings.
    Supports vector similarity search on project descriptions.
    """
    
    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        collection_name: str = "projects"
    ):
        self.qdrant = qdrant_service
        self.embedding = embedding_service
        self.collection_name = collection_name
    
    def _name_to_id(self, name: str) -> int:
        """
        Convert project name to integer ID using hash.
        
        Args:
            name: Project name
            
        Returns:
            Integer ID for Qdrant
        """
        # Use MD5 hash and take first 8 bytes to create a positive integer
        hash_bytes = hashlib.md5(name.encode()).digest()[:8]
        # Convert to unsigned integer (always positive)
        return int.from_bytes(hash_bytes, byteorder='big')
    
    def _ensure_collection(self):
        """Ensure collection exists with proper vector size."""
        try:
            self.qdrant.get_collection_info(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it with embedding dimension
            vector_size = self.embedding.get_dimension()
            self.qdrant.ensure_collection(self.collection_name, vector_size=vector_size)
    
    def create(self, project: Project) -> Project:
        """
        Create a new project with vector embedding.
        
        Args:
            project: Project model to create
            
        Returns:
            Created project
        """
        logger.info(f"Creating project: {project.name}")
        self._ensure_collection()
        
        # Convert project name to integer ID
        project_id = self._name_to_id(project.name)
        logger.debug(f"Project ID for '{project.name}': {project_id}")
        
        # Check if project already exists
        try:
            existing = self.qdrant.client.retrieve(
                collection_name=self.collection_name,
                ids=[project_id]
            )
            if existing:
                # Check if the existing project has the same name
                existing_payload = existing[0].payload
                if existing_payload.get("name") == project.name:
                    logger.warning(f"Project '{project.name}' already exists")
                    raise ValueError(f"Project '{project.name}' already exists")
        except ValueError:
            raise
        except Exception as e:
            logger.debug(f"Error checking existing project: {e}")
            pass
        
        # Generate embedding for project description (or name if no description)
        text_to_embed = project.description or project.name
        logger.debug(f"Generating embedding for: {text_to_embed[:50]}...")
        vector = self.embedding.embed_text(text_to_embed)
        
        # Create point with embedding vector (use integer ID)
        point = PointStruct(
            id=project_id,
            vector=vector,
            payload=project.to_payload()
        )
        
        self.qdrant.upsert_points(self.collection_name, [point])
        logger.info(f"Project '{project.name}' created successfully")
        return project
    
    def get_by_name(self, name: str) -> Optional[Project]:
        """
        Get project by name.
        
        Args:
            name: Project name
            
        Returns:
            Project if found, None otherwise
        """
        try:
            self._ensure_collection()
            # Convert name to integer ID
            project_id = self._name_to_id(name)
            
            points = self.qdrant.client.retrieve(
                collection_name=self.collection_name,
                ids=[project_id]
            )
            
            if not points or len(points) == 0:
                return None
            
            point = points[0]
            payload = point.payload
            
            # Verify the name matches (in case of hash collision)
            if payload.get("name") != name:
                return None
            
            return Project(
                name=payload.get("name", name),
                description=payload.get("description")
            )
        except Exception:
            return None
    
    def get_all(self) -> List[Project]:
        """
        Get all projects.
        
        Returns:
            List of all projects
        """
        try:
            self._ensure_collection()
            # Scroll through all points
            result = self.qdrant.client.scroll(
                collection_name=self.collection_name,
                limit=1000
            )
            
            projects = []
            for point in result[0]:  # result is (points, next_page_offset)
                payload = point.payload
                projects.append(Project(
                    name=payload.get("name", point.id),
                    description=payload.get("description")
                ))
            
            return projects
        except Exception:
            return []
    
    def update(self, name: str, description: Optional[str] = None) -> Optional[Project]:
        """
        Update project description and regenerate embedding.
        
        Args:
            name: Project name
            description: New description (optional)
            
        Returns:
            Updated project if found, None otherwise
        """
        try:
            self._ensure_collection()
            
            # Get existing project
            project = self.get_by_name(name)
            if not project:
                return None
            
            # Update description
            if description is not None:
                project.description = description
                
                # Regenerate embedding for updated description
                text_to_embed = project.description or project.name
                vector = self.embedding.embed_text(text_to_embed)
                
                # Convert name to integer ID
                project_id = self._name_to_id(name)
                
                # Update both payload and vector in Qdrant
                self.qdrant.client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(
                        id=project_id,
                        vector=vector,
                        payload=project.to_payload()
                    )]
                )
            
            return project
        except Exception:
            return None
    
    def delete(self, name: str) -> bool:
        """
        Delete a project.
        
        Args:
            name: Project name to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self._ensure_collection()
            # Convert name to integer ID
            project_id = self._name_to_id(name)
            self.qdrant.delete_points(self.collection_name, [project_id])
            return True
        except Exception:
            return False
    
    def exists(self, name: str) -> bool:
        """
        Check if project exists.
        
        Args:
            name: Project name
            
        Returns:
            True if exists
        """
        return self.get_by_name(name) is not None
    
    def count(self) -> int:
        """
        Get total number of projects.
        
        Returns:
            Total count
        """
        try:
            self._ensure_collection()
            info = self.qdrant.get_collection_info(self.collection_name)
            return info.points_count
        except Exception:
            return 0
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search projects by description using vector similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with project data and scores
        """
        try:
            self._ensure_collection()
            
            # Generate query embedding
            query_vector = self.embedding.embed_text(query)
            
            # Search in Qdrant (no user_id filter for projects - they are shared)
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                # Filter by min_score if provided
                if min_score is not None and result.score < min_score:
                    continue
                
                payload = result.payload
                formatted_results.append({
                    "name": payload.get("name", result.id),
                    "description": payload.get("description"),
                    "score": result.score
                })
            
            return formatted_results
        except Exception:
            return []
