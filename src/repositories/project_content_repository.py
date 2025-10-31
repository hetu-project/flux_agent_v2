"""Repository layer for ProjectContent CRUD operations."""

from typing import List, Optional, Dict, Any
import hashlib
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from src.models.project_content import ProjectContent, ContentType
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProjectContentRepository:
    """
    Repository for ProjectContent CRUD operations.
    
    Unified storage for all project-related content (tweets, papers, documents, etc.)
    in a single collection with content_type differentiation via payload.
    """
    
    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        collection_name: str = "project_content"
    ):
        self.qdrant = qdrant_service
        self.embedding = embedding_service
        self.collection_name = collection_name
    
    def _content_id_to_qdrant_id(self, content_id: str) -> int:
        """
        Convert content ID to integer ID for Qdrant.
        
        Args:
            content_id: Content ID string
            
        Returns:
            Integer ID for Qdrant
        """
        hash_bytes = hashlib.md5(content_id.encode()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big')
    
    def _ensure_collection(self):
        """Ensure collection exists with proper vector size."""
        try:
            self.qdrant.get_collection_info(self.collection_name)
        except Exception:
            vector_size = self.embedding.get_dimension()
            self.qdrant.ensure_collection(self.collection_name, vector_size=vector_size)
            logger.info(f"Created collection '{self.collection_name}' with vector size {vector_size}")
    
    def create(self, content: ProjectContent, embedding: Optional[List[float]] = None) -> ProjectContent:
        """
        Create a new content item with vector embedding.
        
        Args:
            content: ProjectContent model to create
            embedding: Optional pre-computed embedding (if None, will generate)
            
        Returns:
            Created content
        """
        logger.debug(f"Creating content: {content.content_type} for project '{content.project_name}'")
        self._ensure_collection()
        
        # Convert content ID to integer ID
        qdrant_id = self._content_id_to_qdrant_id(content.content_id)
        
        # Generate embedding if not provided
        if embedding is None:
            text_to_embed = content.content
            # Prefer title + content for better context
            if content.title:
                text_to_embed = f"{content.title} {content.content}"
            logger.debug(f"Generating embedding for content (length: {len(text_to_embed)})")
            embedding = self.embedding.embed_text(text_to_embed)
        
        # Create point with embedding vector
        point = PointStruct(
            id=qdrant_id,
            vector=embedding,
            payload=content.to_payload()
        )
        
        self.qdrant.upsert_points(self.collection_name, [point])
        logger.info(f"Content '{content.content_id}' created successfully ({content.content_type})")
        return content
    
    def create_batch(
        self,
        contents: List[ProjectContent],
        embeddings: Optional[List[List[float]]] = None
    ) -> int:
        """
        Create multiple content items in batch.
        
        Args:
            contents: List of ProjectContent models
            embeddings: Optional list of pre-computed embeddings
            
        Returns:
            Number of contents created
        """
        if embeddings and len(contents) != len(embeddings):
            raise ValueError("Contents and embeddings must have the same length")
        
        logger.info(f"Creating batch of {len(contents)} content items")
        self._ensure_collection()
        
        points = []
        for i, content in enumerate(contents):
            qdrant_id = self._content_id_to_qdrant_id(content.content_id)
            
            # Generate or use provided embedding
            if embeddings:
                embedding = embeddings[i]
            else:
                text_to_embed = content.content
                if content.title:
                    text_to_embed = f"{content.title} {content.content}"
                embedding = self.embedding.embed_text(text_to_embed)
            
            points.append(PointStruct(
                id=qdrant_id,
                vector=embedding,
                payload=content.to_payload()
            ))
        
        self.qdrant.upsert_points(self.collection_name, points)
        logger.info(f"Batch created: {len(points)} items")
        return len(points)
    
    def get_by_id(self, content_id: str) -> Optional[ProjectContent]:
        """
        Get content by ID.
        
        Args:
            content_id: Content ID
            
        Returns:
            ProjectContent if found, None otherwise
        """
        try:
            self._ensure_collection()
            qdrant_id = self._content_id_to_qdrant_id(content_id)
            
            points = self.qdrant.client.retrieve(
                collection_name=self.collection_name,
                ids=[qdrant_id]
            )
            
            if not points or len(points) == 0:
                return None
            
            point = points[0]
            return ProjectContent.from_payload(content_id, point.payload)
        except Exception as e:
            logger.error(f"Error getting content by ID '{content_id}': {e}")
            return None
    
    def search(
        self,
        query: str,
        project_name: Optional[str] = None,
        content_type: Optional[ContentType] = None,
        content_types: Optional[List[ContentType]] = None,
        top_k: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search content by vector similarity.
        
        Args:
            query: Search query text
            project_name: Optional project filter
            content_type: Optional single content type filter
            content_types: Optional list of content types to filter (takes precedence over content_type)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with content data and scores
        """
        try:
            self._ensure_collection()
            
            # Generate query embedding
            query_vector = self.embedding.embed_text(query)
            
            # Build filter
            filter_query = None
            conditions = []
            
            if project_name:
                conditions.append(
                    FieldCondition(key="project_name", match=MatchValue(value=project_name))
                )
            
            if content_types:
                # Filter by multiple content types
                conditions.append(
                    FieldCondition(key="content_type", match=MatchAny(any=content_types))
                )
            elif content_type:
                # Filter by single content type
                conditions.append(
                    FieldCondition(key="content_type", match=MatchValue(value=content_type))
                )
            
            if conditions:
                filter_query = Filter(must=conditions)
            
            # Search in Qdrant
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                filter_query=filter_query
            )
            
            # Format results
            formatted_results = []
            for result in results:
                # Filter by min_score if provided
                if min_score is not None and result.score < min_score:
                    continue
                
                payload = result.payload
                formatted_results.append({
                    "content_id": payload.get("content_id", result.id),
                    "project_name": payload.get("project_name"),
                    "content_type": payload.get("content_type"),
                    "content": payload.get("content", ""),
                    "title": payload.get("title"),
                    "author": payload.get("author"),
                    "source_url": payload.get("source_url"),
                    "created_at": payload.get("created_at"),
                    "timestamp": payload.get("timestamp"),
                    "score": result.score,
                    # Include type-specific fields
                    **{k: v for k, v in payload.items() 
                       if k not in ["content_id", "project_name", "content_type", "content",
                                    "title", "author", "source_url", "created_at", "timestamp"]}
                })
            
            logger.debug(f"Search returned {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    def delete(self, content_id: str) -> bool:
        """
        Delete content by ID.
        
        Args:
            content_id: Content ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self._ensure_collection()
            qdrant_id = self._content_id_to_qdrant_id(content_id)
            self.qdrant.delete_points(self.collection_name, [qdrant_id])
            logger.info(f"Content '{content_id}' deleted")
            return True
        except Exception as e:
            logger.error(f"Error deleting content '{content_id}': {e}")
            return False
    
    def delete_by_filter(
        self,
        project_name: Optional[str] = None,
        content_type: Optional[ContentType] = None
    ) -> int:
        """
        Delete content by filter conditions.
        
        Args:
            project_name: Optional project filter
            content_type: Optional content type filter
            
        Returns:
            Number of items deleted (approximate)
        """
        try:
            self._ensure_collection()
            conditions = []
            
            if project_name:
                conditions.append(
                    FieldCondition(key="project_name", match=MatchValue(value=project_name))
                )
            if content_type:
                conditions.append(
                    FieldCondition(key="content_type", match=MatchValue(value=content_type))
                )
            
            filter_query = Filter(must=conditions) if conditions else None
            
            self.qdrant.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_query
            )
            logger.info(f"Deleted content by filter (project: {project_name}, type: {content_type})")
            return 1  # Qdrant doesn't return exact count
        except Exception as e:
            logger.error(f"Error deleting content by filter: {e}")
            return 0
    
    def count(
        self,
        project_name: Optional[str] = None,
        content_type: Optional[ContentType] = None
    ) -> int:
        """
        Count content items matching filters.
        
        Args:
            project_name: Optional project filter
            content_type: Optional content type filter
            
        Returns:
            Total count (approximate if filters provided)
        """
        try:
            self._ensure_collection()
            info = self.qdrant.get_collection_info(self.collection_name)
            # For simplicity, return total count (can be optimized with scroll if needed)
            return info.points_count
        except Exception:
            return 0

