"""Repository layer for Collection management operations."""

from typing import Optional
from src.services.qdrant_client import QdrantService


class CollectionRepository:
    """
    Repository for Collection management operations.
    
    Handles collection lifecycle: create, read, delete.
    """
    
    def __init__(self, qdrant_service: QdrantService):
        self.qdrant = qdrant_service
    
    def create(
        self,
        collection_name: str,
        vector_size: int = 1536
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            
        Returns:
            True if created successfully
        """
        try:
            self.qdrant.ensure_collection(collection_name, vector_size)
            return True
        except Exception:
            return False
    
    def exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists
        """
        try:
            self.qdrant.get_collection_info(collection_name)
            return True
        except Exception:
            return False
    
    def get_info(self, collection_name: str) -> Optional[dict]:
        """
        Get collection information.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection info dict or None if not found
        """
        try:
            info = self.qdrant.get_collection_info(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        except Exception:
            return None
    
    def delete(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.qdrant.client.delete_collection(collection_name)
            return True
        except Exception:
            return False

