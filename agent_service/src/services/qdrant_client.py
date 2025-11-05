"""Qdrant client service."""

from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config import get_settings

settings = get_settings()


class QdrantService:
    """Qdrant vector database service."""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
    
    def ensure_collection(self, collection_name: str, vector_size: int = 1536):
        """Ensure collection exists, create if not."""
        try:
            self.client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection '{collection_name}'")
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct]
    ):
        """Insert or update points in collection."""
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_query: Optional[Filter] = None
    ) -> List:
        """Search for similar vectors."""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_query
        )
        return results
    
    def delete_points(
        self,
        collection_name: str,
        ids: List[str]
    ):
        """Delete points by IDs."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )
    
    def get_collection_info(self, collection_name: str):
        """Get collection information."""
        return self.client.get_collection(collection_name)

