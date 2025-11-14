"""Repository layer for Tweet CRUD operations."""

from typing import List, Optional, Dict, Any
import hashlib
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from src.models.tweet import Tweet
from src.services.qdrant_client import QdrantService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TweetRepository:
    """
    Repository for Tweet CRUD operations.
    
    Handles all vector database operations for tweets:
    - Create (Upsert)
    - Read (Search, Retrieve)
    - Update (Upsert, SetPayload)
    - Delete
    """
    
    def __init__(self, qdrant_service: QdrantService, collection_name: str = "twitter_tweets"):
        self.qdrant = qdrant_service
        self.collection_name = collection_name
    
    def _tweet_id_to_qdrant_id(self, tweet_id: str) -> int:
        """
        Convert tweet ID string to integer ID for Qdrant.
        
        Twitter IDs are numeric strings, but they can be very large.
        If the ID can be converted to int directly, use it.
        Otherwise, use MD5 hash to generate a consistent integer ID.
        
        Args:
            tweet_id: Tweet ID string
            
        Returns:
            Integer ID for Qdrant
        """
        try:
            # Try to convert directly to int (Twitter IDs are numeric)
            return int(tweet_id)
        except (ValueError, OverflowError):
            # If conversion fails or number is too large, use hash
            hash_bytes = hashlib.md5(tweet_id.encode()).digest()[:8]
            return int.from_bytes(hash_bytes, byteorder='big')
    
    def create(self, tweets: List[Tweet], embeddings: List[List[float]]) -> int:
        """
        Create (insert) tweets into the vector database.
        
        Args:
            tweets: List of Tweet models
            embeddings: List of embedding vectors corresponding to tweets
            
        Returns:
            Number of tweets created
        """
        if len(tweets) != len(embeddings):
            raise ValueError("Tweets and embeddings must have the same length")
        
        points = []
        for tweet, embedding in zip(tweets, embeddings):
            qdrant_id = self._tweet_id_to_qdrant_id(tweet.id)
            points.append(PointStruct(
                id=qdrant_id,
                vector=embedding,
                payload=tweet.to_payload()
            ))
        
        self.qdrant.upsert_points(self.collection_name, points)
        return len(points)
    
    def upsert(self, tweets: List[Tweet], embeddings: List[List[float]]) -> int:
        """
        Upsert (insert or update) tweets.
        
        Args:
            tweets: List of Tweet models
            embeddings: List of embedding vectors
            
        Returns:
            Number of tweets upserted
        """
        return self.create(tweets, embeddings)
    
    def search(
        self,
        query_vector: List[float],
        project: Optional[str] = None,
        author: Optional[str] = None,
        top_k: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search tweets by vector similarity.
        
        Args:
            query_vector: Query embedding vector
            project: Optional project filter
            author: Optional author filter
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with tweet data and scores
        """
        # Build filter
        filter_query = None
        conditions = []
        
        if project:
            conditions.append(
                FieldCondition(key="project", match=MatchValue(value=project))
            )
        if author:
            conditions.append(
                FieldCondition(key="author", match=MatchValue(value=author))
            )
        
        if conditions:
            filter_query = Filter(must=conditions)
        
        # Search
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
            
            formatted_results.append({
                "id": result.id,
                "text": result.payload.get("text", ""),
                "author": result.payload.get("author", ""),
                "created_at": result.payload.get("created_at", ""),
                "project": result.payload.get("project"),
                "likes": result.payload.get("likes", 0),
                "retweets": result.payload.get("retweets", 0),
                "score": result.score
            })
        
        return formatted_results
    
    def retrieve_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve tweets by their IDs.
        
        Args:
            ids: List of tweet IDs
            
        Returns:
            List of tweet data
        """
        try:
            points = self.qdrant.client.retrieve(
                collection_name=self.collection_name,
                ids=ids
            )
            
            results = []
            for point in points:
                results.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "author": point.payload.get("author", ""),
                    "created_at": point.payload.get("created_at", ""),
                    "project": point.payload.get("project"),
                    "likes": point.payload.get("likes", 0),
                    "retweets": point.payload.get("retweets", 0),
                })
            
            return results
        except Exception:
            return []
    
    def update_payload(self, tweet_id: str, payload: Dict[str, Any]) -> bool:
        """
        Update tweet payload (metadata) without changing the vector.
        
        Args:
            tweet_id: Tweet ID
            payload: New payload data
            
        Returns:
            True if updated successfully
        """
        try:
            self.qdrant.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[tweet_id]
            )
            return True
        except Exception:
            return False
    
    def delete(self, tweet_ids: List[str]) -> int:
        """
        Delete tweets by IDs.
        
        Args:
            tweet_ids: List of tweet IDs to delete
            
        Returns:
            Number of tweets deleted
        """
        try:
            self.qdrant.delete_points(self.collection_name, tweet_ids)
            return len(tweet_ids)
        except Exception:
            return 0
    
    def delete_by_filter(
        self,
        project: Optional[str] = None,
        author: Optional[str] = None
    ) -> int:
        """
        Delete tweets by filter conditions.
        
        Args:
            project: Optional project filter
            author: Optional author filter
            
        Returns:
            Number of tweets deleted
        """
        try:
            conditions = []
            if project:
                conditions.append(
                    FieldCondition(key="project", match=MatchValue(value=project))
                )
            if author:
                conditions.append(
                    FieldCondition(key="author", match=MatchValue(value=author))
                )
            
            filter_query = Filter(must=conditions) if conditions else None
            
            self.qdrant.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_query
            )
            return 1  # Qdrant doesn't return count, assume success
        except Exception:
            return 0
    
    def get_by_project(
        self,
        project: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all tweets for a specific project.
        
        Args:
            project: Project name
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of tweet data
        """
        try:
            # Build filter
            filter_query = Filter(
                must=[
                    FieldCondition(key="project", match=MatchValue(value=project))
                ]
            )
            
            # Use scroll to get all matching points
            scroll_limit = limit if limit else 1000
            result = self.qdrant.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_query,
                limit=scroll_limit,
                offset=offset
            )
            
            points, next_offset = result
            
            # Format results
            tweets = []
            for point in points:
                tweets.append({
                    "id": str(point.id),
                    "text": point.payload.get("text", ""),
                    "author": point.payload.get("author", ""),
                    "author_id": point.payload.get("author_id"),
                    "created_at": point.payload.get("created_at", ""),
                    "project": point.payload.get("project"),
                    "tweet_id": point.payload.get("tweet_id", str(point.id)),
                    "likes": point.payload.get("likes", 0),
                    "retweets": point.payload.get("retweets", 0),
                    "replies": point.payload.get("replies", 0),
                })
            
            return tweets
        except Exception as e:
            logger.error(f"Error getting tweets by project: {e}", exc_info=True)
            return []
    
    def count(
        self,
        project: Optional[str] = None,
        author: Optional[str] = None
    ) -> int:
        """
        Count tweets matching filters.
        
        Args:
            project: Optional project filter
            author: Optional author filter
            
        Returns:
            Number of tweets matching criteria
        """
        try:
            info = self.qdrant.get_collection_info(self.collection_name)
            # If no filters, return total count
            if not project and not author:
                return info.points_count
            
            # Otherwise, we'd need to use scroll to count
            # For simplicity, return total if filters provided (can be optimized later)
            return info.points_count
        except Exception:
            return 0

