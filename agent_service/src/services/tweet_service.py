"""Service layer for Tweet business logic."""

from typing import List
from src.models.tweet import Tweet
from src.services.twitter import TwitterService
from src.services.embedding import EmbeddingService
from src.repositories.tweet_repository import TweetRepository
from src.repositories.collection_repository import CollectionRepository


class TweetService:
    """Service for Tweet business operations."""
    
    def __init__(
        self,
        tweet_repo: TweetRepository,
        collection_repo: CollectionRepository,
        twitter_service: TwitterService,
        embedding_service: EmbeddingService,
    ):
        self.tweet_repo = tweet_repo
        self.collection_repo = collection_repo
        self.twitter_service = twitter_service
        self.embedding_service = embedding_service
    
    async def collect_and_store_tweets(
        self,
        project_name: str,
        username: str = None,
        query: str = None,
        max_tweets: int = 100
    ) -> int:
        """
        Collect tweets from Twitter and store in vector database.
        
        Args:
            project_name: Project name
            username: Twitter username (optional)
            query: Search query (optional)
            max_tweets: Maximum number of tweets
            
        Returns:
            Number of tweets collected
        """
        # Ensure collection exists
        collection_name = "twitter_tweets"
        vector_size = self.embedding_service.get_dimension()
        self.collection_repo.create(collection_name, vector_size)
        
        # Fetch tweets from Twitter
        if query:
            tweets = await self.twitter_service.search_tweets(
                query=query,
                max_results=max_tweets
            )
        elif username:
            tweets = await self.twitter_service.get_user_tweets(
                username=username,
                max_results=max_tweets
            )
        else:
            raise ValueError("Either 'query' or 'username' must be provided")
        
        # Add project metadata
        for tweet in tweets:
            tweet.meta.project = project_name
        
        # Generate embeddings
        texts = [tweet.text for tweet in tweets]
        embeddings = self.embedding_service.embed_batch(texts)
        
        # Store in vector database
        count = self.tweet_repo.create(tweets, embeddings)
        
        return count

