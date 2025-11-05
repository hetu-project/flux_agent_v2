"""Twitter API service for fetching tweets."""

from typing import List, Optional
from datetime import datetime
import httpx
from src.config import get_settings
from src.models.tweet import Tweet, TweetMeta, TweetMetrics

settings = get_settings()


class TwitterService:
    """Service for interacting with Twitter API."""
    
    def __init__(self):
        self.bearer_token = settings.twitter_bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
    
    async def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> List[Tweet]:
        """
        Search tweets using Twitter API v2.
        
        Args:
            query: Search query
            max_results: Maximum number of results (max 100)
            start_time: Start time for search
            user_id: Filter by specific user
            
        Returns:
            List of Tweet objects
        """
        url = f"{self.base_url}/tweets/search/recent"
        
        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,author_id,public_metrics,lang",
            "user.fields": "username",
            "expansions": "author_id"
        }
        
        if start_time:
            params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if user_id:
            params["query"] = f"{query} from:{user_id}"
        
        all_tweets = []
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse tweets
            users_map = {user["id"]: user for user in data.get("includes", {}).get("users", [])}
            
            for tweet_data in data.get("data", []):
                author_id = tweet_data["author_id"]
                author = users_map.get(author_id, {})
                
                tweet = Tweet(
                    id=tweet_data["id"],
                    text=tweet_data["text"],
                    meta=TweetMeta(
                        author=author.get("username", "unknown"),
                        created_at=datetime.fromisoformat(tweet_data["created_at"].replace("Z", "+00:00")),
                        author_id=author_id
                    ),
                    metrics=TweetMetrics(
                        like_count=tweet_data.get("public_metrics", {}).get("like_count", 0),
                        retweet_count=tweet_data.get("public_metrics", {}).get("retweet_count", 0),
                        reply_count=tweet_data.get("public_metrics", {}).get("reply_count", 0),
                        quote_count=tweet_data.get("public_metrics", {}).get("quote_count", 0)
                    )
                )
                all_tweets.append(tweet)
        
        return all_tweets
    
    async def get_user_tweets(
        self,
        username: str,
        max_results: int = 100
    ) -> List[Tweet]:
        """
        Get tweets from a specific user.
        
        Args:
            username: Twitter username (without @)
            max_results: Maximum number of results
            
        Returns:
            List of Tweet objects
        """
        return await self.search_tweets(
            query="lang:en",  # Only English tweets
            max_results=max_results,
            user_id=username
        )

