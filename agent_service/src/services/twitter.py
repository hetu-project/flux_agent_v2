"""Twitter API service for fetching tweets using RapidAPI."""

from typing import List, Optional
from datetime import datetime, timezone
import asyncio
import httpx
from src.config import get_settings
from src.models.tweet import Tweet, TweetMeta, TweetMetrics
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class TwitterService:
    """Service for interacting with Twitter API using RapidAPI."""
    
    def __init__(self):
        self.rapid_api_key = settings.rapid_api_key
        self.headers = {
            "x-rapidapi-key": self.rapid_api_key,
            "x-rapidapi-host": "twitter241.p.rapidapi.com"
        }
        self.base_url = "https://twitter241.p.rapidapi.com"
    
    async def get_user_id_by_username(self, username: str) -> str:
        """
        Get Twitter user ID by username using RapidAPI.
        
        Args:
            username: Twitter username (without @)
            
        Returns:
            Twitter user ID (numeric string)
            
        Raises:
            ValueError: If user not found or error occurred
        """
        url = f"{self.base_url}/user"
        querystring = {"username": username}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self.headers,
                    params=querystring,
                    timeout=30.0
                )
                response.raise_for_status()
                user_data = response.json()
                
                # Check for errors
                if 'errors' in user_data:
                    raise ValueError(f"Error fetching user ID: {user_data.get('errors', [])}")
                
                # Extract user ID from response
                user_id = user_data.get('result', {}).get('data', {}).get('user', {}).get('result', {}).get('rest_id')
                
                if not user_id:
                    raise ValueError(f"User '{username}' not found or user ID not available")
                
                logger.info(f"Found user ID {user_id} for username @{username}")
                return str(user_id)
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching user ID: {e.response.status_code} - {e.response.text}")
                raise ValueError(f"Failed to fetch user ID for username '{username}': {e.response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching user ID: {e}", exc_info=True)
                raise
    
    async def get_user_tweets_by_id(
        self,
        user_id: str,
        max_results: int = 10
    ) -> List[Tweet]:
        """
        Get tweets from a specific user by Twitter user ID using RapidAPI.
        
        Args:
            user_id: Twitter user ID (numeric string)
            max_results: Maximum number of tweets to fetch
            
        Returns:
            List of Tweet objects
        """
        url = f"{self.base_url}/user-tweets"
        querystring = {"user": user_id, "count": str(max_results)}
        
        all_tweets = []
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self.headers,
                    params=querystring,
                    timeout=30.0
                )
                response.raise_for_status()
                tweets_data = response.json()
                
                # Parse response based on RapidAPI structure
                instructions = tweets_data.get('result', {}).get('timeline', {}).get('instructions', [])
                if not instructions:
                    logger.warning("No instructions found in response")
                    return all_tweets
                
                # Collect entries from all instructions
                # Instructions can be:
                # - TimelineClearCache: no entries
                # - TimelinePinEntry: pinned tweet (has 'entry' not 'entries')
                # - TimelineAddEntries: list of tweets (has 'entries')
                all_entries = []
                
                for instruction in instructions:
                    instruction_type = instruction.get('type', '')
                    
                    # Handle TimelineAddEntries (most common)
                    if 'entries' in instruction:
                        entries = instruction.get('entries', [])
                        all_entries.extend(entries)
                    
                    # Handle TimelinePinEntry (pinned tweet)
                    elif 'entry' in instruction:
                        entry = instruction.get('entry')
                        if entry:
                            all_entries.append(entry)
                
                logger.info(f"Found {len(all_entries)} entries in response (from {len(instructions)} instructions)")
                
                for item in all_entries:
                    entry_id = item.get('entryId', '')
                    created_at = None
                    tweet_text = None
                    tweet_id = None
                    author_id = None
                    author_username = None
                    metrics = {
                        'like_count': 0,
                        'retweet_count': 0,
                        'reply_count': 0,
                        'quote_count': 0
                    }
                    
                    try:
                        # Try to extract tweet data from direct tweet
                        if 'content' in item and 'itemContent' in item['content']:
                            tweet_results = item['content']['itemContent'].get('tweet_results', {}).get('result', {})
                            if tweet_results and 'legacy' in tweet_results:
                                legacy = tweet_results['legacy']
                                # Check author id matches
                                author_id = tweet_results.get('core', {}).get('user_results', {}).get('result', {}).get('rest_id', '')
                                if author_id == str(user_id):
                                    created_at = legacy.get('created_at', '')
                                    tweet_text = legacy.get('full_text', '')
                                    tweet_id = legacy.get('id_str', '')
                                    author_username = tweet_results.get('core', {}).get('user_results', {}).get('result', {}).get('legacy', {}).get('screen_name', '')
                                    
                                    # Extract metrics
                                    metrics['like_count'] = legacy.get('favorite_count', 0)
                                    metrics['retweet_count'] = legacy.get('retweet_count', 0)
                                    metrics['reply_count'] = legacy.get('reply_count', 0)
                                    metrics['quote_count'] = legacy.get('quote_count', 0)
                        
                        # Try to extract from conversation thread
                        elif 'content' in item and 'items' in item['content']:
                            for content_item in item['content']['items']:
                                if 'item' in content_item:
                                    item_content = content_item['item'].get('itemContent', {})
                                    tweet_results = item_content.get('tweet_results', {}).get('result', {})
                                    if tweet_results and 'legacy' in tweet_results:
                                        legacy = tweet_results['legacy']
                                        author_id = tweet_results.get('core', {}).get('user_results', {}).get('result', {}).get('rest_id', '')
                                        if author_id == str(user_id):
                                            created_at = legacy.get('created_at', '')
                                            tweet_text = legacy.get('full_text', '')
                                            tweet_id = legacy.get('id_str', '')
                                            author_username = tweet_results.get('core', {}).get('user_results', {}).get('result', {}).get('legacy', {}).get('screen_name', '')
                                            
                                            # Extract metrics
                                            metrics['like_count'] = legacy.get('favorite_count', 0)
                                            metrics['retweet_count'] = legacy.get('retweet_count', 0)
                                            metrics['reply_count'] = legacy.get('reply_count', 0)
                                            metrics['quote_count'] = legacy.get('quote_count', 0)
                                            break
                    
                    except Exception as e:
                        logger.debug(f"Error extracting tweet data from entry {entry_id}: {e}")
                        continue
                    
                    # Create Tweet object if we have valid data
                    if tweet_id and tweet_text and created_at:
                        try:
                            # Parse created_at: "Mon Jan 01 12:00:00 +0000 2024"
                            tweet_time = datetime.strptime(
                                created_at,
                                "%a %b %d %H:%M:%S +0000 %Y"
                            ).replace(tzinfo=timezone.utc)
                            
                            tweet = Tweet(
                                id=tweet_id,
                                text=tweet_text,
                                meta=TweetMeta(
                                    author=author_username or "unknown",
                                    created_at=tweet_time,
                                    author_id=author_id or user_id
                                ),
                                metrics=TweetMetrics(
                                    like_count=metrics['like_count'],
                                    retweet_count=metrics['retweet_count'],
                                    reply_count=metrics['reply_count'],
                                    quote_count=metrics['quote_count']
                                )
                            )
                            all_tweets.append(tweet)
                            logger.debug(f"Added tweet {tweet_id} from user {user_id}")
                        
                        except ValueError as e:
                            logger.warning(f"Invalid date format for tweet {tweet_id}: {created_at}, error: {e}")
                            continue
                
                logger.info(f"Successfully parsed {len(all_tweets)} tweets from {len(all_entries)} entries")
                return all_tweets
            
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching tweets: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error fetching tweets: {e}", exc_info=True)
                raise
