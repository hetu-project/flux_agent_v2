"""Test script for fetching tweets using TwitterService with RapidAPI."""

import asyncio
import json
from src.services.twitter import TwitterService
from src.config import get_settings

settings = get_settings()


async def test_fetch_tweets():
    """Test fetching tweets by user ID."""
    # Check if RapidAPI key is loaded
    print(f"RapidAPI Key loaded: {'Yes' if settings.rapid_api_key else 'No'}")
    if settings.rapid_api_key:
        print(f"Key preview: {settings.rapid_api_key[:20]}...")
    else:
        print("WARNING: RapidAPI Key is empty!")
        print("Please check your .env file and ensure RAPID_API_KEY is set.")
        return
    
    # Initialize TwitterService
    twitter_service = TwitterService()
    
    # Test: Get user ID by username
    username = "litterpigger"
    print(f"\nLooking up user ID for username: @{username}")
    try:
        user_id = await twitter_service.get_user_id_by_username(username)
        print(f"Found user ID: {user_id}")
    except Exception as e:
        print(f"Error getting user ID: {e}")
        return
    
    max_tweets = 10
    
    print(f"\nFetching {max_tweets} tweets for user ID: {user_id}")
    print("-" * 80)
    
    try:
        # Fetch tweets
        tweets = await twitter_service.get_user_tweets_by_id(
            user_id=user_id,
            max_results=max_tweets
        )
        
        print(f"\nSuccessfully fetched {len(tweets)} tweets\n")
        print("=" * 80)
        
        # Print each tweet
        for i, tweet in enumerate(tweets, 1):
            print(f"\nTweet #{i}:")
            print(f"  ID: {tweet.id}")
            print(f"  Author: @{tweet.meta.author} (ID: {tweet.meta.author_id})")
            print(f"  Created At: {tweet.meta.created_at}")
            print(f"  Text: {tweet.text}")
            print(f"  Metrics:")
            print(f"    - Likes: {tweet.metrics.like_count}")
            print(f"    - Retweets: {tweet.metrics.retweet_count}")
            print(f"    - Replies: {tweet.metrics.reply_count}")
            print(f"    - Quotes: {tweet.metrics.quote_count}")
            print("-" * 80)
        
        # Also print as JSON for easier inspection
        print("\n\nJSON Format:")
        print("=" * 80)
        tweets_json = [
            {
                "id": tweet.id,
                "text": tweet.text,
                "author": tweet.meta.author,
                "author_id": tweet.meta.author_id,
                "created_at": tweet.meta.created_at.isoformat(),
                "metrics": {
                    "like_count": tweet.metrics.like_count,
                    "retweet_count": tweet.metrics.retweet_count,
                    "reply_count": tweet.metrics.reply_count,
                    "quote_count": tweet.metrics.quote_count,
                }
            }
            for tweet in tweets
        ]
        print(json.dumps(tweets_json, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\nError fetching tweets: {e}")
        print(f"\nError type: {type(e).__name__}")
        
        # Check if it's an authentication error
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n⚠️  401 Unauthorized Error - Possible causes:")
            print("1. RapidAPI Key is invalid or expired")
            print("2. RapidAPI Key format is incorrect")
            print("3. Key doesn't have access to Twitter API")
            print("\nNote: RapidAPI Key should be obtained from:")
            print("   https://rapidapi.com/hub")
            print("   Make sure you subscribe to the Twitter API service")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_fetch_tweets())

