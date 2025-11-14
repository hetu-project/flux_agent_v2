"""Test script to debug why @499_DAO tweets are not being fetched."""

import asyncio
import sys
import os
import json

# Add the app directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(script_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from src.services.twitter import TwitterService
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Test fetching tweets for @499_DAO."""
    print("=" * 80)
    print("Testing @499_DAO Tweet Fetching")
    print("=" * 80)
    
    twitter_service = TwitterService()
    username = "499_DAO"
    user_id = "1229555715129233408"
    
    # Step 1: Verify user ID
    print(f"\nStep 1: Verifying user ID for @{username}...")
    print(f"  Expected user_id: {user_id}")
    
    try:
        fetched_user_id = await twitter_service.get_user_id_by_username(username)
        print(f"  ✅ Fetched user_id: {fetched_user_id}")
        
        if fetched_user_id != user_id:
            print(f"  ⚠️  WARNING: User ID mismatch!")
            print(f"     Expected: {user_id}")
            print(f"     Got: {fetched_user_id}")
            user_id = fetched_user_id
    except Exception as e:
        print(f"  ❌ Error fetching user ID: {e}")
        return
    
    # Step 2: Fetch tweets
    print(f"\nStep 2: Fetching tweets for user_id: {user_id}...")
    
    try:
        tweets = await twitter_service.get_user_tweets_by_id(
            user_id=user_id,
            max_results=10
        )
        
        print(f"\n✅ Successfully fetched {len(tweets)} tweets")
        
        if tweets:
            print(f"\n{'=' * 80}")
            print("Tweet Details:")
            print("=" * 80)
            for i, tweet in enumerate(tweets, 1):
                print(f"\nTweet #{i}:")
                print(f"  ID: {tweet.id}")
                print(f"  Author: @{tweet.meta.author}")
                print(f"  Created: {tweet.meta.created_at}")
                print(f"  Text: {tweet.text[:100]}..." if len(tweet.text) > 100 else f"  Text: {tweet.text}")
                print(f"  Metrics: {tweet.metrics.like_count} likes, {tweet.metrics.retweet_count} retweets")
        else:
            print("\n⚠️  No tweets found. This could mean:")
            print("  - The account has no tweets")
            print("  - The account is private")
            print("  - The API response format is different")
            print("  - There's an issue with the parsing logic")
            
            # Let's check the raw API response
            print(f"\n{'=' * 80}")
            print("Debugging: Checking raw API response...")
            print("=" * 80)
            
            import httpx
            url = f"{twitter_service.base_url}/user-tweets"
            querystring = {"user": user_id, "count": "10"}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=twitter_service.headers,
                    params=querystring,
                    timeout=30.0
                )
                response.raise_for_status()
                tweets_data = response.json()
                
                print(f"\nResponse structure:")
                print(f"  Top-level keys: {list(tweets_data.keys())}")
                
                if 'result' in tweets_data:
                    result = tweets_data['result']
                    print(f"  Result keys: {list(result.keys())}")
                    
                    if 'timeline' in result:
                        timeline = result['timeline']
                        print(f"  Timeline keys: {list(timeline.keys())}")
                        
                        if 'instructions' in timeline:
                            instructions = timeline['instructions']
                            print(f"  Number of instructions: {len(instructions)}")
                            
                            for i, instruction in enumerate(instructions):
                                print(f"\n  Instruction {i}:")
                                print(f"    Type: {instruction.get('type', 'unknown')}")
                                if 'entries' in instruction:
                                    entries = instruction['entries']
                                    print(f"    Number of entries: {len(entries)}")
                                    
                                    # Show first entry structure
                                    if entries:
                                        print(f"    First entry keys: {list(entries[0].keys())}")
                                        print(f"    First entry ID: {entries[0].get('entryId', 'N/A')}")
                
                # Save full response for inspection
                print(f"\n{'=' * 80}")
                print("Full response (first 2000 chars):")
                print("=" * 80)
                response_str = json.dumps(tweets_data, indent=2, ensure_ascii=False, default=str)
                print(response_str[:2000])
                if len(response_str) > 2000:
                    print(f"\n... (truncated, total length: {len(response_str)} chars)")
        
    except Exception as e:
        print(f"\n❌ Error fetching tweets: {e}")
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

