"""Script to fetch tweets by Twitter ID and store them in database with project mapping."""

import asyncio
import sys
import os
import json

# Add the app directory to the path so we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(script_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from src.services.twitter import TwitterService
from src.services.embedding import EmbeddingService
from src.services.qdrant_client import QdrantService
from src.repositories.tweet_repository import TweetRepository
from src.repositories.collection_repository import CollectionRepository
from src.services.tweet_service import TweetService
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Twitter name to Twitter ID mapping
TWITTER_NAME_TO_ID = {
    "499_DAO": "1229555715129233408",
    "AkashaDao": "1872550795608219648",
    "Aura_Sci": "1713678161177690113",
    "Ch40sChain": "1890220319010705408",
    "OpenBuildxyz": "1643153237120516097",
    "PredX_AI": "1586070579097894914",
    "dephynetwork": "1725368719726284800",
    "dlifexyz": "1755251730353467392",
    "hetu_protocol": "1754745579618418688",
    "linkolfun": "1186668206561185793",
    "lokachain": "1950422731788558337",
    "outland_art": "1433317836584226816",
}

# Twitter name to Project name mapping
TWITTER_NAME_TO_PROJECT = {
    "499_DAO": "499DAO",
    "AkashaDao": "Akasha Dao",
    "Aura_Sci": "AuraSci",
    "Ch40sChain": "ChaosChain",
    "OpenBuildxyz": "OpenBuild",
    "PredX_AI": "PredX",
    "dephynetwork": "DePHY",
    "dlifexyz": "dLife",
    "hetu_protocol": "Hetu Protocol",
    "linkolfun": "Linkol",
    "lokachain": "Loka",
    "outland_art": "Outland",
}


async def main():
    """Fetch tweets for all Twitter IDs and store them in database."""
    print("=" * 80)
    print("Fetching and Storing Tweets for All Projects")
    print("=" * 80)
    
    settings = get_settings()
    
    # Initialize services
    print("\nInitializing services...")
    qdrant_service = QdrantService()
    embedding_service = EmbeddingService()
    twitter_service = TwitterService()
    collection_repo = CollectionRepository(qdrant_service)
    tweet_repo = TweetRepository(qdrant_service)
    tweet_service = TweetService(
        tweet_repo=tweet_repo,
        collection_repo=collection_repo,
        twitter_service=twitter_service,
        embedding_service=embedding_service,
    )
    
    print("✅ Services initialized")
    
    # Ensure collection exists
    collection_name = "twitter_tweets"
    vector_size = embedding_service.get_dimension()
    collection_repo.create(collection_name, vector_size)
    print(f"✅ Collection '{collection_name}' ready (vector size: {vector_size})")
    
    # Process each Twitter account
    results = {}
    total_tweets = 0
    
    print(f"\n{'=' * 80}")
    print(f"Processing {len(TWITTER_NAME_TO_ID)} Twitter accounts...")
    print("=" * 80)
    
    for twitter_name, twitter_id in sorted(TWITTER_NAME_TO_ID.items()):
        project_name = TWITTER_NAME_TO_PROJECT.get(twitter_name, twitter_name)
        
        print(f"\n[{len(results) + 1}/{len(TWITTER_NAME_TO_ID)}] Processing @{twitter_name} -> {project_name}")
        print(f"  Twitter ID: {twitter_id}")
        print(f"  Fetching latest 30 tweets...")
        
        try:
            # Fetch tweets
            tweets = await twitter_service.get_user_tweets_by_id(
                user_id=twitter_id,
                max_results=30
            )
            
            if not tweets:
                print(f"  ⚠️  No tweets found for @{twitter_name}")
                results[twitter_name] = {
                    "project_name": project_name,
                    "tweets_fetched": 0,
                    "tweets_stored": 0,
                    "status": "no_tweets",
                }
                continue
            
            print(f"  ✅ Fetched {len(tweets)} tweets")
            
            # Add project metadata
            for tweet in tweets:
                tweet.meta.project = project_name
            
            # Generate embeddings
            print(f"  Generating embeddings...")
            texts = [tweet.text for tweet in tweets]
            embeddings = embedding_service.embed_batch(texts)
            
            # Store in database
            print(f"  Storing tweets in database...")
            count = tweet_repo.create(tweets, embeddings)
            
            print(f"  ✅ Successfully stored {count} tweets for {project_name}")
            
            results[twitter_name] = {
                "project_name": project_name,
                "tweets_fetched": len(tweets),
                "tweets_stored": count,
                "status": "success",
            }
            total_tweets += count
            
        except Exception as e:
            print(f"  ❌ Error processing @{twitter_name}: {e}")
            logger.error(f"Error processing @{twitter_name}: {e}", exc_info=True)
            results[twitter_name] = {
                "project_name": project_name,
                "tweets_fetched": 0,
                "tweets_stored": 0,
                "status": "error",
                "error": str(e),
            }
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print("=" * 80)
    print(f"  - Total accounts processed: {len(TWITTER_NAME_TO_ID)}")
    print(f"  - Total tweets stored: {total_tweets}")
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    no_tweets_count = sum(1 for r in results.values() if r["status"] == "no_tweets")
    
    print(f"  - Successful: {success_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - No tweets: {no_tweets_count}")
    
    print(f"\n{'=' * 80}")
    print("Detailed Results:")
    print("=" * 80)
    for twitter_name, result in sorted(results.items()):
        status_icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "error" else "⚠️"
        print(f"  {status_icon} @{twitter_name} -> {result['project_name']}: "
              f"{result['tweets_stored']} tweets stored")
        if result["status"] == "error":
            print(f"      Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'=' * 80}")
    print("JSON Format:")
    print("=" * 80)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

