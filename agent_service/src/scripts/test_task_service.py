"""Test script for TaskService."""

import asyncio
import sys
import os

# Add the app directory to the path so we can import from src
# When running in Docker, /app is the working directory
# When running locally, we need to add the parent of src
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from scripts/ to src/ to app/
app_dir = os.path.dirname(os.path.dirname(script_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from src.services.task_service import TaskService
from src.services.twitter import TwitterService
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Test TaskService."""
    print("=" * 80)
    print("Testing TaskService - get_all_twitter_names()")
    print("=" * 80)
    
    # Check if TASK_API_URL is set
    from src.config import get_settings
    settings = get_settings()
    if not settings.task_api_url:
        print("ERROR: TASK_API_URL not set in environment variables or .env file")
        print("Please set TASK_API_URL in your .env file")
        return
    
    print(f"\nTask API URL: {settings.task_api_url}")
    print("-" * 80)
    
    # Initialize TaskService
    task_service = TaskService()
    
    # Test: Get all Twitter names
    print("\nFetching all Twitter names from all tasks...")
    print("This will paginate through all tasks to collect unique Twitter names.")
    print("-" * 80)
    
    try:
        # Step 1: Get all Twitter names from tasks
        twitter_names = await task_service.get_all_twitter_names(limit=100)
        
        print(f"\n✅ Successfully fetched all Twitter names!")
        print(f"\nSummary:")
        print(f"  - Total unique Twitter names: {len(twitter_names)}")
        
        if not twitter_names:
            print("  No Twitter names found (all tasks have empty twitter_name)")
            return
        
        print(f"\n{'=' * 80}")
        print("Twitter Names (sorted):")
        print("=" * 80)
        
        sorted_names = sorted(twitter_names)
        for i, name in enumerate(sorted_names, 1):
            print(f"  {i}. @{name}")
        
        # Step 2: Get user IDs for each Twitter name
        print(f"\n{'=' * 80}")
        print("Fetching User IDs for each Twitter name...")
        print("=" * 80)
        
        twitter_service = TwitterService()
        twitter_name_to_id: dict[str, str] = {}
        failed_names: list[str] = []
        
        for i, name in enumerate(sorted_names, 1):
            print(f"\n[{i}/{len(sorted_names)}] Fetching ID for @{name}...")
            try:
                user_id = await twitter_service.get_user_id_by_username(name)
                twitter_name_to_id[name] = user_id
                print(f"  ✅ @{name} -> {user_id}")
            except Exception as e:
                print(f"  ❌ Failed to get ID for @{name}: {e}")
                failed_names.append(name)
                logger.warning(f"Failed to get user ID for @{name}: {e}")
        
        print(f"\n{'=' * 80}")
        print("Results Summary:")
        print("=" * 80)
        print(f"  - Successfully fetched IDs: {len(twitter_name_to_id)}")
        print(f"  - Failed to fetch IDs: {len(failed_names)}")
        
        if failed_names:
            print(f"\n  Failed names: {', '.join(f'@{name}' for name in failed_names)}")
        
        print(f"\n{'=' * 80}")
        print("Twitter Name to User ID Dictionary:")
        print("=" * 80)
        
        for name, user_id in sorted(twitter_name_to_id.items()):
            print(f"  @{name}: {user_id}")
        
        print(f"\n{'=' * 80}")
        print("Dictionary Format (JSON):")
        print("=" * 80)
        import json
        print(json.dumps(twitter_name_to_id, indent=2, ensure_ascii=False))
        
        return twitter_name_to_id
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error in test: {e}", exc_info=True)
        return


if __name__ == "__main__":
    asyncio.run(main())

