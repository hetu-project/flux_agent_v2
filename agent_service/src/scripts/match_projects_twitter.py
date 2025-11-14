"""Script to match project names with Twitter names."""

import asyncio
import sys
import os
import json

# Add the app directory to the path so we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(script_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from src.services.task_service import TaskService
from src.services.twitter import TwitterService
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Project name to Twitter name mapping
# Based on the project names and Twitter names we've seen
PROJECT_TO_TWITTER_MAPPING = {
    "AuraSci": "Aura_Sci",
    "ChaosChain": "Ch40sChain",
    "Outland": "outland_art",
    "OpenBuild": "OpenBuildxyz",
    "DePHY": "dephynetwork",
    "Hetu Protocol": "hetu_protocol",
    "Akasha Dao": "AkashaDao",
    "PredX": "PredX_AI",
    "Linkol": "linkolfun",
    "dLife": "dlifexyz",
    "Loka": "lokachain",
    "499DAO": "499_DAO",
}


async def main():
    """Match projects with Twitter names and get user IDs."""
    print("=" * 80)
    print("Matching Projects with Twitter Names")
    print("=" * 80)
    
    # Step 1: Get all Twitter names from tasks
    print("\nStep 1: Fetching Twitter names from tasks...")
    print("-" * 80)
    
    task_service = TaskService()
    twitter_names = await task_service.get_all_twitter_names(limit=100)
    
    print(f"✅ Found {len(twitter_names)} unique Twitter names")
    
    # Step 2: Get Twitter user IDs
    print(f"\nStep 2: Fetching User IDs for Twitter names...")
    print("-" * 80)
    
    twitter_service = TwitterService()
    twitter_name_to_id: dict[str, str] = {}
    
    for name in sorted(twitter_names):
        try:
            user_id = await twitter_service.get_user_id_by_username(name)
            twitter_name_to_id[name] = user_id
            print(f"  ✅ @{name} -> {user_id}")
        except Exception as e:
            print(f"  ❌ Failed to get ID for @{name}: {e}")
            logger.warning(f"Failed to get user ID for @{name}: {e}")
    
    # Step 3: Create project to Twitter ID mapping
    print(f"\n{'=' * 80}")
    print("Step 3: Creating Project to Twitter ID mapping...")
    print("=" * 80)
    
    project_to_twitter_id: dict[str, str] = {}
    project_to_twitter_name: dict[str, str] = {}
    unmatched_projects: list[str] = []
    unmatched_twitter_names: set[str] = set(twitter_names)
    
    for project_name, twitter_name in PROJECT_TO_TWITTER_MAPPING.items():
        if twitter_name in twitter_name_to_id:
            project_to_twitter_id[project_name] = twitter_name_to_id[twitter_name]
            project_to_twitter_name[project_name] = twitter_name
            unmatched_twitter_names.discard(twitter_name)
            print(f"  ✅ {project_name} -> @{twitter_name} -> {twitter_name_to_id[twitter_name]}")
        else:
            unmatched_projects.append(project_name)
            print(f"  ⚠️  {project_name} -> @{twitter_name} (Twitter name not found in tasks)")
    
    # Display results
    print(f"\n{'=' * 80}")
    print("Results Summary:")
    print("=" * 80)
    print(f"  - Total projects in mapping: {len(PROJECT_TO_TWITTER_MAPPING)}")
    print(f"  - Successfully matched: {len(project_to_twitter_id)}")
    print(f"  - Unmatched projects: {len(unmatched_projects)}")
    print(f"  - Unmatched Twitter names: {len(unmatched_twitter_names)}")
    
    if unmatched_projects:
        print(f"\n  Unmatched projects: {', '.join(unmatched_projects)}")
    
    if unmatched_twitter_names:
        print(f"\n  Unmatched Twitter names: {', '.join(f'@{name}' for name in sorted(unmatched_twitter_names))}")
    
    # Display mappings
    print(f"\n{'=' * 80}")
    print("Project to Twitter Name Mapping:")
    print("=" * 80)
    for project_name, twitter_name in sorted(project_to_twitter_name.items()):
        print(f"  {project_name} -> @{twitter_name}")
    
    print(f"\n{'=' * 80}")
    print("Project to Twitter ID Mapping:")
    print("=" * 80)
    for project_name, twitter_id in sorted(project_to_twitter_id.items()):
        print(f"  {project_name} -> {twitter_id}")
    
    # JSON output
    print(f"\n{'=' * 80}")
    print("JSON Format - Project to Twitter Name:")
    print("=" * 80)
    print(json.dumps(project_to_twitter_name, indent=2, ensure_ascii=False))
    
    print(f"\n{'=' * 80}")
    print("JSON Format - Project to Twitter ID:")
    print("=" * 80)
    print(json.dumps(project_to_twitter_id, indent=2, ensure_ascii=False))
    
    return {
        "project_to_twitter_name": project_to_twitter_name,
        "project_to_twitter_id": project_to_twitter_id,
        "unmatched_projects": unmatched_projects,
        "unmatched_twitter_names": list(unmatched_twitter_names),
    }


if __name__ == "__main__":
    asyncio.run(main())

