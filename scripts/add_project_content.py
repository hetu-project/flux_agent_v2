"""Script to add project content (papers, tweets, etc.) directly to database."""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repositories.project_content_repository import ProjectContentRepository
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.models.project_content import ProjectContent
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def create_paper_content(
    content_id: str,
    project_name: str,
    title: str,
    author: str,
    content: str,
    source_url: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    paper_id: Optional[str] = None,
    created_at: Optional[datetime] = None
) -> ProjectContent:
    """Create a paper ProjectContent object."""
    return ProjectContent(
        content_id=content_id,
        project_name=project_name,
        content_type="paper",
        content=content,
        title=title,
        author=author,
        source_url=source_url,
        created_at=created_at or datetime.now(),
        arxiv_id=arxiv_id,
        paper_id=paper_id,
    )


def create_tweet_content(
    content_id: str,
    project_name: str,
    content: str,
    author: str,
    tweet_id: Optional[str] = None,
    author_id: Optional[str] = None,
    likes: Optional[int] = None,
    retweets: Optional[int] = None,
    replies: Optional[int] = None,
    source_url: Optional[str] = None,
    created_at: Optional[datetime] = None
) -> ProjectContent:
    """Create a tweet ProjectContent object."""
    return ProjectContent(
        content_id=content_id,
        project_name=project_name,
        content_type="tweet",
        content=content,
        author=author,
        tweet_id=tweet_id or content_id,
        author_id=author_id,
        likes=likes,
        retweets=retweets,
        replies=replies,
        source_url=source_url,
        created_at=created_at or datetime.now(),
    )


async def add_content(content: ProjectContent) -> bool:
    """Add a single content item to the database."""
    try:
        logger.info(f"Initializing services...")
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()
        
        repo = ProjectContentRepository(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            collection_name="project_content"
        )
        
        logger.info(f"Adding content: {content.content_type} for project '{content.project_name}'")
        logger.info(f"Content ID: {content.content_id}")
        logger.info(f"Title: {content.title or 'N/A'}")
        logger.info(f"Content length: {len(content.content)} characters")
        
        # Create content in repository (embedding will be generated automatically)
        created_content = repo.create(content)
        
        logger.info(f"✅ Content '{created_content.content_id}' added successfully!")
        logger.info(f"   Type: {created_content.content_type}")
        logger.info(f"   Project: {created_content.project_name}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error adding content: {e}")
        import traceback
        traceback.print_exc()
        return False


async def add_content_from_dict(data: dict) -> bool:
    """Add content from a dictionary."""
    try:
        # Parse datetime if provided as string
        created_at = None
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
            else:
                created_at = data["created_at"]
        
        # Create ProjectContent from dict
        content = ProjectContent(
            content_id=data["content_id"],
            project_name=data["project_name"],
            content_type=data["content_type"],
            content=data["content"],
            title=data.get("title"),
            author=data.get("author"),
            source_url=data.get("source_url"),
            created_at=created_at,
            tweet_id=data.get("tweet_id"),
            author_id=data.get("author_id"),
            likes=data.get("likes"),
            retweets=data.get("retweets"),
            replies=data.get("replies"),
            paper_id=data.get("paper_id"),
            arxiv_id=data.get("arxiv_id"),
        )
        
        return await add_content(content)
    except Exception as e:
        logger.error(f"❌ Error creating content from dict: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add project content to database")
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to JSON file containing content data"
    )
    parser.add_argument(
        "--content-id",
        type=str,
        help="Content ID"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        help="Project name"
    )
    parser.add_argument(
        "--content-type",
        type=str,
        choices=["tweet", "paper", "document", "blog", "project_description"],
        help="Content type"
    )
    parser.add_argument(
        "--content",
        type=str,
        help="Content text"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title (for papers, documents, blogs)"
    )
    parser.add_argument(
        "--author",
        type=str,
        help="Author"
    )
    
    args = parser.parse_args()
    
    # If file is provided, read from file
    if args.file:
        logger.info(f"Reading content from file: {args.file}")
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Multiple items
            logger.info(f"Found {len(data)} content items")
            for item in data:
                asyncio.run(add_content_from_dict(item))
        else:
            # Single item
            asyncio.run(add_content_from_dict(data))
    else:
        # Use command line arguments (for simple cases)
        if not all([args.content_id, args.project_name, args.content_type, args.content]):
            parser.error("--content-id, --project-name, --content-type, and --content are required when not using --file")
        
        data = {
            "content_id": args.content_id,
            "project_name": args.project_name,
            "content_type": args.content_type,
            "content": args.content,
        }
        
        if args.title:
            data["title"] = args.title
        if args.author:
            data["author"] = args.author
        
        asyncio.run(add_content_from_dict(data))


if __name__ == "__main__":
    main()

