"""Dependency injection for API routes."""

from functools import lru_cache
from src.services.embedding import EmbeddingService
from src.services.twitter import TwitterService
from src.services.tweet_service import TweetService
from src.services.qdrant_client import QdrantService
from src.repositories.tweet_repository import TweetRepository
from src.repositories.collection_repository import CollectionRepository
from src.repositories.project_repository import ProjectRepository
from src.repositories.project_content_repository import ProjectContentRepository
from src.agents.rag_agent import RAGAgent
from src.agents.linkol_agent import LinkolAgent


# Global instances (initialized in main.py)
_project_repo: ProjectRepository = None
_project_content_repo: ProjectContentRepository = None
_tweet_repo: TweetRepository = None
_tweet_service: TweetService = None
_embedding_service: EmbeddingService = None
_rag_agent: RAGAgent = None
_linkol_agent: LinkolAgent = None
_collection_repo: CollectionRepository = None


def set_dependencies(
    project_repo: ProjectRepository,
    project_content_repo: ProjectContentRepository,
    tweet_repo: TweetRepository,
    tweet_service: TweetService,
    embedding_service: EmbeddingService,
    rag_agent: RAGAgent,
    linkol_agent: LinkolAgent,
    collection_repo: CollectionRepository,
):
    """Set global dependencies."""
    global _project_repo, _project_content_repo, _tweet_repo, _tweet_service, _embedding_service, _rag_agent, _linkol_agent, _collection_repo
    _project_repo = project_repo
    _project_content_repo = project_content_repo
    _tweet_repo = tweet_repo
    _tweet_service = tweet_service
    _embedding_service = embedding_service
    _rag_agent = rag_agent
    _linkol_agent = linkol_agent
    _collection_repo = collection_repo


def get_project_repo() -> ProjectRepository:
    return _project_repo


def get_project_content_repo() -> ProjectContentRepository:
    return _project_content_repo


def get_tweet_repo() -> TweetRepository:
    return _tweet_repo


def get_tweet_service() -> TweetService:
    return _tweet_service


def get_embedding_service() -> EmbeddingService:
    return _embedding_service


def get_rag_agent() -> RAGAgent:
    return _rag_agent


def get_linkol_agent() -> LinkolAgent:
    return _linkol_agent


def get_collection_repo() -> CollectionRepository:
    return _collection_repo

