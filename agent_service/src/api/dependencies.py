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
from src.agents.hetu_agent import HetuAgent
from src.agents.agent_mcp.mcp_agent import MCPAgent
from src.agents.agent_mcp.hetu_agent import HetuMCPAgent
from src.agents.v2.rag_agent_v2 import RAGAgentV2
from src.agents.v2.linkol_agent_v2 import LinkolAgentV2
from src.agents.v2.hetu_agent_v2 import HetuAgentV2
from src.agents.fortune_agent import FortuneAgent
from src.agents.health_agent import HealthAgent
from src.agents.bazi_agent import BaziAgent
from src.agents.crypto_agent import CryptoAgent
from src.agents.company_agent import CompanyAgent


# Global instances (initialized in main.py)
_project_repo: ProjectRepository = None
_project_content_repo: ProjectContentRepository = None
_tweet_repo: TweetRepository = None
_tweet_service: TweetService = None
_embedding_service: EmbeddingService = None
_rag_agent: RAGAgent = None
_linkol_agent: LinkolAgent = None
_hetu_agent: HetuAgent = None
_mcp_agent: MCPAgent = None
_hetu_mcp_agent: HetuMCPAgent = None
_collection_repo: CollectionRepository = None
_rag_agent_v2: RAGAgentV2 = None
_linkol_agent_v2: LinkolAgentV2 = None
_hetu_agent_v2: HetuAgentV2 = None
_fortune_agent: FortuneAgent = None
_health_agent: HealthAgent = None
_bazi_agent: BaziAgent = None
_crypto_agent: CryptoAgent = None
_company_agent: CompanyAgent = None


def set_dependencies(
    project_repo: ProjectRepository,
    project_content_repo: ProjectContentRepository,
    tweet_repo: TweetRepository,
    tweet_service: TweetService,
    embedding_service: EmbeddingService,
    rag_agent: RAGAgent,
    linkol_agent: LinkolAgent,
    hetu_agent: HetuAgent,
    mcp_agent: MCPAgent,
    collection_repo: CollectionRepository,
    rag_agent_v2: RAGAgentV2 = None,
    linkol_agent_v2: LinkolAgentV2 = None,
    hetu_agent_v2: HetuAgentV2 = None,
    fortune_agent: FortuneAgent = None,
    hetu_mcp_agent: HetuMCPAgent = None,
    health_agent: HealthAgent = None,
    bazi_agent: BaziAgent = None,
    crypto_agent: CryptoAgent = None,
    company_agent: CompanyAgent = None,
):
    """Set global dependencies."""
    global _project_repo, _project_content_repo, _tweet_repo, _tweet_service, _embedding_service, _rag_agent, _linkol_agent, _hetu_agent, _mcp_agent, _hetu_mcp_agent, _collection_repo, _rag_agent_v2, _linkol_agent_v2, _hetu_agent_v2, _fortune_agent, _health_agent, _bazi_agent, _crypto_agent, _company_agent
    _project_repo = project_repo
    _project_content_repo = project_content_repo
    _tweet_repo = tweet_repo
    _tweet_service = tweet_service
    _embedding_service = embedding_service
    _rag_agent = rag_agent
    _linkol_agent = linkol_agent
    _hetu_agent = hetu_agent
    _mcp_agent = mcp_agent
    _hetu_mcp_agent = hetu_mcp_agent
    _collection_repo = collection_repo
    _rag_agent_v2 = rag_agent_v2
    _linkol_agent_v2 = linkol_agent_v2
    _hetu_agent_v2 = hetu_agent_v2
    _fortune_agent = fortune_agent
    _health_agent = health_agent
    _bazi_agent = bazi_agent
    _crypto_agent = crypto_agent
    _company_agent = company_agent


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


def get_hetu_agent() -> HetuAgent:
    return _hetu_agent


def get_mcp_agent() -> MCPAgent:
    return _mcp_agent


def get_rag_agent_v2() -> RAGAgentV2:
    return _rag_agent_v2


def get_linkol_agent_v2() -> LinkolAgentV2:
    return _linkol_agent_v2


def get_hetu_agent_v2() -> HetuAgentV2:
    return _hetu_agent_v2


def get_fortune_agent() -> FortuneAgent:
    return _fortune_agent


def get_hetu_mcp_agent() -> HetuMCPAgent:
    return _hetu_mcp_agent


def get_health_agent() -> HealthAgent:
    return _health_agent


def get_bazi_agent() -> BaziAgent:
    return _bazi_agent


def get_crypto_agent() -> CryptoAgent:
    return _crypto_agent


def get_company_agent() -> CompanyAgent:
    return _company_agent

