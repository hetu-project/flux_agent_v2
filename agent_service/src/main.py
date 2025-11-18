"""FastAPI application."""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.utils.logger import setup_logging, get_logger

# Initialize logging first
setup_logging()
logger = get_logger(__name__)
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.services.twitter import TwitterService
from src.services.tweet_service import TweetService
from src.agents.rag_agent import RAGAgent
from src.repositories.tweet_repository import TweetRepository
from src.repositories.collection_repository import CollectionRepository
from src.repositories.project_repository import ProjectRepository
from src.repositories.project_content_repository import ProjectContentRepository
from src.agents.linkol_agent import LinkolAgent
from src.agents.hetu_agent import HetuAgent
from src.agents.agent_mcp.mcp_agent import MCPAgent
from src.agents.v2.rag_agent_v2 import RAGAgentV2
from src.agents.v2.linkol_agent_v2 import LinkolAgentV2
from src.agents.v2.hetu_agent_v2 import HetuAgentV2
from src.agents.fortune_agent import FortuneAgent

# Import API routes
from src.api.v1 import projects, tweets, chat, collections, project_content
from src.api.v2 import chat as chat_v2
from src.api.dependencies import set_dependencies

# Initialize app
app = FastAPI(
    title="Hetu Agent",
    description="RAG Agent for Twitter project analysis",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services and repositories
logger.info("Initializing services and repositories...")
settings = get_settings()
qdrant_service = QdrantService()
embedding_service = EmbeddingService()
twitter_service = TwitterService()
logger.info("Services initialized successfully")

# Initialize repositories first
project_repo = ProjectRepository(
    qdrant_service=qdrant_service,
    embedding_service=embedding_service,
    collection_name="projects"
)

# Initialize project content repository (for unified storage of tweets, papers, etc.)
project_content_repo = ProjectContentRepository(
    qdrant_service=qdrant_service,
    embedding_service=embedding_service,
    collection_name="project_content"
)

# Initialize RAG agent with project repository and project content repository
rag_agent = RAGAgent(
    project_repo=project_repo,
    project_content_repo=project_content_repo
)

# Initialize Linkol agent with project content repository
linkol_agent = LinkolAgent(
    project_content_repo=project_content_repo
)

# Initialize Hetu agent with project repository and project content repository
hetu_agent = HetuAgent(
    project_repo=project_repo,
    project_content_repo=project_content_repo
)

# Initialize MCP agent (uses MCP service for tool calling)
mcp_agent = MCPAgent()

# Initialize repositories
tweet_repo = TweetRepository(qdrant_service, collection_name="twitter_tweets")
collection_repo = CollectionRepository(qdrant_service)

# Initialize V2 agents (with cheaper models and RAG-based tweet search)
rag_agent_v2 = RAGAgentV2(
    project_repo=project_repo,
    project_content_repo=project_content_repo,
    tweet_repo=tweet_repo
)

linkol_agent_v2 = LinkolAgentV2(
    project_content_repo=project_content_repo,
    tweet_repo=tweet_repo
)

hetu_agent_v2 = HetuAgentV2(
    project_repo=project_repo,
    project_content_repo=project_content_repo,
    tweet_repo=tweet_repo
)

# Initialize Fortune agent
fortune_agent = FortuneAgent()

# Initialize business services
tweet_service = TweetService(
    tweet_repo=tweet_repo,
    collection_repo=collection_repo,
    twitter_service=twitter_service,
    embedding_service=embedding_service,
)

# Set dependencies for API routes
logger.info("Setting up dependencies...")
set_dependencies(
    project_repo=project_repo,
    project_content_repo=project_content_repo,
    tweet_repo=tweet_repo,
    tweet_service=tweet_service,
    embedding_service=embedding_service,
    rag_agent=rag_agent,
    linkol_agent=linkol_agent,
    hetu_agent=hetu_agent,
    mcp_agent=mcp_agent,
    collection_repo=collection_repo,
    rag_agent_v2=rag_agent_v2,
    linkol_agent_v2=linkol_agent_v2,
    hetu_agent_v2=hetu_agent_v2,
    fortune_agent=fortune_agent,
)
logger.info("Dependencies set successfully")

# Register routes
logger.info("Registering API routes...")
app.include_router(projects.router)
app.include_router(tweets.router)
app.include_router(chat.router)
app.include_router(collections.router)
app.include_router(project_content.router)
# Register V2 routes
app.include_router(chat_v2.router)
logger.info("API routes registered successfully")
logger.info("Application startup complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Hetu Agent API",
        "version": "0.1.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
