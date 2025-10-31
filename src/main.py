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

# Import API routes
from src.api.v1 import projects, tweets, chat, collections
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

# Initialize RAG agent with project repository
rag_agent = RAGAgent(project_repo=project_repo)

# Initialize repositories
tweet_repo = TweetRepository(qdrant_service, collection_name="twitter_tweets")
collection_repo = CollectionRepository(qdrant_service)

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
    tweet_repo=tweet_repo,
    tweet_service=tweet_service,
    embedding_service=embedding_service,
    rag_agent=rag_agent,
    collection_repo=collection_repo,
)
logger.info("Dependencies set successfully")

# Register routes
logger.info("Registering API routes...")
app.include_router(projects.router)
app.include_router(tweets.router)
app.include_router(chat.router)
app.include_router(collections.router)
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
