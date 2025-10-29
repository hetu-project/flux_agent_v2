"""Tweet API routes."""

from fastapi import APIRouter, HTTPException, Depends
from src.repositories.tweet_repository import TweetRepository
from src.repositories.project_repository import ProjectRepository
from src.services.embedding import EmbeddingService
from src.services.tweet_service import TweetService
from src.schemas.tweet_schema import (
    CollectTweetsRequest,
    CollectTweetsResponse,
    TweetSearchRequest,
    TweetSearchResponse,
    TweetSearchResult,
)
from src.api.dependencies import (
    get_project_repo,
    get_tweet_repo,
    get_tweet_service,
    get_embedding_service,
)


router = APIRouter(prefix="/api/v1/tweets", tags=["tweets"])


@router.post("/collect", response_model=CollectTweetsResponse)
async def collect_tweets(
    request: CollectTweetsRequest,
    project_repo: ProjectRepository = Depends(get_project_repo),
    tweet_service: TweetService = Depends(get_tweet_service),
):
    """
    Collect tweets from Twitter and store in Qdrant.
    """
    try:
        # Check if project exists
        if not project_repo.exists(request.project_name):
            raise HTTPException(
                status_code=404,
                detail=f"Project '{request.project_name}' not found. Please create it first."
            )
        
        # Use service layer for business logic
        count = await tweet_service.collect_and_store_tweets(
            project_name=request.project_name,
            username=request.username,
            query=request.query,
            max_tweets=request.max_tweets
        )
        
        return CollectTweetsResponse(
            status="success",
            project=request.project_name,
            tweets_collected=count,
            message=f"Successfully collected and stored {count} tweets"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=TweetSearchResponse)
async def search_tweets(
    request: TweetSearchRequest,
    tweet_repo: TweetRepository = Depends(get_tweet_repo),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """
    Search tweets using vector similarity.
    """
    try:
        # Generate query embedding
        query_vector = embedding_service.embed_text(request.query)
        
        # Use repository to search
        results = tweet_repo.search(
            query_vector=query_vector,
            project=request.project,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # Format results
        search_results = [
            TweetSearchResult(
                id=r["id"],
                text=r["text"],
                author=r["author"],
                created_at=r["created_at"],
                project=r.get("project"),
                score=r["score"]
            )
            for r in results
        ]
        
        return TweetSearchResponse(
            results=search_results,
            total=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tweet_id}")
async def get_tweet(
    tweet_id: str,
    tweet_repo: TweetRepository = Depends(get_tweet_repo),
):
    """
    Get a tweet by ID.
    """
    try:
        results = tweet_repo.retrieve_by_ids([tweet_id])
        if not results:
            raise HTTPException(status_code=404, detail=f"Tweet '{tweet_id}' not found")
        
        return results[0]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tweet_id}", status_code=204)
async def delete_tweet(
    tweet_id: str,
    tweet_repo: TweetRepository = Depends(get_tweet_repo),
):
    """
    Delete a tweet by ID.
    """
    try:
        count = tweet_repo.delete([tweet_id])
        if count == 0:
            raise HTTPException(status_code=404, detail=f"Tweet '{tweet_id}' not found")
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

