"""Collection API routes."""

from fastapi import APIRouter, HTTPException, Depends
from src.repositories.collection_repository import CollectionRepository
from src.schemas.collection_schema import CollectionInfo, CollectionCreateRequest
from src.api.dependencies import get_collection_repo


router = APIRouter(prefix="/api/v1/collections", tags=["collections"])


@router.get("/{collection_name}/info", response_model=CollectionInfo)
async def get_collection_info(
    collection_name: str,
    collection_repo: CollectionRepository = Depends(get_collection_repo),
):
    """
    Get information about a collection.
    """
    try:
        info = collection_repo.get_info(collection_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
        
        return CollectionInfo(**info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=CollectionInfo, status_code=201)
async def create_collection(
    request: CollectionCreateRequest,
    collection_repo: CollectionRepository = Depends(get_collection_repo),
):
    """
    Create a new collection.
    """
    try:
        success = collection_repo.create(request.name, request.vector_size)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to create collection '{request.name}'")
        
        info = collection_repo.get_info(request.name)
        return CollectionInfo(**info)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

