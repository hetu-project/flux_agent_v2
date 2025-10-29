"""Chat API routes."""

from fastapi import APIRouter, HTTPException, Depends
from src.agents.rag_agent import RAGAgent
from src.schemas.chat_schema import ChatRequest, ChatResponse, Source
from src.api.dependencies import get_rag_agent


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_agent: RAGAgent = Depends(get_rag_agent),
):
    """
    Chat with the RAG agent.
    
    The agent will automatically:
    1. Judge if the question is about a project
    2. If yes, search for the relevant project in database
    3. If project found, include project info in the response
    4. If project not found or question is not about project, answer directly
    
    The 'project' parameter is optional and can be used to explicitly specify a project.
    If not provided, the agent will auto-detect and search for relevant projects.
    """
    try:
        result = await rag_agent.query(
            user_question=request.query,
            project=request.project,
            top_k=request.top_k
        )
        
        # Format sources (may be empty if not using tweet context)
        sources = [
            Source(
                text=s["text"],
                author=s["author"],
                created_at=s["created_at"],
                score=s["score"]
            )
            for s in result["sources"]
        ]
        
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            num_sources=result["num_sources"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

