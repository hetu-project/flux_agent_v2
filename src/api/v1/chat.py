"""Chat API routes."""

from fastapi import APIRouter, HTTPException, Depends
from src.agents.rag_agent import RAGAgent
from src.schemas.chat_schema import ChatRequest, ChatResponse, ChatMessage, Choice
from src.api.dependencies import get_rag_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_agent: RAGAgent = Depends(get_rag_agent),
):
    """
    Chat with the RAG agent.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question"}
        ]
    }
    
    The agent will automatically:
    1. Extract user query from the last message in messages array
    2. Search for matching projects in database
    3. If project found, include project info in the response
    4. If project not found, answer directly
    
    The 'project' parameter is optional and can be used to explicitly specify a project.
    If not provided, the agent will auto-detect and search for relevant projects.
    """
    try:
        # Extract user query from messages (get last message content)
        user_query = request.get_user_query()
        logger.info(f"Chat request received: query='{user_query[:100]}...', project={request.project}, model={request.model}")
        
        result = await rag_agent.query(
            user_question=user_query,
            project=request.project,
            top_k=request.top_k
        )
        
        logger.info(f"Chat response generated successfully")
        
        # Format response in OpenAI-compatible format
        message = ChatMessage(
            role="assistant",  # Hardcoded as assistant
            content=result["answer"]
        )
        
        choice = Choice(message=message)
        
        return ChatResponse(
            choices=[choice]
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

