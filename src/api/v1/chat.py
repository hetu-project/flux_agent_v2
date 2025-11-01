"""Chat API routes."""

from fastapi import APIRouter, HTTPException, Depends
from src.agents.rag_agent import RAGAgent
from src.agents.linkol_agent import LinkolAgent
from src.agents.hetu_agent import HetuAgent
from src.schemas.chat_schema import ChatRequest, ChatResponse, ChatMessage, Choice
from src.api.dependencies import get_rag_agent, get_linkol_agent, get_hetu_agent
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


@router.post("/linkol", response_model=ChatResponse)
async def chat_linkol(
    request: ChatRequest,
    linkol_agent: LinkolAgent = Depends(get_linkol_agent),
):
    """
    Chat with the Linkol agent.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question about Linkol or KOL"}
        ],
        "top_k": 5  // Optional, number of documents to retrieve
    }
    
    The agent will automatically:
    1. Analyze if the question is related to Linkol, KOL, or Twitter influencers
    2. If related, search for Linkol-related content in project_content
    3. Get top-ranked KOL and their price from Linkol API
    4. Generate response combining content and KOL data
    
    If the question is not Linkol-related, it will return a message asking
    the user to ask about Linkol-related topics.
    """
    try:
        # Extract user query from messages (get last message content)
        user_query = request.get_user_query()
        logger.info(f"Linkol chat request received: query='{user_query[:100]}...', model={request.model}")
        
        result = await linkol_agent.query(
            user_question=user_query,
            top_k=request.top_k
        )
        
        logger.info(f"Linkol chat response generated successfully")
        
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
        logger.error(f"Error processing Linkol chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hetu", response_model=ChatResponse)
async def chat_hetu(
    request: ChatRequest,
    hetu_agent: HetuAgent = Depends(get_hetu_agent),
):
    """
    Chat with the Hetu Protocol agent.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question about Hetu Protocol"}
        ],
        "top_k": 5  // Optional, number of documents to retrieve
    }
    
    The agent is specialized in answering questions about Hetu Protocol.
    It will:
    1. Always use Hetu Protocol project information from database
    2. Search for relevant content related to Hetu Protocol
    3. Search Twitter for recent tweets about Hetu Protocol (especially from 2025)
    4. Generate response as a Hetu Protocol introducer
    """
    try:
        # Extract user query from messages (get last message content)
        user_query = request.get_user_query()
        logger.info(f"Hetu chat request received: query='{user_query[:100]}...', model={request.model}")
        
        result = await hetu_agent.query(
            user_question=user_query,
            top_k=request.top_k
        )
        
        logger.info(f"Hetu chat response generated successfully")
        
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
        logger.error(f"Error processing Hetu chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

