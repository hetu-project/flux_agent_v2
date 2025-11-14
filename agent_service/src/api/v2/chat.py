"""Chat API routes V2 with cheaper models and RAG-based tweet search."""

from fastapi import APIRouter, HTTPException, Depends, Request
from src.agents.v2.rag_agent_v2 import RAGAgentV2
from src.agents.v2.linkol_agent_v2 import LinkolAgentV2
from src.agents.v2.hetu_agent_v2 import HetuAgentV2
from src.schemas.chat_schema import ChatRequest, ChatResponse, ChatMessage, Choice
from src.api.dependencies import get_rag_agent_v2, get_linkol_agent_v2, get_hetu_agent_v2
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v2/chat", tags=["chat-v2"])


def extract_api_key_from_auth_header(auth_header: str | None) -> str | None:
    """
    Extract API key from Authorization Bearer header.
    
    Args:
        auth_header: Authorization header value, e.g., "Bearer sk-xxxxx" or None
    
    Returns:
        API key string if valid Bearer token found, None otherwise
    """
    if not auth_header:
        return None
    
    # Normalize to handle case-insensitive "Bearer"
    auth_header_lower = auth_header.lower()
    if auth_header_lower.startswith("bearer "):
        # Extract token after "Bearer " prefix (case-insensitive)
        return auth_header[7:]  # Use original case for the token
    
    return None


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    rag_agent: RAGAgentV2 = Depends(get_rag_agent_v2),
):
    """
    Chat with the RAG agent V2 (using cheaper models and RAG-based tweet search).
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question"}
        ]
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    The agent will automatically:
    1. Extract user query from the last message in messages array
    2. Search for matching projects in database
    3. Search for relevant tweets using RAG (vector similarity search)
    4. If project found, include project info, content, and tweets in the response
    5. If project not found, answer directly with relevant tweets
    
    V2 improvements:
    - Uses cheaper model (gpt-4o-mini instead of grok)
    - Uses RAG-based tweet search from database instead of requiring LLM to search Twitter
    """
    try:
        # Extract API key from Authorization Bearer header (if provided, uses custom API)
        auth_header = http_request.headers.get("authorization") or http_request.headers.get("Authorization")
        api_key = extract_api_key_from_auth_header(auth_header)
        
        # Set base_url if API key is provided
        base_url = None
        if api_key:
            base_url = "https://aiclub.v1.hetu.org/v1"
        
        # Extract user query from messages (get last message content)
        user_query = request.get_user_query()
        logger.info(f"Chat V2 request received: query='{user_query[:100]}...', project={request.project}, model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await rag_agent.query(
            user_question=user_query,
            project=request.project,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"Chat V2 response generated successfully")
        
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
        logger.error(f"Error processing chat V2 request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/linkol", response_model=ChatResponse)
async def chat_linkol(
    request: ChatRequest,
    http_request: Request,
    linkol_agent: LinkolAgentV2 = Depends(get_linkol_agent_v2),
):
    """
    Chat with the Linkol agent V2 (using cheaper models and RAG-based tweet search).
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question about Linkol or KOL"}
        ],
        "top_k": 5  // Optional, number of documents to retrieve
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    The agent will automatically:
    1. Analyze if the question is related to Linkol, KOL, or Twitter influencers
    2. If related, search for Linkol-related content and tweets using RAG
    3. Get top-ranked KOL and their price from Linkol API
    4. Generate response combining content, tweets, and KOL data
    
    V2 improvements:
    - Uses cheaper model (gpt-4o-mini instead of grok)
    - Uses RAG-based tweet search from database instead of requiring LLM to search Twitter
    """
    try:
        # Extract API key from Authorization Bearer header (if provided, uses custom API)
        auth_header = http_request.headers.get("authorization") or http_request.headers.get("Authorization")
        api_key = extract_api_key_from_auth_header(auth_header)
        
        # Set base_url if API key is provided
        base_url = None
        if api_key:
            base_url = "https://aiclub.v1.hetu.org/v1"
        
        # Extract user query from messages (get last message content)
        user_query = request.get_user_query()
        logger.info(f"Linkol chat V2 request received: query='{user_query[:100]}...', model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await linkol_agent.query(
            user_question=user_query,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"Linkol chat V2 response generated successfully")
        
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
        logger.error(f"Error processing Linkol chat V2 request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hetu", response_model=ChatResponse)
async def chat_hetu(
    request: ChatRequest,
    http_request: Request,
    hetu_agent: HetuAgentV2 = Depends(get_hetu_agent_v2),
):
    """
    Chat with the Hetu Protocol agent V2 (using cheaper models and RAG-based tweet search).
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question about Hetu Protocol"}
        ],
        "top_k": 5  // Optional, number of documents to retrieve
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    The agent is specialized in answering questions about Hetu Protocol.
    It will:
    1. Always use Hetu Protocol project information from database
    2. Search for relevant content related to Hetu Protocol
    3. Search for relevant tweets using RAG (vector similarity search)
    4. Generate response as a Hetu Protocol introducer
    
    V2 improvements:
    - Uses cheaper model (gpt-4o-mini instead of grok)
    - Uses RAG-based tweet search from database instead of requiring LLM to search Twitter
    """
    try:
        # Extract API key from Authorization Bearer header (if provided, uses custom API)
        auth_header = http_request.headers.get("authorization") or http_request.headers.get("Authorization")
        api_key = extract_api_key_from_auth_header(auth_header)
        
        # Set base_url if API key is provided
        base_url = None
        if api_key:
            base_url = "https://aiclub.v1.hetu.org/v1"
        
        # Extract user query from messages (get last message content)
        user_query = request.get_user_query()
        logger.info(f"Hetu chat V2 request received: query='{user_query[:100]}...', model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await hetu_agent.query(
            user_question=user_query,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"Hetu chat V2 response generated successfully")
        
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
        logger.error(f"Error processing Hetu chat V2 request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

