"""Chat API routes."""

from fastapi import APIRouter, HTTPException, Depends, Request
from src.agents.rag_agent import RAGAgent
from src.agents.linkol_agent import LinkolAgent
from src.agents.hetu_agent import HetuAgent
from src.agents.agent_mcp.mcp_agent import MCPAgent
from src.agents.fortune_agent import FortuneAgent
from src.schemas.chat_schema import ChatRequest, ChatResponse, ChatMessage, Choice
from src.api.dependencies import get_rag_agent, get_linkol_agent, get_hetu_agent, get_mcp_agent, get_fortune_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


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
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    The agent will automatically:
    1. Extract user query from the last message in messages array
    2. Search for matching projects in database
    3. If project found, include project info in the response
    4. If project not found, answer directly
    
    The 'project' parameter is optional and can be used to explicitly specify a project.
    If not provided, the agent will auto-detect and search for relevant projects.
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
        logger.info(f"Chat request received: query='{user_query[:100]}...', project={request.project}, model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await rag_agent.query(
            user_question=user_query,
            project=request.project,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
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
    http_request: Request,
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
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    The agent will automatically:
    1. Analyze if the question is related to Linkol, KOL, or Twitter influencers
    2. If related, search for Linkol-related content in project_content
    3. Get top-ranked KOL and their price from Linkol API
    4. Generate response combining content and KOL data
    
    If the question is not Linkol-related, it will return a message asking
    the user to ask about Linkol-related topics.
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
        logger.info(f"Linkol chat request received: query='{user_query[:100]}...', model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await linkol_agent.query(
            user_question=user_query,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
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
    http_request: Request,
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
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    The agent is specialized in answering questions about Hetu Protocol.
    It will:
    1. Always use Hetu Protocol project information from database
    2. Search for relevant content related to Hetu Protocol
    3. Search Twitter for recent tweets about Hetu Protocol (especially from 2025)
    4. Generate response as a Hetu Protocol introducer
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
        logger.info(f"Hetu chat request received: query='{user_query[:100]}...', model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await hetu_agent.query(
            user_question=user_query,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
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


@router.post("/mcp", response_model=ChatResponse)
async def chat_mcp(
    request: ChatRequest,
    http_request: Request,
    mcp_agent: MCPAgent = Depends(get_mcp_agent),
):
    """
    Chat with the MCP agent.
    
    This agent uses MCP (Model Context Protocol) service for tool calling instead of function calling.
    It will:
    1. Get all available tools from MCP server
    2. Let LLM analyze the question and choose the appropriate tool
    3. Call the tool through MCP service
    4. Generate final answer based on tool results
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question"}
        ],
        "top_k": 5  // Optional, ignored for MCP agent
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
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
        logger.info(f"MCP chat request received: query='{user_query[:100]}...', model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await mcp_agent.query(
            user_question=user_query,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"MCP chat response generated successfully")
        
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
        logger.error(f"Error processing MCP chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fortune", response_model=ChatResponse)
async def chat_fortune(
    request: ChatRequest,
    http_request: Request,
    fortune_agent: FortuneAgent = Depends(get_fortune_agent),
):
    """
    Chat with the Fortune Telling agent.
    
    The agent will extract name, birth year, and zodiac sign from the conversation.
    If information is incomplete, it will ask the user to provide missing information.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "我叫张三，1990年出生，白羊座，帮我算一下明天的运势"}
        ]
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    Returns OpenAI-compatible response format:
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "明天的运势预测..."
                }
            }
        ]
    }
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
        
        # Convert messages to conversation history format
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        logger.info(
            f"Fortune prediction request received: query='{user_query[:100]}...', "
            f"model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}"
        )
        
        result = await fortune_agent.query(
            user_query=user_query,
            conversation_history=conversation_history,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info("Fortune prediction generated successfully")
        
        # Format response in OpenAI-compatible format (consistent with other agents)
        message = ChatMessage(
            role="assistant",  # Hardcoded as assistant
            content=result["answer"]
        )
        
        choice = Choice(message=message)
        
        return ChatResponse(
            choices=[choice]
        )
        
    except Exception as e:
        logger.error(f"Error processing fortune prediction request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

