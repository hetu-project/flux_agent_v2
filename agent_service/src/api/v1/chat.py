"""Chat API routes."""

from fastapi import APIRouter, HTTPException, Depends, Request
from src.agents.rag_agent import RAGAgent
from src.agents.linkol_agent import LinkolAgent
from src.agents.hetu_agent import HetuAgent
from src.agents.agent_mcp.mcp_agent import MCPAgent
from src.agents.agent_mcp.hetu_agent import HetuMCPAgent
from src.agents.fortune_agent import FortuneAgent
from src.agents.health_agent import HealthAgent
from src.agents.bazi_agent import BaziAgent
from src.agents.crypto_agent import CryptoAgent
from src.agents.company_agent import CompanyAgent
from src.agents.tarot_agent import TarotAgent
from src.schemas.chat_schema import (
    ChatRequest, ChatResponse, ChatMessage, Choice,
    GetChatHistoryRequest, GetChatHistoryResponse,
    ChatHistoryConversation, ChatHistoryMessage
)
from src.api.dependencies import get_rag_agent, get_linkol_agent, get_hetu_agent, get_mcp_agent, get_hetu_mcp_agent, get_fortune_agent, get_health_agent, get_bazi_agent, get_crypto_agent, get_company_agent, get_tarot_agent
from src.services.database import get_async_session
from src.repositories.user_repository import UserRepository
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.sql_models.agent import Agent
from sqlalchemy import select
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


@router.post("/mcp/hetu", response_model=ChatResponse)
async def chat_mcp_hetu(
    request: ChatRequest,
    http_request: Request,
    hetu_mcp_agent: HetuMCPAgent = Depends(get_hetu_mcp_agent),
):
    """
    Chat with the Hetu Protocol MCP agent.
    
    This agent uses MCP (Model Context Protocol) service for tool calling instead of function calling.
    It is specialized in answering questions about Hetu Protocol and will:
    1. Always use Hetu Protocol project information from database
    2. Search for relevant content related to Hetu Protocol
    3. Get available MCP tools and let LLM choose appropriate tools to call
    4. Call MCP tools to get additional information
    5. Generate response as a Hetu Protocol introducer
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "your question about Hetu Protocol"}
        ],
        "top_k": 5  // Optional, number of documents to retrieve
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
        logger.info(f"Hetu MCP chat request received: query='{user_query[:100]}...', model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}")
        
        result = await hetu_mcp_agent.query(
            user_question=user_query,
            top_k=request.top_k,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"Hetu MCP chat response generated successfully")
        
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
        logger.error(f"Error processing Hetu MCP chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fortune", response_model=ChatResponse)
async def chat_fortune(
    request: ChatRequest,
    http_request: Request,
    tarot_agent: TarotAgent = Depends(get_tarot_agent),
):
    """
    Chat with the Tarot card reading agent (replaces Fortune Telling agent).
    
    Simple version: single interaction, automatic 3-card spread (Past-Present-Future).
    The agent will automatically draw cards and provide a complete reading based on the user's question.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "我想知道我的感情运势如何？"}
        ]
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    Returns OpenAI-compatible response format:
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "🔮 塔罗牌占卜结果..."
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
        
        logger.info(
            f"Tarot reading request received (via /fortune endpoint): query='{user_query[:100]}...', "
            f"model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}"
        )
        
        result = await tarot_agent.query(
            user_query=user_query,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info("Tarot reading response generated successfully")
        
        # Format response in OpenAI-compatible format
        message = ChatMessage(
            role="assistant",
            content=result["answer"]
        )
        
        choice = Choice(message=message)
        
        return ChatResponse(
            choices=[choice]
        )
        
    except Exception as e:
        logger.error(f"Error processing tarot reading request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health", response_model=ChatResponse)
async def chat_health(
    request: ChatRequest,
    http_request: Request,
    health_agent: HealthAgent = Depends(get_health_agent),
):
    """
    Chat with the Health agent.
    
    The agent provides health-related consultations and advice, including:
    - Nutrition advice
    - Exercise recommendations
    - Mental health support
    - Common disease prevention
    - General health tips
    
    Important: The agent provides general health information and advice only.
    It does not provide medical diagnosis. For serious symptoms, users should
    consult professional doctors.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "我最近总是感觉很累，有什么建议吗？"}
        ]
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    Returns OpenAI-compatible response format:
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "健康建议..."
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
            f"Health consultation request received: query='{user_query[:100]}...', "
            f"model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}"
        )
        
        result = await health_agent.query(
            user_query=user_query,
            conversation_history=conversation_history,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info("Health consultation generated successfully")
        
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
        logger.error(f"Error processing health consultation request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bazi", response_model=ChatResponse)
async def chat_bazi(
    request: ChatRequest,
    http_request: Request,
    bazi_agent: BaziAgent = Depends(get_bazi_agent),
):
    """
    Chat with the Bazi (Eight Characters) agent.
    
    The agent will extract birth information (lunar calendar date, time, and location) from the conversation.
    If information is incomplete, it will ask the user to provide missing information.
    
    Required information:
    - Birth year (lunar calendar)
    - Birth month (lunar calendar, 1-12)
    - Birth day (lunar calendar, 1-31)
    - Birth hour (0-23)
    - Birth minute (0-59)
    - Birth location (city name)
    - Current location (city name)
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "我1990年农历5月15日14时30分在北京出生，现在在上海，帮我算一下八字"}
        ],
        "session_id": "optional-session-id",  // Optional, for maintaining conversation context
        "user_id": "optional-user-id"  // Optional, for user-specific context (client_id)
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    Returns OpenAI-compatible response format:
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "八字计算结果..."
                }
            }
        ],
        "session_id": "session-id"  // Returned for subsequent requests
    }
    """
    # Get database session for context persistence
    async with get_async_session() as db_session:
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
            
            # Get session_id from request body (only use memory if session_id is explicitly provided)
            session_id = request.session_id
            
            # Generate or get session_id if not provided
            effective_session_id = session_id
            if not effective_session_id:
                # Generate a new session_id if not provided
                import uuid
                effective_session_id = str(uuid.uuid4())
                logger.info(f"Generated new session_id: {effective_session_id}")
            
            # Get or create user if user_id (client_id) is provided
            user_id_int = None
            if request.user_id:
                try:
                    user_repo = UserRepository(db_session)
                    user = await user_repo.get_or_create_by_client_id(request.user_id)
                    user_id_int = user.id
                    logger.debug(f"Resolved user_id: {request.user_id} -> {user_id_int}")
                except Exception as e:
                    logger.warning(f"Failed to resolve user_id '{request.user_id}': {e}")
            
            # Convert messages to conversation history format
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            logger.info(
                f"Bazi calculation request received: query='{user_query[:100]}...', "
                f"session_id={effective_session_id or 'none (no memory)'}, user_id={user_id_int}, "
                f"model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}"
            )
            
            # Set database session on agent for this request
            bazi_agent.db_session = db_session
            
            result = await bazi_agent.query(
                user_query=user_query,
                conversation_history=conversation_history,
                session_id=effective_session_id,
                user_id=user_id_int,
                api_key=api_key,
                base_url=base_url
            )
            
            answer_content = result.get("answer", "")
            answer_length = len(answer_content) if answer_content else 0
            logger.info(f"Bazi calculation generated successfully. Answer length: {answer_length} characters")
            
            # Log a preview of the answer (first and last 100 chars) for debugging
            if answer_content:
                preview_start = answer_content[:100] if len(answer_content) > 100 else answer_content
                preview_end = answer_content[-100:] if len(answer_content) > 100 else ""
                logger.debug(f"Answer preview - Start: {preview_start}... End: ...{preview_end}")
            
            # Format response in OpenAI-compatible format (consistent with other agents)
            message = ChatMessage(
                role="assistant",  # Hardcoded as assistant
                content=answer_content
            )
            
            choice = Choice(message=message)
            
            return ChatResponse(
                choices=[choice],
                session_id=effective_session_id  # Return session_id in response
            )
            
        except Exception as e:
            logger.error(f"Error processing bazi calculation request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/crypto", response_model=ChatResponse)
async def chat_crypto(
    request: ChatRequest,
    http_request: Request,
    crypto_agent: CryptoAgent = Depends(get_crypto_agent),
):
    """
    Chat with the Crypto agent.
    
    The agent provides cryptocurrency-related consultations and advice, including:
    - Cryptocurrency basics (Bitcoin, Ethereum, etc.)
    - Blockchain technology principles
    - Investment advice and risk warnings
    - Market analysis and trends
    - DeFi, NFT, Web3 introduction
    - Wallet usage and security advice
    
    Important: The agent provides educational information and general advice only.
    It does not provide specific investment advice or price predictions. Users should
    do their own research (DYOR) and invest cautiously.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "什么是比特币？"}
        ]
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    Returns OpenAI-compatible response format:
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "加密货币建议..."
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
            f"Crypto consultation request received: query='{user_query[:100]}...', "
            f"model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}"
        )
        
        result = await crypto_agent.query(
            user_query=user_query,
            conversation_history=conversation_history,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info("Crypto consultation generated successfully")
        
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
        logger.error(f"Error processing crypto consultation request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/company", response_model=ChatResponse)
async def chat_company(
    request: ChatRequest,
    http_request: Request,
    company_agent: CompanyAgent = Depends(get_company_agent),
):
    """
    Chat with the Company agent for companionship and daily conversation.
    
    The agent provides emotional companionship, daily conversation, and support, including:
    - Emotional companionship and listening
    - Daily conversation and chat
    - Encouragement and support
    - Sharing interesting topics
    - Helping relieve stress and loneliness
    - Positive emotional support
    
    This agent is designed to be warm, friendly, and empathetic, like a friend.
    It maintains conversation context through session_id and persists to database for more natural dialogue.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "今天心情不太好"}
        ],
        "session_id": "optional-session-id",  // Optional, for maintaining conversation context
        "user_id": "optional-user-id"  // Optional, for user-specific context (client_id)
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    Returns OpenAI-compatible response format:
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "陪伴回复..."
                }
            }
        ]
    }
    """
    # Get database session for context persistence
    async with get_async_session() as db_session:
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
            
            # Get session_id from request body (only use memory if session_id is explicitly provided)
            session_id = request.session_id
            
            # Get or create user if user_id (client_id) is provided
            user_id_int = None
            if request.user_id:
                try:
                    user_repo = UserRepository(db_session)
                    user = await user_repo.get_or_create_by_client_id(request.user_id)
                    user_id_int = user.id
                    logger.debug(f"Resolved user_id: {request.user_id} -> {user_id_int}")
                except Exception as e:
                    logger.warning(f"Failed to resolve user_id '{request.user_id}': {e}")
            
            # Convert messages to conversation history format
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            logger.info(
                f"Company companionship request received: query='{user_query[:100]}...', "
                f"session_id={session_id or 'none (no memory)'}, user_id={user_id_int}, "
                f"model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}"
            )
            
            # Set database session on agent for this request
            company_agent.db_session = db_session
            
            # Generate or get session_id if not provided
            effective_session_id = session_id
            if not effective_session_id:
                # Generate a new session_id if not provided
                import uuid
                effective_session_id = str(uuid.uuid4())
                logger.info(f"Generated new session_id: {effective_session_id}")
            
            result = await company_agent.query(
                user_query=user_query,
                conversation_history=conversation_history,
                session_id=effective_session_id,
                user_id=user_id_int,
                api_key=api_key,
                base_url=base_url
            )
            
            logger.info("Company companionship response generated successfully")
            
            # Format response in OpenAI-compatible format (consistent with other agents)
            message = ChatMessage(
                role="assistant",  # Hardcoded as assistant
                content=result["answer"]
            )
            
            choice = Choice(message=message)
            
            return ChatResponse(
                choices=[choice],
                session_id=effective_session_id  # Return session_id in response
            )
            
        except Exception as e:
            logger.error(f"Error processing company companionship request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/tarot", response_model=ChatResponse)
async def chat_tarot(
    request: ChatRequest,
    http_request: Request,
    tarot_agent: TarotAgent = Depends(get_tarot_agent),
):
    """
    Chat with the Tarot card reading agent.
    
    Simple version: single interaction, automatic 3-card spread (Past-Present-Future).
    The agent will automatically draw cards and provide a complete reading.
    
    Accepts OpenAI-compatible request format:
    {
        "model": "any-model-name",  // Ignored
        "messages": [
            {"role": "user", "content": "我想知道我的感情运势如何？"}
        ]
    }
    
    Optional header: Authorization: Bearer <api-key> - If provided, uses OpenRouter API instead of AIHubMix
    
    Returns OpenAI-compatible response format:
    {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "🔮 塔罗牌占卜结果..."
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
        
        logger.info(
            f"Tarot reading request received: query='{user_query[:100]}...', "
            f"model={request.model}, using_api={'Custom API' if api_key else 'AIHubMix'}"
        )
        
        result = await tarot_agent.query(
            user_query=user_query,
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info("Tarot reading response generated successfully")
        
        # Format response in OpenAI-compatible format
        message = ChatMessage(
            role="assistant",
            content=result["answer"]
        )
        
        choice = Choice(message=message)
        
        return ChatResponse(
            choices=[choice]
        )
        
    except Exception as e:
        logger.error(f"Error processing tarot reading request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

