"""Company Agent for companionship and daily conversation."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sql_models.conversation import Conversation
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.config import get_settings
from src.utils.logger import get_logger
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.sql_models.agent import Agent

settings = get_settings()
logger = get_logger(__name__)


class CompanyAgent:
    """Company Agent for providing companionship, emotional support, and daily conversation."""
    
    AGENT_NAME = "company"  # Agent name in database
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize Company Agent.
        
        Args:
            db_session: Optional async database session for context persistence
        """
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = settings.chat_model
        
        # Initialize LangChain ChatOpenAI (default)
        self.langchain_llm = ChatOpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url,
            model=settings.chat_model,
            temperature=0.7  # Higher temperature for more natural, friendly conversation
        )
        
        # Initialize LangChain Memory storage for different sessions
        # Each session_id will have its own memory instance
        # Keep last 15 messages for context (more than other agents for better companionship)
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}
        self._default_memory_window = 15
        
        # Database session for CRDT operations
        self.db_session = db_session
        self._agent_id: Optional[int] = None
    
    def _get_llm_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
        """
        Get LLM client with optional API key and base URL.
        If provided, uses OpenRouter; otherwise uses default AIHubMix.
        
        Args:
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            OpenAI client instance
        """
        if api_key:
            # Use OpenRouter if API key is provided
            client_base_url = base_url or settings.openrouter_base_url
            return OpenAI(
                api_key=api_key,
                base_url=client_base_url
            )
        else:
            # Use default AIHubMix
            return self.llm
    
    def _get_langchain_llm(self, api_key: Optional[str] = None, base_url: Optional[str] = None, max_tokens: Optional[int] = None) -> ChatOpenAI:
        """
        Get LangChain ChatOpenAI client with optional API key and base URL.
        
        Args:
            api_key: Optional API key (if provided, uses custom API)
            base_url: Optional base URL (if provided, uses this URL)
            max_tokens: Optional maximum tokens for response generation
            
        Returns:
            LangChain ChatOpenAI instance
        """
        if api_key:
            # Use custom API if provided
            client_base_url = base_url or settings.openrouter_base_url
            model_name = settings.openrouter_model
        else:
            # Use default AIHubMix
            client_base_url = settings.aihubmix_base_url
            model_name = self.chat_model
        
        llm_kwargs = {
            "api_key": api_key or settings.aihubmix_api_key,
            "base_url": client_base_url,
            "model": model_name,
            "temperature": 0.8  # Higher temperature for more natural, warm conversation
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        
        return ChatOpenAI(**llm_kwargs)
    
    def _get_or_create_memory(self, session_id: Optional[str] = None) -> Optional[ConversationBufferWindowMemory]:
        """
        Get or create a memory instance for a specific session.
        If no session_id is provided, returns None (no memory storage).
        
        Args:
            session_id: Optional session identifier. If None, returns None
            
        Returns:
            ConversationBufferWindowMemory instance for the session, or None if no session_id
        """
        # If no session_id provided, don't use memory
        if not session_id:
            return None
        
        # Create memory if it doesn't exist for this session
        if session_id not in self._memories:
            self._memories[session_id] = ConversationBufferWindowMemory(
                k=self._default_memory_window,
                return_messages=True,
                memory_key="chat_history"
            )
            logger.debug(f"Created new memory for session: {session_id}")
        
        return self._memories[session_id]
    
    def _load_conversation_history_to_memory(
        self, 
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: Optional[str] = None
    ):
        """
        Load conversation history into LangChain memory for a specific session.
        Only loads if session_id is provided.
        
        Args:
            conversation_history: List of messages with role and content
            session_id: Optional session identifier. If None, no memory is stored.
        """
        # If no session_id, don't store memory
        if not session_id:
            return
        
        if not conversation_history:
            return
        
        # Get memory for this session
        memory = self._get_or_create_memory(session_id)
        if not memory:
            return
        
        # Clear existing memory to avoid duplicates when loading from external history
        # This ensures we use the provided history as the source of truth
        memory.clear()
        
        # Load history into memory
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                memory.chat_memory.add_user_message(content)
            elif role == "assistant":
                memory.chat_memory.add_ai_message(content)
            elif role == "system":
                # System messages are handled separately in prompts
                pass
        
        logger.debug(f"Loaded {len(conversation_history)} messages into memory for session: {session_id}")
    
    def clear_session_memory(self, session_id: str):
        """
        Clear memory for a specific session.
        
        Args:
            session_id: Session identifier to clear
        """
        if session_id in self._memories:
            self._memories[session_id].clear()
            logger.debug(f"Cleared memory for session: {session_id}")
    
    def get_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self._memories)
    
    async def _get_agent_id(self) -> Optional[int]:
        """
        Get agent ID from database by name.
        
        Returns:
            Agent ID if found, None otherwise
        """
        if self._agent_id:
            return self._agent_id
        
        if not self.db_session:
            logger.warning("Database session not available, cannot get agent_id")
            return None
        
        try:
            stmt = select(Agent).where(Agent.name == self.AGENT_NAME)
            result = await self.db_session.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if agent:
                self._agent_id = agent.id
                logger.info(f"Found agent_id={self._agent_id} for agent '{self.AGENT_NAME}'")
                return self._agent_id
            else:
                logger.warning(f"Agent '{self.AGENT_NAME}' not found in database")
                return None
        except Exception as e:
            logger.error(f"Error getting agent_id: {e}", exc_info=True)
            return None
    
    async def _get_or_create_conversation(
        self,
        session_id: Optional[str],
        user_id: Optional[int]
    ) -> Optional["Conversation"]:
        """
        Get or create conversation for the session.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID
            
        Returns:
            Conversation instance or None
        """
        if not self.db_session or not session_id:
            return None
        
        try:
            agent_id = await self._get_agent_id()
            if not agent_id:
                return None
            
            # Get or create conversation
            conv_repo = ConversationRepository(self.db_session)
            conversation = await conv_repo.get_by_session_id(
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id
            )
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error getting or creating conversation: {e}", exc_info=True)
            return None
    
    async def _load_conversation_from_db(
        self,
        session_id: Optional[str],
        user_id: Optional[int]
    ) -> Optional[List[Dict[str, str]]]:
        """
        Load conversation history from database.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID
            
        Returns:
            List of messages in format [{"role": "...", "content": "..."}] or None
        """
        conversation = await self._get_or_create_conversation(session_id, user_id)
        if not conversation:
            return None
        
        try:
            # Load messages
            msg_repo = MessageRepository(self.db_session)
            messages = await msg_repo.get_recent_messages(
                conversation_id=conversation.id,
                limit=self._default_memory_window
            )
            
            # Convert to format expected by LangChain
            history = []
            for msg in messages:
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            logger.info(f"Loaded {len(history)} messages from database for session: {session_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error loading conversation from database: {e}", exc_info=True)
            return None
    
    async def _save_message_to_db(
        self,
        session_id: Optional[str],
        user_id: Optional[int],
        role: str,
        content: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save message to database.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID
            role: Message role (user, assistant, system)
            content: Message content
            extra_metadata: Optional metadata
            
        Returns:
            True if saved successfully, False otherwise
        """
        conversation = await self._get_or_create_conversation(session_id, user_id)
        if not conversation:
            return False
        
        try:
            # Save message
            msg_repo = MessageRepository(self.db_session)
            await msg_repo.create(
                conversation_id=conversation.id,
                role=role,
                content=content,
                extra_metadata=extra_metadata
            )
            
            # Commit the transaction
            await self.db_session.commit()
            logger.debug(f"Saved {role} message to database for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving message to database: {e}", exc_info=True)
            await self.db_session.rollback()
            return False
    
    def _detect_language(self, text: str) -> str:
        """
        Detect if the text is primarily Chinese or English.
        
        Args:
            text: Input text to detect language
            
        Returns:
            "zh" for Chinese, "en" for English
        """
        # Simple heuristic: count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len([c for c in text if c.isalnum() or '\u4e00' <= c <= '\u9fff'])
        
        if total_chars == 0:
            return "en"  # Default to English if no characters
        
        # If more than 30% are Chinese characters, consider it Chinese
        if chinese_chars / total_chars > 0.3:
            return "zh"
        return "en"
    
    async def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process companionship query from user.
        
        Args:
            user_query: User's message or question
            conversation_history: Optional conversation history (list of messages with role and content)
            session_id: Optional session identifier for maintaining conversation context
            user_id: Optional user ID for database persistence
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing companionship response
        """
        logger.info(f"Processing companionship query: {user_query[:100]}... (session: {session_id or 'no-session'}, user_id: {user_id})")
        
        # Detect language first
        language = self._detect_language(user_query)
        
        # Load conversation history from database if available
        db_history = None
        if session_id and self.db_session:
            db_history = await self._load_conversation_from_db(session_id, user_id)
        
        # Use database history if available, otherwise use provided history
        effective_history = db_history if db_history else conversation_history
        
        # Load conversation history into LangChain memory for this session (only if session_id is provided)
        self._load_conversation_history_to_memory(effective_history, session_id=session_id)
        
        # Get memory for this session (only if session_id is provided)
        memory = self._get_or_create_memory(session_id)
        
        # Save user message to database
        if session_id:
            await self._save_message_to_db(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content=user_query
            )
        
        # Build system prompt for companionship
        if language == "zh":
            system_prompt = """你是一位温暖、友善、善解人意的陪伴助手。你的主要职责是：

1. 提供情感陪伴和倾听
2. 进行日常对话和聊天
3. 在用户需要时给予鼓励和支持
4. 分享有趣的话题和想法
5. 帮助用户缓解压力和孤独感
6. 提供积极正面的情绪支持

重要原则：
- 语气要温暖、亲切、自然，就像朋友一样
- 要真诚、有同理心，能够理解用户的感受
- 保持积极正面的态度，但也要真实
- 可以分享一些轻松有趣的话题
- 当用户表达负面情绪时，要给予理解和支持
- 不要过于正式或机械，要像真正的朋友一样交流
- 可以适当使用表情符号来增加亲切感（但不要过度）
- 记住对话的上下文，让对话更连贯自然

请用中文回答，语言要自然、亲切、温暖。"""
        else:
            system_prompt = """You are a warm, friendly, and empathetic companionship assistant. Your main responsibilities are:

1. Provide emotional companionship and listening
2. Engage in daily conversation and chat
3. Offer encouragement and support when users need it
4. Share interesting topics and ideas
5. Help users relieve stress and loneliness
6. Provide positive emotional support

Important Principles:
- Tone should be warm, friendly, and natural, like a friend
- Be sincere and empathetic, able to understand users' feelings
- Maintain a positive attitude, but also be genuine
- Can share some light and interesting topics
- When users express negative emotions, provide understanding and support
- Don't be too formal or mechanical, communicate like a real friend
- Can appropriately use emojis to add warmth (but don't overdo it)
- Remember conversation context to make dialogue more coherent and natural

Please answer in English, with natural, friendly, and warm language."""
        
        # Build prompt template using LangChain
        chat_prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),  # Inject conversation history
            ("human", "{user_input}")
        ])
        
        # Get LangChain LLM
        langchain_llm = self._get_langchain_llm(api_key=api_key, base_url=base_url, max_tokens=800)
        
        # Get messages from memory for this session (only if memory exists)
        memory_messages = memory.chat_memory.messages if memory and hasattr(memory.chat_memory, 'messages') else []
        
        try:
            # Format prompt with conversation history (empty list if no memory)
            messages = chat_prompt_template.format_messages(
                chat_history=memory_messages,
                user_input=user_query
            )
            
            # Invoke LLM
            response = await langchain_llm.ainvoke(messages)
            answer = response.content.strip()
            
            logger.info(f"Generated companionship response (length: {len(answer)}, session: {session_id or 'no-session'})")
            
            # Save assistant response to database
            if session_id:
                await self._save_message_to_db(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=answer,
                    extra_metadata={
                        "model": langchain_llm.model_name if hasattr(langchain_llm, 'model_name') else None,
                        "language": language
                    }
                )
            
            return {
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Error generating companionship response: {e}", exc_info=True)
            # Return a friendly error message instead of raising exception
            if language == "zh":
                error_message = "抱歉，我遇到了一些技术问题，但我会一直在这里陪伴你。"
                if "402" in str(e) or "Insufficient credits" in str(e):
                    error_message = "抱歉，服务暂时不可用（余额不足）。请稍后再试，我会一直在这里等你。"
                elif "401" in str(e) or "Unauthorized" in str(e):
                    error_message = "抱歉，服务认证失败。请检查配置后重试。"
                else:
                    error_message = f"抱歉，遇到了一些技术问题：{str(e)[:100]}。不过没关系，我们可以稍后再聊。"
            else:
                error_message = "Sorry, I encountered some technical issues, but I'm still here for you."
                if "402" in str(e) or "Insufficient credits" in str(e):
                    error_message = "Sorry, the service is temporarily unavailable (insufficient credits). Please try again later, I'll be here waiting for you."
                elif "401" in str(e) or "Unauthorized" in str(e):
                    error_message = "Sorry, service authentication failed. Please check the configuration and try again."
                else:
                    error_message = f"Sorry, encountered some technical issues: {str(e)[:100]}. But it's okay, we can chat again later."
            
            return {
                "answer": error_message
            }

