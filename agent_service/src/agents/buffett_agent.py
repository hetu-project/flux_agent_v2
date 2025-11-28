"""Buffett Agent for investment advice and stock analysis based on Warren Buffett's value investing principles."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import re
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.utils.logger import get_logger
from src.agents.base_agent_with_context import BaseAgentWithContext

settings = get_settings()
logger = get_logger(__name__)


class BuffettAgent(BaseAgentWithContext):
    """Buffett Agent for investment advice and stock analysis based on Warren Buffett's value investing principles."""
    
    AGENT_NAME = "buffett"
    DEFAULT_MEMORY_WINDOW = 10
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        # Initialize base class with context support
        super().__init__(db_session=db_session)
        
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        # Keep OpenAI client for backward compatibility
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = "gemini-2.0-flash"
        
        # Initialize LangChain ChatOpenAI (default)
        self.langchain_llm = ChatOpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url,
            model="gemini-2.0-flash",
            temperature=0.3
        )
    
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
            model_name = "gemini-2.0-flash"
        else:
            # Use default AIHubMix
            client_base_url = settings.aihubmix_base_url
            model_name = "gemini-2.0-flash"
        
        llm_kwargs = {
            "api_key": api_key or settings.aihubmix_api_key,
            "base_url": client_base_url,
            "model": model_name,
            "temperature": 0.3
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        
        return ChatOpenAI(**llm_kwargs)
    
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
        Process investment advice query in the style of Warren Buffett.
        
        Args:
            user_query: User's natural language input
            conversation_history: Optional conversation history
            session_id: Optional session identifier
            user_id: Optional user ID for database persistence
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing investment advice or analysis
        """
        logger.info(f"Processing investment query: {user_query[:100]}... (session: {session_id or 'no-session'}, user_id: {user_id})")
        
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
        
        # Detect language first
        language = self._detect_language(user_query)
        
        # Build prompt template using LangChain
        if language == "zh":
            system_prompt = """你是沃伦·巴菲特（Warren Buffett）风格的投资顾问。你以价值投资理念为核心，擅长分析公司基本面、评估投资价值，并提供长期投资建议。

你的投资哲学包括：
1. 价值投资：寻找被低估的优质公司
2. 长期持有：关注企业的长期价值而非短期波动
3. 护城河分析：重视企业的竞争优势和护城河
4. 财务健康：关注企业的盈利能力、现金流和财务稳健性
5. 简单易懂：投资自己能够理解的企业和行业
6. 安全边际：以合理的价格买入优质资产

请用中文回答，语气要专业、睿智，体现巴菲特式的投资智慧。记住：不要在回复中提及任何模型名称、AI名称或生成来源，不要添加类似"以上内容由XXX生成"的说明。"""
            user_prompt_template = "用户问题：{user_input}"
        else:
            system_prompt = """You are an investment advisor in the style of Warren Buffett. You focus on value investing principles, analyzing company fundamentals, evaluating investment value, and providing long-term investment advice.

Your investment philosophy includes:
1. Value Investing: Finding undervalued quality companies
2. Long-term Holding: Focusing on long-term value rather than short-term volatility
3. Moat Analysis: Valuing competitive advantages and business moats
4. Financial Health: Focusing on profitability, cash flow, and financial stability
5. Simple and Understandable: Investing in businesses and industries you can understand
6. Margin of Safety: Buying quality assets at reasonable prices

Please answer in English, with a professional and wise tone, reflecting Buffett-style investment wisdom. Remember: Do not mention any model names, AI names, or generation sources in your response, and do not add disclaimers like "The above content was generated by XXX"."""
            user_prompt_template = "User question: {user_input}"
        
        query_prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),  # Inject conversation history
            ("human", user_prompt_template)
        ])
        
        # Get LangChain LLM
        langchain_llm = self._get_langchain_llm(api_key=api_key, base_url=base_url, max_tokens=2000)
        
        # Get messages from memory for this session (only if memory exists)
        memory_messages = memory.chat_memory.messages if memory and hasattr(memory.chat_memory, 'messages') else []
        
        try:
            # Format prompt with conversation history (empty list if no memory)
            messages = query_prompt_template.format_messages(
                chat_history=memory_messages,
                user_input=user_query
            )
            
            # Invoke LLM
            response = await langchain_llm.ainvoke(messages)
            answer = response.content.strip()
            
            logger.info(f"Generated investment advice (length: {len(answer)})")
            
            # Save assistant response to database
            if session_id:
                await self._save_message_to_db(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=answer,
                    extra_metadata={"language": language}
                )
            
            return {
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Error generating investment advice: {e}", exc_info=True)
            # Return a friendly error message instead of raising exception
            if language == "zh":
                error_message = "抱歉，处理您的投资咨询时遇到了问题。请稍后再试。"
                if "402" in str(e) or "Insufficient credits" in str(e):
                    error_message = "抱歉，API 服务暂时不可用（余额不足）。请稍后再试或联系管理员。"
                elif "401" in str(e) or "Unauthorized" in str(e):
                    error_message = "抱歉，API 认证失败。请检查 API key 是否正确。"
            else:
                error_message = "Sorry, I encountered a problem while processing your investment inquiry. Please try again later."
                if "402" in str(e) or "Insufficient credits" in str(e):
                    error_message = "Sorry, the API service is temporarily unavailable (insufficient credits). Please try again later or contact the administrator."
                elif "401" in str(e) or "Unauthorized" in str(e):
                    error_message = "Sorry, API authentication failed. Please check if the API key is correct."
            
            # Save error response to database
            if session_id:
                await self._save_message_to_db(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=error_message,
                    extra_metadata={"language": language, "error": str(e)}
                )
            
            return {
                "answer": error_message
            }

