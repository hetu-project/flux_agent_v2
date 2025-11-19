"""Bazi (Eight Characters) Agent for calculating Chinese birth chart."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import re
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class BaziAgent:
    """Bazi (Eight Characters) Agent for calculating Chinese birth chart based on lunar calendar birth date, time, and location."""
    
    def __init__(self):
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        # Keep OpenAI client for backward compatibility
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
            temperature=0.3
        )
        
        # Initialize LangChain Memory storage for different sessions
        # Each session_id will have its own memory instance
        # Keep last 10 messages for context (can be adjusted)
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}
        self._default_memory_window = 10
    
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
            "temperature": 0.3
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
    
    async def extract_info(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract birth information from natural language input.
        Uses LangChain for context management and LLM interaction.
        
        Args:
            user_query: User's natural language input
            conversation_history: Optional conversation history (list of messages with role and content)
            session_id: Optional session identifier
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing extracted information and completeness status
        """
        logger.info(f"Extracting information from query: {user_query[:100]}... (session: {session_id or 'no-session'})")
        
        # Load conversation history into LangChain memory for this session (only if session_id is provided)
        self._load_conversation_history_to_memory(conversation_history, session_id=session_id)
        
        # Get memory for this session (only if session_id is provided)
        memory = self._get_or_create_memory(session_id)
        
        # Detect language first
        language = self._detect_language(user_query)
        
        # Build prompt template using LangChain
        if language == "zh":
            system_prompt = """你是一个信息提取助手，擅长从自然语言中提取结构化信息。请严格按照要求返回JSON格式。

首先，判断用户的提问是否与八字、四柱八字、命理、生辰八字相关。如果完全不相关（比如问天气、新闻、其他话题），请将 is_bazi_related 设置为 false。

如果与八字命理相关，请仔细提取以下信息（所有时间都是阴历/农历）：
1. 出生年份（birth_year）：4位数字的年份，如1990、2000等
2. 出生月份（birth_month）：1-12的月份数字
3. 出生日期（birth_day）：1-31的日期数字
4. 出生小时（birth_hour）：0-23的小时数字
5. 出生分钟（birth_minute）：0-59的分钟数字
6. 出生地点（birth_location）：城市名称，如"北京"、"上海"、"深圳"等
7. 现在所在的地点（current_location）：城市名称，如"北京"、"上海"、"深圳"等

注意：
- 所有时间都是阴历（农历），不是阳历（公历）
- 如果用户提到"农历"、"阴历"、"旧历"，那就是阴历
- 如果用户只说了日期没有说明，默认是阴历
- 出生地点和现在所在的地点可以是同一个城市，也可以是不同的城市

请以JSON格式返回：
{{
    "is_bazi_related": true或false（判断是否与八字命理相关）,
    "birth_year": 提取的年份数字或null,
    "birth_month": 提取的月份数字或null,
    "birth_day": 提取的日期数字或null,
    "birth_hour": 提取的小时数字或null,
    "birth_minute": 提取的分钟数字或null,
    "birth_location": "提取的出生地点或null",
    "current_location": "提取的现在所在的地点或null"
}}

只返回JSON，不要其他文字。"""
            user_prompt_template = "用户输入：{user_input}"
        else:
            system_prompt = """You are an information extraction assistant skilled at extracting structured information from natural language. Please strictly return JSON format.

First, determine if the user's question is related to Bazi (Eight Characters), Four Pillars, Chinese astrology, or birth chart calculation. If it's completely unrelated (e.g., asking about weather, news, other topics), set is_bazi_related to false.

If related to Bazi/Chinese astrology, please carefully extract the following information (all times are in lunar calendar):
1. Birth year (birth_year): 4-digit year, such as 1990, 2000, etc.
2. Birth month (birth_month): Month number 1-12
3. Birth day (birth_day): Day number 1-31
4. Birth hour (birth_hour): Hour number 0-23
5. Birth minute (birth_minute): Minute number 0-59
6. Birth location (birth_location): City name, such as "Beijing", "Shanghai", "Shenzhen", etc.
7. Current location (current_location): City name, such as "Beijing", "Shanghai", "Shenzhen", etc.

Note:
- All times are in lunar calendar, not solar calendar
- If user mentions "农历", "阴历", "旧历", it's lunar calendar
- If user only mentions date without specification, default to lunar calendar
- Birth location and current location can be the same city or different cities

Please return in JSON format:
{{
    "is_bazi_related": true or false (determine if related to Bazi/Chinese astrology),
    "birth_year": extracted year number or null,
    "birth_month": extracted month number or null,
    "birth_day": extracted day number or null,
    "birth_hour": extracted hour number or null,
    "birth_minute": extracted minute number or null,
    "birth_location": "extracted birth location or null",
    "current_location": "extracted current location or null"
}}

Return only JSON, no other text."""
            user_prompt_template = "User input: {user_input}"
        
        extract_prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),  # Inject conversation history
            ("human", user_prompt_template)
        ])
        
        # Get LangChain LLM
        langchain_llm = self._get_langchain_llm(api_key=api_key, base_url=base_url)
        
        # Get messages from memory for this session (only if memory exists)
        memory_messages = memory.chat_memory.messages if memory and hasattr(memory.chat_memory, 'messages') else []
        
        try:
            # Format prompt with conversation history (empty list if no memory)
            messages = extract_prompt_template.format_messages(
                chat_history=memory_messages,
                user_input=user_query
            )
            
            # Invoke LLM
            response = await langchain_llm.ainvoke(messages)
            response_text = response.content.strip()
            logger.debug(f"Extraction response: {response_text}")
            
            # Try to parse JSON from response
            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*', '', response_text)
            response_text = re.sub(r'```\s*', '', response_text)
            response_text = response_text.strip()
            
            try:
                extracted_info = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract manually
                logger.warning("Failed to parse JSON, trying manual extraction")
                extracted_info = self._manual_extract(user_query)
            
            # Also try manual extraction as a fallback/verification
            # If LLM extraction missed something, use manual extraction to fill gaps
            manual_extracted = self._manual_extract(user_query)
            for key in ["birth_year", "birth_month", "birth_day", "birth_hour", "birth_minute", "birth_location", "current_location"]:
                if not extracted_info.get(key) and manual_extracted.get(key):
                    extracted_info[key] = manual_extracted[key]
                    logger.debug(f"Manual extraction found {key}: {manual_extracted[key]}")
            
            # Check if related to bazi (default to True if not specified)
            is_bazi_related = extracted_info.get("is_bazi_related", True)
            if not is_bazi_related:
                extracted_info["is_bazi_related"] = False
                extracted_info["language"] = language
                logger.info(f"Query is not related to bazi, language: {language}")
                return extracted_info
            
            # Check completeness
            missing = []
            required_fields = ["birth_year", "birth_month", "birth_day", "birth_hour", "birth_minute", "birth_location", "current_location"]
            for field in required_fields:
                if not extracted_info.get(field):
                    missing.append(field)
            
            extracted_info["missing_info"] = missing
            extracted_info["is_complete"] = len(missing) == 0
            extracted_info["language"] = language  # Store detected language
            extracted_info["is_bazi_related"] = True  # Ensure it's set
            
            logger.info(f"Extracted info: birth_year={extracted_info.get('birth_year')}, birth_month={extracted_info.get('birth_month')}, birth_day={extracted_info.get('birth_day')}, birth_hour={extracted_info.get('birth_hour')}, birth_minute={extracted_info.get('birth_minute')}, birth_location={extracted_info.get('birth_location')}, current_location={extracted_info.get('current_location')}, missing={missing}")
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting information: {e}", exc_info=True)
            # Try manual extraction as fallback when LLM fails
            logger.info("LLM extraction failed, trying manual extraction as fallback")
            extracted_info = self._manual_extract(user_query)
            
            # Check if related to bazi (default to True for manual extraction)
            is_bazi_related = extracted_info.get("is_bazi_related", True)
            if not is_bazi_related:
                extracted_info["is_bazi_related"] = False
                extracted_info["language"] = language
                logger.info(f"Query is not related to bazi (manual extraction), language: {language}")
                return extracted_info
            
            # Check completeness
            missing = []
            required_fields = ["birth_year", "birth_month", "birth_day", "birth_hour", "birth_minute", "birth_location", "current_location"]
            for field in required_fields:
                if not extracted_info.get(field):
                    missing.append(field)
            
            extracted_info["missing_info"] = missing
            extracted_info["is_complete"] = len(missing) == 0
            extracted_info["language"] = language  # Store detected language
            extracted_info["is_bazi_related"] = True  # Ensure it's set
            
            logger.info(f"Manual extraction result: missing={missing}, language={language}")
            
            return extracted_info
    
    def _manual_extract(self, text: str) -> Dict[str, Any]:
        """Manual extraction fallback using regex patterns."""
        result = {
            "birth_year": None,
            "birth_month": None,
            "birth_day": None,
            "birth_hour": None,
            "birth_minute": None,
            "birth_location": None,
            "current_location": None
        }
        
        # Extract birth year (4-digit number between 1900-2100)
        year_patterns = [
            r'(\d{4})年',
            r'出生[于]?(\d{4})年',
            r'\b(19\d{2}|20[0-1]\d|2100)\b',
        ]
        
        for pattern in year_patterns:
            year_match = re.search(pattern, text)
            if year_match:
                year = int(year_match.group(1))
                if 1900 <= year <= 2100:
                    result["birth_year"] = year
                    break
        
        # Extract birth month (1-12)
        month_patterns = [
            r'(\d{1,2})月',
            r'农历[的]?(\d{1,2})月',
            r'阴历[的]?(\d{1,2})月',
        ]
        
        for pattern in month_patterns:
            month_match = re.search(pattern, text)
            if month_match:
                month = int(month_match.group(1))
                if 1 <= month <= 12:
                    result["birth_month"] = month
                    break
        
        # Extract birth day (1-31)
        day_patterns = [
            r'(\d{1,2})[日号]',
            r'农历[的]?(\d{1,2})[日号]',
            r'阴历[的]?(\d{1,2})[日号]',
        ]
        
        for pattern in day_patterns:
            day_match = re.search(pattern, text)
            if day_match:
                day = int(day_match.group(1))
                if 1 <= day <= 31:
                    result["birth_day"] = day
                    break
        
        # Extract birth hour (0-23)
        hour_patterns = [
            r'(\d{1,2})[点时]',
            r'(\d{1,2}):(\d{1,2})',  # For time format like "14:30"
        ]
        
        for pattern in hour_patterns:
            hour_match = re.search(pattern, text)
            if hour_match:
                hour = int(hour_match.group(1))
                if 0 <= hour <= 23:
                    result["birth_hour"] = hour
                    # If there's a minute group, extract it
                    if len(hour_match.groups()) > 1:
                        minute = int(hour_match.group(2))
                        if 0 <= minute <= 59:
                            result["birth_minute"] = minute
                    break
        
        # Extract birth minute (0-59) if not already extracted
        if not result["birth_minute"]:
            minute_patterns = [
                r'(\d{1,2})分',
            ]
            for pattern in minute_patterns:
                minute_match = re.search(pattern, text)
                if minute_match:
                    minute = int(minute_match.group(1))
                    if 0 <= minute <= 59:
                        result["birth_minute"] = minute
                        break
        
        # Extract birth location (city names)
        location_patterns = [
            r'出生[于在]([^\s，,。.]+)',
            r'在([^\s，,。.]+)出生',
            r'出生地[：:]([^\s，,。.]+)',
        ]
        
        for pattern in location_patterns:
            location_match = re.search(pattern, text)
            if location_match:
                location = location_match.group(1).strip()
                # Remove common suffixes
                location = re.sub(r'[，,。.\s]+.*$', '', location)
                if location and len(location) <= 20:
                    result["birth_location"] = location
                    break
        
        # Extract current location (city names)
        current_location_patterns = [
            r'现在[在]([^\s，,。.]+)',
            r'目前[在]([^\s，,。.]+)',
            r'现居([^\s，,。.]+)',
            r'现在所在[：:]([^\s，,。.]+)',
        ]
        
        for pattern in current_location_patterns:
            current_location_match = re.search(pattern, text)
            if current_location_match:
                current_location = current_location_match.group(1).strip()
                # Remove common suffixes
                current_location = re.sub(r'[，,。.\s]+.*$', '', current_location)
                if current_location and len(current_location) <= 20:
                    result["current_location"] = current_location
                    break
        
        return result
    
    async def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process bazi calculation query from natural language.
        
        Args:
            user_query: User's natural language input
            conversation_history: Optional conversation history
            session_id: Optional session identifier
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing bazi calculation result or reminder message
        """
        # Extract information from the query (includes intent detection)
        extracted_info = await self.extract_info(
            user_query=user_query,
            conversation_history=conversation_history,
            session_id=session_id,
            api_key=api_key,
            base_url=base_url
        )
        
        # Get detected language
        language = extracted_info.get("language", "zh")
        
        # Check if the query is related to bazi
        is_bazi_related = extracted_info.get("is_bazi_related", True)
        if not is_bazi_related:
            if language == "zh":
                reminder = """我是一个八字（四柱八字）计算助手，专门帮助用户计算和分析八字命理。

我可以根据您的出生信息（阴历/农历）计算八字，包括：
- 四柱八字（年柱、月柱、日柱、时柱）
- 天干地支的详细说明
- 五行分析（金、木、水、火、土）
- 命理特点分析
- 性格特征
- 运势建议

请在提问中提供以下信息（所有时间都是阴历/农历）：
- 您的出生年份（例如：1990）
- 您的出生月份（1-12月）
- 您的出生日期（1-31日）
- 您的出生小时（0-23时）
- 您的出生分钟（0-59分）
- 您的出生地点（例如：北京、上海）
- 您现在所在的地点（例如：北京、上海）

例如：
"我1990年农历5月15日14时30分在北京出生，现在在上海，帮我算一下八字" """
            else:
                reminder = """I am a Bazi (Eight Characters) calculation assistant, specialized in helping users calculate and analyze Bazi Chinese astrology.

I can calculate Bazi based on your birth information (lunar calendar), including:
- Four Pillars (Year, Month, Day, Hour pillars)
- Detailed explanation of Heavenly Stems and Earthly Branches
- Five Elements analysis (Metal, Wood, Water, Fire, Earth)
- Destiny characteristics analysis
- Personality traits
- Fortune suggestions

Please include the following information in your question (all times are in lunar calendar):
- Your birth year (e.g., 1990)
- Your birth month (1-12)
- Your birth day (1-31)
- Your birth hour (0-23)
- Your birth minute (0-59)
- Your birth location (e.g., Beijing, Shanghai)
- Your current location (e.g., Beijing, Shanghai)

For example:
"I was born on May 15, 1990 at 14:30 in Beijing (lunar calendar), now in Shanghai, please calculate my Bazi" """
            
            return {
                "answer": reminder
            }
        
        # If information is incomplete, return reminder
        if not extracted_info.get("is_complete"):
            missing = extracted_info.get("missing_info", [])
            missing_chinese = {
                "birth_year": "出生年份",
                "birth_month": "出生月份",
                "birth_day": "出生日期",
                "birth_hour": "出生小时",
                "birth_minute": "出生分钟",
                "birth_location": "出生地点",
                "current_location": "现在所在的地点"
            }
            missing_list = [missing_chinese.get(m, m) for m in missing]
            
            reminder = f"为了给您计算八字，我还需要以下信息（所有时间都是阴历/农历）：{', '.join(missing_list)}。\n\n请告诉我：\n"
            if "birth_year" in missing:
                reminder += "- 您的出生年份（例如：1990）\n"
            if "birth_month" in missing:
                reminder += "- 您的出生月份（1-12月）\n"
            if "birth_day" in missing:
                reminder += "- 您的出生日期（1-31日）\n"
            if "birth_hour" in missing:
                reminder += "- 您的出生小时（0-23时）\n"
            if "birth_minute" in missing:
                reminder += "- 您的出生分钟（0-59分）\n"
            if "birth_location" in missing:
                reminder += "- 您的出生地点（例如：北京、上海）\n"
            if "current_location" in missing:
                reminder += "- 您现在所在的地点（例如：北京、上海）\n"
            
            # Format as text response (consistent with other agents)
            response_text = reminder
            if extracted_info.get("birth_year"):
                response_text = f"已获取信息：出生年份 {extracted_info.get('birth_year')}年\n\n" + response_text
            if extracted_info.get("birth_month"):
                response_text = f"已获取信息：出生月份 {extracted_info.get('birth_month')}月\n\n" + response_text
            if extracted_info.get("birth_day"):
                response_text = f"已获取信息：出生日期 {extracted_info.get('birth_day')}日\n\n" + response_text
            if extracted_info.get("birth_hour") is not None:
                response_text = f"已获取信息：出生小时 {extracted_info.get('birth_hour')}时\n\n" + response_text
            if extracted_info.get("birth_minute") is not None:
                response_text = f"已获取信息：出生分钟 {extracted_info.get('birth_minute')}分\n\n" + response_text
            if extracted_info.get("birth_location"):
                response_text = f"已获取信息：出生地点 {extracted_info.get('birth_location')}\n\n" + response_text
            if extracted_info.get("current_location"):
                response_text = f"已获取信息：现在所在的地点 {extracted_info.get('current_location')}\n\n" + response_text
            
            return {
                "answer": response_text
            }
        
        # If complete, proceed with bazi calculation
        return await self.calculate(
            birth_year=extracted_info["birth_year"],
            birth_month=extracted_info["birth_month"],
            birth_day=extracted_info["birth_day"],
            birth_hour=extracted_info["birth_hour"],
            birth_minute=extracted_info["birth_minute"],
            birth_location=extracted_info["birth_location"],
            current_location=extracted_info["current_location"],
            api_key=api_key,
            base_url=base_url
        )
    
    async def calculate(
        self,
        birth_year: int,
        birth_month: int,
        birth_day: int,
        birth_hour: int,
        birth_minute: int,
        birth_location: str,
        current_location: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate bazi (eight characters) based on user's birth information.
        
        Args:
            birth_year: Birth year (lunar calendar)
            birth_month: Birth month (lunar calendar, 1-12)
            birth_day: Birth day (lunar calendar, 1-31)
            birth_hour: Birth hour (0-23)
            birth_minute: Birth minute (0-59)
            birth_location: Birth location (city name)
            current_location: Current location (city name)
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing bazi calculation result and related information
        """
        logger.info(f"Calculating bazi for birth: {birth_year}年{birth_month}月{birth_day}日 {birth_hour}时{birth_minute}分, 出生地: {birth_location}, 现居: {current_location}")
        
        # Build prompt for bazi calculation
        prompt = f"""请为以下用户计算八字（四柱八字）：

出生信息（阴历/农历）：
- 出生年份：{birth_year}年
- 出生月份：{birth_month}月
- 出生日期：{birth_day}日
- 出生时间：{birth_hour}时{birth_minute}分
- 出生地点：{birth_location}
- 现在所在的地点：{current_location}

请根据以上信息计算并分析：
1. 四柱八字（年柱、月柱、日柱、时柱）
2. 天干地支的详细说明
3. 五行分析（金、木、水、火、土）
4. 命理特点分析
5. 性格特征
6. 运势建议

请用中文回答，语气要专业、准确，体现传统命理学的深度。注意要考虑到出生地点和现在所在的地点对时区的影响。
请控制回答长度在800字以内，重点突出，简洁明了。"""
        
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating bazi calculation (model: {model_name})")
        
        try:
            # Use LangChain for calculation
            # Limit output to ~1000 tokens (approximately 800 Chinese characters)
            langchain_llm = self._get_langchain_llm(api_key=api_key, base_url=base_url, max_tokens=1000)
            # Use moderate temperature for accurate calculations
            langchain_llm.temperature = 0.5
            
            # Build prompt template
            calculate_prompt_template = ChatPromptTemplate.from_messages([
                ("system", "你是一位经验丰富的命理师，精通八字（四柱八字）计算和分析。你能够根据阴历出生日期、时间和地点，准确计算四柱八字，并进行深入的命理分析。你的分析风格专业、准确，能够给用户提供有价值的命理指导。"),
                ("human", "{prompt}")
            ])
            
            # Format and invoke
            messages = calculate_prompt_template.format_messages(prompt=prompt)
            response = await langchain_llm.ainvoke(messages)
            
            calculation_text = response.content
            logger.info(f"Generated bazi calculation (length: {len(calculation_text)})")
            
            return {
                "answer": calculation_text
            }
        except Exception as e:
            logger.error(f"Error generating bazi calculation: {e}", exc_info=True)
            # Return a friendly error message instead of raising exception
            error_message = f"抱歉，计算八字时遇到了问题。"
            if "402" in str(e) or "Insufficient credits" in str(e):
                error_message = "抱歉，API 服务暂时不可用（余额不足）。请稍后再试或联系管理员。"
            elif "401" in str(e) or "Unauthorized" in str(e):
                error_message = "抱歉，API 认证失败。请检查 API key 是否正确。"
            else:
                error_message = f"抱歉，计算八字时遇到了技术问题：{str(e)[:100]}"
            
            return {
                "answer": error_message
            }

