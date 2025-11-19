"""Fortune Telling Agent for predicting tomorrow's fortune."""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
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


class FortuneAgent:
    """Fortune Telling Agent for predicting tomorrow's fortune based on name, birth year, and zodiac sign."""
    
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
        Extract name, birth year, and zodiac sign from natural language input.
        Uses LangChain for context management and LLM interaction.
        
        Args:
            user_query: User's natural language input
            conversation_history: Optional conversation history (list of messages with role and content)
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing extracted information and completeness status
        """
        logger.info(f"Extracting information from query: {user_query[:100]}... (session: {session_id or 'no-session'})")
        
        # Detect language from user query
        language = self._detect_language(user_query)
        
        # Load conversation history into LangChain memory for this session (only if session_id is provided)
        self._load_conversation_history_to_memory(conversation_history, session_id=session_id)
        
        # Get memory for this session (only if session_id is provided)
        memory = self._get_or_create_memory(session_id)
        
        # Build bilingual prompt template
        if language == "zh":
            system_prompt = """你是一个信息提取助手，擅长从自然语言中提取结构化信息。请严格按照要求返回JSON格式。

首先，判断用户的提问是否与占卜、运势预测、命理相关。如果完全不相关（比如问天气、新闻、其他话题），请将 is_fortune_related 设置为 false。

如果与占卜预测相关，请仔细提取以下信息：

必需信息（基本信息）：
1. 姓名（name）：用户的名字，可能出现在"我叫"、"我是"、"姓名叫"、"名字是"等表达中
2. 出生年份（birth_year）：4位数字的年份，如1990、2000等，可能出现在"出生"、"年出生"等表达中
3. 星座（zodiac_sign）：十二星座之一，可能是中文（白羊座、金牛座等）或英文（Aries、Taurus等）

可选信息（预测内容）：
4. 运势类型（prediction_type）：用户想要预测的运势类型，如"整体运势"、"桃花运"、"事业运"、"财运"、"健康运"、"爱情运"等。如果用户没有明确指定，默认为"整体运势"
5. 时间范围（time_range）：用户想要预测的时间范围，如"明天"、"下周"、"下个月"、"本周"、"本月"、"今年"等。如果用户没有明确指定，默认为"明天"

星座对照表：
- 白羊座/Aries
- 金牛座/Taurus
- 双子座/Gemini
- 巨蟹座/Cancer
- 狮子座/Leo
- 处女座/Virgo
- 天秤座/Libra
- 天蝎座/Scorpio
- 射手座/Sagittarius
- 摩羯座/Capricorn
- 水瓶座/Aquarius
- 双鱼座/Pisces

请以JSON格式返回：
{{
    "is_fortune_related": true或false（判断是否与占卜预测相关）,
    "name": "提取的姓名或null",
    "birth_year": 提取的年份数字或null,
    "zodiac_sign": "提取的星座（英文）或null",
    "prediction_type": "提取的运势类型或null（如果为null，表示整体运势）",
    "time_range": "提取的时间范围或null（如果为null，表示明天）"
}}

只返回JSON，不要其他文字。"""
            user_prompt_template = "用户输入：{user_input}"
        else:
            system_prompt = """You are an information extraction assistant skilled at extracting structured information from natural language. Please strictly return JSON format.

First, determine if the user's question is related to fortune telling, prediction, or astrology. If it's completely unrelated (e.g., asking about weather, news, other topics), set is_fortune_related to false.

If related to fortune telling/prediction, please carefully extract the following information:

Required information (basic info):
1. Name (name): User's name, may appear in expressions like "I'm", "My name is", "I am called", etc.
2. Birth year (birth_year): 4-digit year, such as 1990, 2000, etc., may appear in expressions like "born in", "born", etc.
3. Zodiac sign (zodiac_sign): One of the twelve zodiac signs, may be in Chinese (白羊座, 金牛座, etc.) or English (Aries, Taurus, etc.)

Optional information (prediction content):
4. Prediction type (prediction_type): The type of fortune the user wants to predict, such as "overall fortune", "love fortune", "career fortune", "wealth fortune", "health fortune", "romance fortune", etc. If not specified, default to "overall fortune"
5. Time range (time_range): The time range the user wants to predict, such as "tomorrow", "next week", "next month", "this week", "this month", "this year", etc. If not specified, default to "tomorrow"

Zodiac sign reference:
- 白羊座/Aries
- 金牛座/Taurus
- 双子座/Gemini
- 巨蟹座/Cancer
- 狮子座/Leo
- 处女座/Virgo
- 天秤座/Libra
- 天蝎座/Scorpio
- 射手座/Sagittarius
- 摩羯座/Capricorn
- 水瓶座/Aquarius
- 双鱼座/Pisces

Please return in JSON format:
{{
    "is_fortune_related": true or false (determine if related to fortune telling/prediction),
    "name": "extracted name or null",
    "birth_year": extracted year number or null,
    "zodiac_sign": "extracted zodiac sign (in English) or null",
    "prediction_type": "extracted prediction type or null (if null, means overall fortune)",
    "time_range": "extracted time range or null (if null, means tomorrow)"
}}

Return only JSON, no other text."""
            user_prompt_template = "User input: {user_input}"
        
        # Build prompt template using LangChain
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
            if not extracted_info.get("name") and manual_extracted.get("name"):
                extracted_info["name"] = manual_extracted["name"]
                logger.debug(f"Manual extraction found name: {manual_extracted['name']}")
            if not extracted_info.get("birth_year") and manual_extracted.get("birth_year"):
                extracted_info["birth_year"] = manual_extracted["birth_year"]
                logger.debug(f"Manual extraction found birth_year: {manual_extracted['birth_year']}")
            if not extracted_info.get("zodiac_sign") and manual_extracted.get("zodiac_sign"):
                extracted_info["zodiac_sign"] = manual_extracted["zodiac_sign"]
                logger.debug(f"Manual extraction found zodiac_sign: {manual_extracted['zodiac_sign']}")
            
            # Normalize zodiac sign to English
            if extracted_info.get("zodiac_sign"):
                extracted_info["zodiac_sign"] = self._normalize_zodiac_sign(extracted_info["zodiac_sign"])
            
            # Check if related to fortune telling (default to True if not specified)
            is_fortune_related = extracted_info.get("is_fortune_related", True)
            if not is_fortune_related:
                extracted_info["is_fortune_related"] = False
                extracted_info["language"] = language
                logger.info(f"Query is not related to fortune telling, language: {language}")
                return extracted_info
            
            # Set defaults for optional fields
            if not extracted_info.get("prediction_type"):
                extracted_info["prediction_type"] = None  # Will default to "整体运势" or "overall fortune"
            if not extracted_info.get("time_range"):
                extracted_info["time_range"] = None  # Will default to "明天" or "tomorrow"
            
            # Check completeness (only required fields)
            missing = []
            if not extracted_info.get("name"):
                missing.append("name")
            if not extracted_info.get("birth_year"):
                missing.append("birth_year")
            if not extracted_info.get("zodiac_sign"):
                missing.append("zodiac_sign")
            
            extracted_info["missing_info"] = missing
            extracted_info["is_complete"] = len(missing) == 0
            extracted_info["language"] = language  # Store detected language
            extracted_info["is_fortune_related"] = True  # Ensure it's set
            
            logger.info(f"Extracted info: name={extracted_info.get('name')}, birth_year={extracted_info.get('birth_year')}, zodiac_sign={extracted_info.get('zodiac_sign')}, prediction_type={extracted_info.get('prediction_type')}, time_range={extracted_info.get('time_range')}, missing={missing}, language={language}")
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting information: {e}", exc_info=True)
            # Try manual extraction as fallback when LLM fails
            logger.info("LLM extraction failed, trying manual extraction as fallback")
            extracted_info = self._manual_extract(user_query)
            
            # Detect language for fallback
            language = self._detect_language(user_query)
            
            # Normalize zodiac sign to English
            if extracted_info.get("zodiac_sign"):
                extracted_info["zodiac_sign"] = self._normalize_zodiac_sign(extracted_info["zodiac_sign"])
            
            # Check if related to fortune telling (default to True for manual extraction)
            is_fortune_related = extracted_info.get("is_fortune_related", True)
            if not is_fortune_related:
                extracted_info["is_fortune_related"] = False
                extracted_info["language"] = language
                logger.info(f"Query is not related to fortune telling (manual extraction), language: {language}")
                return extracted_info
            
            # Set defaults for optional fields
            if not extracted_info.get("prediction_type"):
                extracted_info["prediction_type"] = None
            if not extracted_info.get("time_range"):
                extracted_info["time_range"] = None
            
            # Check completeness (only required fields)
            missing = []
            if not extracted_info.get("name"):
                missing.append("name")
            if not extracted_info.get("birth_year"):
                missing.append("birth_year")
            if not extracted_info.get("zodiac_sign"):
                missing.append("zodiac_sign")
            
            extracted_info["missing_info"] = missing
            extracted_info["is_complete"] = len(missing) == 0
            extracted_info["language"] = language  # Store detected language
            extracted_info["is_fortune_related"] = True  # Ensure it's set
            
            logger.info(f"Manual extraction result: name={extracted_info.get('name')}, birth_year={extracted_info.get('birth_year')}, zodiac_sign={extracted_info.get('zodiac_sign')}, prediction_type={extracted_info.get('prediction_type')}, time_range={extracted_info.get('time_range')}, missing={missing}, language={language}")
            
            return extracted_info
    
    def _manual_extract(self, text: str) -> Dict[str, Any]:
        """Manual extraction fallback using regex patterns."""
        result = {
            "name": None,
            "birth_year": None,
            "zodiac_sign": None
        }
        
        # Extract birth year (4-digit number between 1900-2100)
        # Look for patterns like "1990年" or "1990年出生" or just "1990"
        year_patterns = [
            r'(\d{4})年出生',
            r'出生[于]?(\d{4})年',
            r'(\d{4})年',
            r'\b(19\d{2}|20[0-1]\d|2100)\b',
        ]
        
        for pattern in year_patterns:
            year_match = re.search(pattern, text)
            if year_match:
                year = int(year_match.group(1))
                if 1900 <= year <= 2100:
                    result["birth_year"] = year
                    break
        
        # Extract zodiac signs (Chinese and English)
        zodiac_patterns = {
            r'白羊座|Aries': 'Aries',
            r'金牛座|Taurus': 'Taurus',
            r'双子座|Gemini': 'Gemini',
            r'巨蟹座|Cancer': 'Cancer',
            r'狮子座|Leo': 'Leo',
            r'处女座|Virgo': 'Virgo',
            r'天秤座|Libra': 'Libra',
            r'天蝎座|Scorpio': 'Scorpio',
            r'射手座|Sagittarius': 'Sagittarius',
            r'摩羯座|Capricorn': 'Capricorn',
            r'水瓶座|Aquarius': 'Aquarius',
            r'双鱼座|Pisces': 'Pisces',
        }
        
        for pattern, zodiac in zodiac_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                result["zodiac_sign"] = zodiac
                break
        
        # Try to extract name (improved patterns)
        name_patterns = [
            r'我叫([^\s，,。.]+)',
            r'我是([^\s，,。.]+)',
            r'我姓名叫([^\s，,。.]+)',  # Match "我姓名叫"
            r'姓名叫([^\s，,。.]+)',
            r'名字[是为]([^\s，,。.]+)',
            r'姓名[是为]([^\s，,。.]+)',
            r'我的姓名叫([^\s，,。.]+)',
            r'我的名字叫([^\s，,。.]+)',
            r'我的名字是([^\s，,。.]+)',
            r'我的姓名是([^\s，,。.]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                # Remove common suffixes that might be captured
                name = re.sub(r'[，,。.\s]+.*$', '', name)
                if 1 <= len(name) <= 10:  # Reasonable name length
                    result["name"] = name
                    break
        
        return result
    
    def _normalize_zodiac_sign(self, zodiac: str) -> str:
        """Normalize zodiac sign to English."""
        zodiac_map = {
            '白羊座': 'Aries', 'Aries': 'Aries',
            '金牛座': 'Taurus', 'Taurus': 'Taurus',
            '双子座': 'Gemini', 'Gemini': 'Gemini',
            '巨蟹座': 'Cancer', 'Cancer': 'Cancer',
            '狮子座': 'Leo', 'Leo': 'Leo',
            '处女座': 'Virgo', 'Virgo': 'Virgo',
            '天秤座': 'Libra', 'Libra': 'Libra',
            '天蝎座': 'Scorpio', 'Scorpio': 'Scorpio',
            '射手座': 'Sagittarius', 'Sagittarius': 'Sagittarius',
            '摩羯座': 'Capricorn', 'Capricorn': 'Capricorn',
            '水瓶座': 'Aquarius', 'Aquarius': 'Aquarius',
            '双鱼座': 'Pisces', 'Pisces': 'Pisces',
        }
        
        zodiac_clean = zodiac.strip()
        return zodiac_map.get(zodiac_clean, zodiac_clean)
    
    async def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process fortune telling query from natural language.
        
        Args:
            user_query: User's natural language input
            conversation_history: Optional conversation history
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing fortune prediction or reminder message
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
        
        # Check if the query is related to fortune telling
        is_fortune_related = extracted_info.get("is_fortune_related", True)
        if not is_fortune_related:
            if language == "zh":
                reminder = """我是一个占卜运势预测助手，专门帮助用户进行运势预测和占卜。

我可以根据您的姓名、出生年份和星座，为您预测：
- 整体运势、桃花运、事业运、财运、健康运、爱情运等
- 明天、下周、下个月、本周、本月、今年等不同时间范围的运势

请在提问中提供以下基本信息：
- 您的姓名
- 您的出生年份（例如：1990）
- 您的星座（例如：白羊座、Aries等）

您也可以同时说明想要预测的运势类型和时间范围，例如：
"我叫张三，1990年出生，白羊座，帮我算一下下周的桃花运"
"我是李四，2000年出生，天秤座，下个月的事业运如何？" """
            else:
                reminder = """I am a fortune telling and prediction assistant, specialized in helping users with fortune predictions and divination.

I can predict for you based on your name, birth year, and zodiac sign:
- Overall fortune, love fortune, career fortune, wealth fortune, health fortune, romance fortune, etc.
- Fortune for different time ranges: tomorrow, next week, next month, this week, this month, this year, etc.

Please include the following basic information in your question:
- Your name
- Your birth year (e.g., 1990)
- Your zodiac sign (e.g., Aries, 白羊座, etc.)

You can also specify the type of fortune and time range you want to predict, for example:
"My name is John, born in 1990, Aries, what's my love fortune for next week?"
"I'm Mary, born in 2000, Libra, how's my career fortune for next month?" """
            
            return {
                "answer": reminder
            }
        
        # If information is incomplete, return reminder
        if not extracted_info.get("is_complete"):
            missing = extracted_info.get("missing_info", [])
            
            if language == "zh":
                missing_chinese = {
                    "name": "姓名",
                    "birth_year": "出生年份",
                    "zodiac_sign": "星座"
                }
                missing_list = [missing_chinese.get(m, m) for m in missing]
                
                reminder = f"为了给您进行运势预测，请在提问中提供以下信息：{', '.join(missing_list)}。\n\n请告诉我：\n"
                if "name" in missing:
                    reminder += "- 您的姓名\n"
                if "birth_year" in missing:
                    reminder += "- 您的出生年份（例如：1990）\n"
                if "zodiac_sign" in missing:
                    reminder += "- 您的星座（例如：白羊座、Aries等）\n"
                
                reminder += "\n您可以在提问中同时说明想要预测的运势类型（如：整体运势、桃花运、事业运、财运等）和时间范围（如：明天、下周、下个月等）。"
                
                # Format as text response (consistent with other agents)
                response_text = reminder
                if extracted_info.get("name"):
                    response_text = f"已获取信息：姓名 {extracted_info.get('name')}\n\n" + response_text
                if extracted_info.get("birth_year"):
                    response_text = f"已获取信息：出生年份 {extracted_info.get('birth_year')}\n\n" + response_text
                if extracted_info.get("zodiac_sign"):
                    response_text = f"已获取信息：星座 {extracted_info.get('zodiac_sign')}\n\n" + response_text
            else:
                missing_english = {
                    "name": "name",
                    "birth_year": "birth year",
                    "zodiac_sign": "zodiac sign"
                }
                missing_list = [missing_english.get(m, m) for m in missing]
                
                reminder = f"To provide you with a fortune prediction, please include the following information in your question: {', '.join(missing_list)}.\n\nPlease tell me:\n"
                if "name" in missing:
                    reminder += "- Your name\n"
                if "birth_year" in missing:
                    reminder += "- Your birth year (e.g., 1990)\n"
                if "zodiac_sign" in missing:
                    reminder += "- Your zodiac sign (e.g., Aries, 白羊座, etc.)\n"
                
                reminder += "\nYou can also specify the type of fortune you want to predict (e.g., overall fortune, love fortune, career fortune, wealth fortune, etc.) and the time range (e.g., tomorrow, next week, next month, etc.) in your question."
                
                # Format as text response (consistent with other agents)
                response_text = reminder
                if extracted_info.get("name"):
                    response_text = f"Information received: Name {extracted_info.get('name')}\n\n" + response_text
                if extracted_info.get("birth_year"):
                    response_text = f"Information received: Birth year {extracted_info.get('birth_year')}\n\n" + response_text
                if extracted_info.get("zodiac_sign"):
                    response_text = f"Information received: Zodiac sign {extracted_info.get('zodiac_sign')}\n\n" + response_text
            
            return {
                "answer": response_text
            }
        
        # If complete, proceed with prediction
        return await self.predict(
            name=extracted_info["name"],
            birth_year=extracted_info["birth_year"],
            zodiac_sign=extracted_info["zodiac_sign"],
            prediction_type=extracted_info.get("prediction_type"),
            time_range=extracted_info.get("time_range"),
            language=language,
            api_key=api_key,
            base_url=base_url
        )
    
    async def predict(
        self,
        name: str,
        birth_year: int,
        zodiac_sign: str,
        prediction_type: Optional[str] = None,
        time_range: Optional[str] = None,
        language: str = "zh",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict fortune based on user's information and query.
        
        Args:
            name: User's name
            birth_year: User's birth year
            zodiac_sign: User's zodiac sign
            prediction_type: Type of fortune to predict (e.g., "整体运势", "桃花运", "事业运", etc.), defaults to overall fortune
            time_range: Time range for prediction (e.g., "明天", "下周", "下个月", etc.), defaults to tomorrow
            language: Language for response ("zh" for Chinese, "en" for English)
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing fortune prediction and related information
        """
        # Set defaults
        if not prediction_type:
            prediction_type = "整体运势" if language == "zh" else "overall fortune"
        if not time_range:
            time_range = "明天" if language == "zh" else "tomorrow"
        
        logger.info(f"Predicting fortune for {name} (born {birth_year}, {zodiac_sign}, type: {prediction_type}, time: {time_range}, language: {language})")
        
        # Calculate age
        current_year = datetime.now().year
        age = current_year - birth_year
        
        # Build prompt for fortune telling based on language
        if language == "zh":
            # Format date based on time range
            if time_range == "明天" or "明天" in time_range:
                target_date = datetime.now() + timedelta(days=1)
                date_str = target_date.strftime("%Y年%m月%d日")
            elif "下周" in time_range or "下个星期" in time_range:
                # Next week (Monday)
                days_until_monday = (7 - datetime.now().weekday()) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                target_date = datetime.now() + timedelta(days=days_until_monday)
                date_str = f"{target_date.strftime('%Y年%m月%d日')}起的一周"
            elif "下个月" in time_range or "下个月份" in time_range:
                # Next month
                if datetime.now().month == 12:
                    date_str = f"{datetime.now().year + 1}年1月"
                else:
                    date_str = f"{datetime.now().year}年{datetime.now().month + 1}月"
            elif "本周" in time_range or "这个星期" in time_range:
                date_str = "本周"
            elif "本月" in time_range or "这个月" in time_range:
                date_str = f"{datetime.now().year}年{datetime.now().month}月"
            elif "今年" in time_range:
                date_str = f"{datetime.now().year}年"
            else:
                # Default to tomorrow
                target_date = datetime.now() + timedelta(days=1)
                date_str = target_date.strftime("%Y年%m月%d日")
            
            prompt = f"""请为以下用户预测{time_range}的{prediction_type}：

姓名：{name}
出生年份：{birth_year}年（今年{age}岁）
星座：{zodiac_sign}
预测时间：{date_str}
运势类型：{prediction_type}

请根据用户的姓名、年龄和星座，预测{time_range}的{prediction_type}。请提供：
1. {prediction_type}的详细预测
2. 幸运数字（3-5个数字）
3. 幸运颜色
4. 相关建议

请用中文回答，语气要友好、积极，但也要保持一定的神秘感。预测要具体但不过于绝对。
请控制回答长度在800字以内，简洁明了。"""
            system_prompt = "你是一位经验丰富的占星师和命理师，擅长根据姓名、出生年份和星座预测各种运势。你的预测风格既神秘又积极，能够给用户带来希望和指导。"
        else:
            # Format date based on time range
            if time_range == "tomorrow" or "tomorrow" in time_range.lower():
                target_date = datetime.now() + timedelta(days=1)
                date_str = target_date.strftime("%B %d, %Y")
            elif "next week" in time_range.lower():
                days_until_monday = (7 - datetime.now().weekday()) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                target_date = datetime.now() + timedelta(days=days_until_monday)
                date_str = f"the week starting from {target_date.strftime('%B %d, %Y')}"
            elif "next month" in time_range.lower():
                if datetime.now().month == 12:
                    date_str = f"January {datetime.now().year + 1}"
                else:
                    next_month = datetime.now().month + 1
                    date_str = datetime(datetime.now().year, next_month, 1).strftime("%B %Y")
            elif "this week" in time_range.lower():
                date_str = "this week"
            elif "this month" in time_range.lower():
                date_str = datetime.now().strftime("%B %Y")
            elif "this year" in time_range.lower():
                date_str = str(datetime.now().year)
            else:
                # Default to tomorrow
                target_date = datetime.now() + timedelta(days=1)
                date_str = target_date.strftime("%B %d, %Y")
            
            prompt = f"""Please predict the {prediction_type} for {time_range} for the following user:

Name: {name}
Birth Year: {birth_year} (Age: {age} this year)
Zodiac Sign: {zodiac_sign}
Prediction Time: {date_str}
Fortune Type: {prediction_type}

Please predict the {prediction_type} for {time_range} based on the user's name, age, and zodiac sign. Please provide:
1. Detailed prediction of {prediction_type}
2. Lucky numbers (3-5 numbers)
3. Lucky color
4. Related advice

Please answer in English, with a friendly and positive tone, but also maintain a certain sense of mystery. Predictions should be specific but not too absolute.
Please keep the answer within 800 words, concise and clear."""
            system_prompt = "You are an experienced astrologer and fortune teller, skilled at predicting various types of fortune based on name, birth year, and zodiac sign. Your prediction style is both mysterious and positive, able to bring hope and guidance to users."
        
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating fortune prediction (model: {model_name}, language: {language})")
        
        try:
            # Use LangChain for prediction
            # Limit output to ~1000 tokens (approximately 800 Chinese characters or 600 English words)
            langchain_llm = self._get_langchain_llm(api_key=api_key, base_url=base_url, max_tokens=1000)
            # Use higher temperature for creative predictions
            langchain_llm.temperature = 0.8
            
            # Build prompt template
            predict_prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{prompt}")
            ])
            
            # Format and invoke
            messages = predict_prompt_template.format_messages(prompt=prompt)
            response = await langchain_llm.ainvoke(messages)
            
            prediction_text = response.content
            logger.info(f"Generated fortune prediction (length: {len(prediction_text)})")
            
            # Try to extract lucky numbers and color from the prediction
            lucky_numbers = self._extract_lucky_numbers(prediction_text)
            lucky_color = self._extract_lucky_color(prediction_text)
            advice = self._extract_advice(prediction_text)
            
            # Format response as text (consistent with other agents)
            # The prediction_text already contains all the information, but we can enhance it
            response_text = prediction_text
            
            # Add extracted structured info if available and not already in text
            if language == "zh":
                if lucky_numbers and "幸运数字" not in prediction_text:
                    response_text += f"\n\n幸运数字：{', '.join(map(str, lucky_numbers))}"
                if lucky_color and "幸运颜色" not in prediction_text:
                    response_text += f"\n幸运颜色：{lucky_color}"
                if advice and "建议" not in prediction_text:
                    response_text += f"\n建议：{advice}"
            else:
                if lucky_numbers and "lucky number" not in prediction_text.lower():
                    response_text += f"\n\nLucky Numbers: {', '.join(map(str, lucky_numbers))}"
                if lucky_color and "lucky color" not in prediction_text.lower():
                    response_text += f"\nLucky Color: {lucky_color}"
                if advice and "advice" not in prediction_text.lower():
                    response_text += f"\nAdvice: {advice}"
            
            return {
                "answer": response_text
            }
        except Exception as e:
            logger.error(f"Error generating fortune prediction: {e}", exc_info=True)
            # Return a friendly error message instead of raising exception
            error_message = f"抱歉，生成运势预测时遇到了问题。"
            if "402" in str(e) or "Insufficient credits" in str(e):
                error_message = "抱歉，API 服务暂时不可用（余额不足）。请稍后再试或联系管理员。"
            elif "401" in str(e) or "Unauthorized" in str(e):
                error_message = "抱歉，API 认证失败。请检查 API key 是否正确。"
            else:
                error_message = f"抱歉，生成运势预测时遇到了技术问题：{str(e)[:100]}"
            
            return {
                "answer": error_message
            }
    
    def _extract_lucky_numbers(self, text: str) -> Optional[list[int]]:
        """Extract lucky numbers from prediction text."""
        import re
        # Look for patterns like "幸运数字：1, 2, 3" or "幸运数字是 5 7 9"
        patterns = [
            r'幸运数字[：:]\s*([0-9,\s]+)',
            r'幸运数字[是为]\s*([0-9,\s]+)',
            r'数字[：:]\s*([0-9,\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                numbers_str = match.group(1)
                # Extract all numbers
                numbers = re.findall(r'\d+', numbers_str)
                if numbers:
                    return [int(n) for n in numbers[:5]]  # Return up to 5 numbers
        
        return None
    
    def _extract_lucky_color(self, text: str) -> Optional[str]:
        """Extract lucky color from prediction text."""
        import re
        # Look for patterns like "幸运颜色：红色" or "幸运颜色是蓝色"
        patterns = [
            r'幸运颜色[：:]\s*([^\n，,。.]+)',
            r'幸运颜色[是为]\s*([^\n，,。.]+)',
            r'颜色[：:]\s*([^\n，,。.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                color = match.group(1).strip()
                # Remove common suffixes
                color = re.sub(r'[，,。.\s]+.*$', '', color)
                if color:
                    return color
        
        return None
    
    def _extract_advice(self, text: str) -> Optional[str]:
        """Extract advice from prediction text."""
        import re
        # Look for patterns like "建议：" or "明日建议："
        patterns = [
            r'[明日]?建议[：:]\s*([^\n]+(?:\n[^\n]+)*)',
            r'建议[：:]\s*([^\n]+(?:\n[^\n]+)*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                advice = match.group(1).strip()
                # Limit length
                if len(advice) > 200:
                    advice = advice[:200] + "..."
                return advice
        
        return None

