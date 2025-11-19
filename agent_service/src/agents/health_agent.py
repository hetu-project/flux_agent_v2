"""Health Agent for health-related consultations and advice."""

from typing import Dict, Any, Optional, List
from openai import OpenAI
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HealthAgent:
    """Health Agent for providing health-related consultations and advice."""
    
    def __init__(self):
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = settings.chat_model
    
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
    
    async def _detect_health_intent(
        self,
        user_query: str,
        language: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> bool:
        """
        Detect if the user's query is related to health.
        
        Args:
            user_query: User's query text
            language: Detected language ("zh" or "en")
            api_key: Optional API key
            base_url: Optional base URL
            
        Returns:
            True if the query is related to health, False otherwise
        """
        # First, try keyword-based detection (fast and reliable)
        text_lower = user_query.lower()
        
        # Chinese keywords
        chinese_keywords = [
            "健康", "身体", "疾病", "症状", "治疗", "药物", "营养", "运动", "减肥", "健身",
            "头痛", "发烧", "咳嗽", "疼痛", "失眠", "焦虑", "抑郁", "心理", "饮食", "睡眠",
            "医生", "医院", "检查", "诊断", "预防", "保健", "养生", "康复", "病", "疼"
        ]
        
        # English keywords
        english_keywords = [
            "health", "body", "disease", "symptom", "treatment", "medicine", "drug", "nutrition", "exercise", "fitness",
            "headache", "fever", "cough", "pain", "insomnia", "anxiety", "depression", "mental", "diet", "sleep",
            "doctor", "hospital", "check", "diagnosis", "prevention", "wellness", "recovery", "illness", "sick", "hurt"
        ]
        
        # Check for keywords
        if language == "zh":
            for keyword in chinese_keywords:
                if keyword in user_query:
                    logger.debug(f"Detected health-related intent via keyword: {keyword}")
                    return True
        else:
            for keyword in english_keywords:
                if keyword in text_lower:
                    logger.debug(f"Detected health-related intent via keyword: {keyword}")
                    return True
        
        # If no keywords found, use LLM for more nuanced detection
        try:
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            model_name = settings.openrouter_model if api_key else self.chat_model
            
            if language == "zh":
                intent_prompt = f"""判断以下用户提问是否与健康、医疗、身体、疾病相关。

用户提问：{user_query}

请只回答"是"或"否"，不要其他文字。"""
                system_prompt = "你是一个意图分析助手，判断用户提问是否与健康医疗相关。只回答'是'或'否'。"
            else:
                intent_prompt = f"""Determine if the following user question is related to health, medical, body, or illness.

User question: {user_query}

Please answer only "yes" or "no", no other text."""
                system_prompt = "You are an intent analysis assistant that determines if user questions are related to health or medical topics. Answer only 'yes' or 'no'."
            
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().lower()
            is_related = "是" in answer or "yes" in answer or "true" in answer
            
            logger.debug(f"LLM health intent detection result: {is_related} (response: {answer})")
            return is_related
            
        except Exception as e:
            logger.warning(f"Error in LLM health intent detection: {e}, defaulting to True")
            # If LLM fails, default to True to avoid blocking legitimate queries
            return True
    
    async def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process health-related query from user.
        
        Args:
            user_query: User's health-related question or concern
            conversation_history: Optional conversation history (list of messages with role and content)
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing health advice or consultation response
        """
        logger.info(f"Processing health query: {user_query[:100]}...")
        
        # Detect language first
        language = self._detect_language(user_query)
        
        # Check if the query is related to health
        is_health_related = await self._detect_health_intent(
            user_query=user_query,
            language=language,
            api_key=api_key,
            base_url=base_url
        )
        
        # If not related to health, return a reminder
        if not is_health_related:
            if language == "zh":
                reminder = """我是一个健康咨询助手，专门帮助用户解答健康相关的问题。

我可以为您提供：
- 健康生活建议（营养、运动、睡眠等）
- 常见疾病的预防知识
- 心理健康支持和建议
- 一般健康问题的咨询

重要提示：
- 我的建议仅供参考，不能替代专业医疗诊断
- 对于严重或紧急的健康问题，请立即就医
- 我不会提供具体的药物建议（除非是常见的非处方药）

请在提问中描述您的健康问题或咨询需求，例如：
"我最近总是感觉很累，有什么建议吗？"
"如何预防感冒？"
"失眠怎么办？" """
            else:
                reminder = """I am a health consultation assistant, specialized in helping users with health-related questions.

I can provide:
- Health lifestyle advice (nutrition, exercise, sleep, etc.)
- Prevention knowledge for common diseases
- Mental health support and advice
- General health consultation

Important Notice:
- My advice is for reference only and cannot replace professional medical diagnosis
- For serious or urgent health issues, please seek medical attention immediately
- I will not provide specific medication advice (except for common over-the-counter medications)

Please describe your health question or consultation need in your question, for example:
"I've been feeling very tired lately, any suggestions?"
"How to prevent colds?"
"What to do about insomnia?" """
            
            return {
                "answer": reminder
            }
        
        # Build context from conversation history if available
        context = ""
        if conversation_history:
            # Include recent messages for context
            recent_messages = conversation_history[-10:]  # Last 10 messages
            context = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_messages])
        
        # Build system prompt for health consultation
        system_prompt = """你是一位专业、友善的健康咨询助手。你的职责是：

1. 提供健康相关的建议和信息
2. 回答关于营养、运动、心理健康、常见疾病预防等方面的问题
3. 给出实用的健康生活建议
4. 提醒用户注意健康风险，但不要进行医疗诊断
5. 对于严重症状，建议用户咨询专业医生

重要原则：
- 始终强调：你的建议仅供参考，不能替代专业医疗诊断
- 对于严重或紧急的健康问题，必须建议用户立即就医
- 提供科学、准确的信息
- 语气要温和、专业、鼓励
- 避免给出具体的药物建议（除非是常见的非处方药）
- 尊重用户的隐私和感受

请用中文回答，语言要清晰易懂。"""
        
        # Build user prompt
        history_section = f'对话历史：\n{context}' if context else ''
        user_prompt = f"""用户问题：{user_query}

{history_section}

请根据用户的问题，提供专业、友善的健康建议。记住：
- 如果问题涉及严重症状或紧急情况，必须建议立即就医
- 提供实用的建议和信息
- 语气要温和、鼓励
- 强调你的建议仅供参考，不能替代专业医疗诊断"""
        
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating health consultation (model: {model_name})")
        
        try:
            # Get LLM client
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.7,  # Balanced temperature for professional yet friendly responses
                max_tokens=1000  # Allow sufficient tokens for detailed advice
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated health consultation response (length: {len(answer)})")
            
            return {
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Error generating health consultation: {e}", exc_info=True)
            # Return a friendly error message instead of raising exception
            error_message = "抱歉，生成健康建议时遇到了问题。"
            if "402" in str(e) or "Insufficient credits" in str(e):
                error_message = "抱歉，API 服务暂时不可用（余额不足）。请稍后再试或联系管理员。"
            elif "401" in str(e) or "Unauthorized" in str(e):
                error_message = "抱歉，API 认证失败。请检查 API key 是否正确。"
            else:
                error_message = f"抱歉，生成健康建议时遇到了技术问题：{str(e)[:100]}"
            
            return {
                "answer": error_message
            }

