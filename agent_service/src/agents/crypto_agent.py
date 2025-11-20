"""Crypto Agent for cryptocurrency-related consultations and advice."""

from typing import Dict, Any, Optional, List
from openai import OpenAI
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class CryptoAgent:
    """Crypto Agent for providing cryptocurrency-related consultations and advice."""
    
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
    
    async def _detect_crypto_intent(
        self,
        user_query: str,
        language: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> bool:
        """
        Detect if the user's query is related to cryptocurrency.
        
        Args:
            user_query: User's query text
            language: Detected language ("zh" or "en")
            api_key: Optional API key
            base_url: Optional base URL
            
        Returns:
            True if the query is related to cryptocurrency, False otherwise
        """
        # First, try keyword-based detection (fast and reliable)
        text_lower = user_query.lower()
        
        # Chinese keywords
        chinese_keywords = [
            "加密货币", "数字货币", "比特币", "以太坊", "区块链", "代币", "币", "加密",
            "BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "DOT", "DOGE", "SHIB",
            "交易", "投资", "挖矿", "DeFi", "NFT", "Web3", "钱包", "交易所",
            "价格", "行情", "涨跌", "牛市", "熊市", "持仓", "买入", "卖出"
        ]
        
        # English keywords
        english_keywords = [
            "crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "token", "coin",
            "btc", "eth", "bnb", "sol", "ada", "xrp", "dot", "doge", "shib",
            "trading", "investment", "mining", "defi", "nft", "web3", "wallet", "exchange",
            "price", "market", "bull", "bear", "hold", "buy", "sell", "altcoin"
        ]
        
        # Check for keywords
        if language == "zh":
            for keyword in chinese_keywords:
                if keyword in user_query:
                    logger.debug(f"Detected crypto-related intent via keyword: {keyword}")
                    return True
        else:
            for keyword in english_keywords:
                if keyword in text_lower:
                    logger.debug(f"Detected crypto-related intent via keyword: {keyword}")
                    return True
        
        # If no keywords found, use LLM for more nuanced detection
        try:
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            model_name = settings.openrouter_model if api_key else self.chat_model
            
            if language == "zh":
                intent_prompt = f"""判断以下用户提问是否与加密货币、数字货币、区块链、比特币、以太坊相关。

用户提问：{user_query}

请只回答"是"或"否"，不要其他文字。"""
                system_prompt = "你是一个意图分析助手，判断用户提问是否与加密货币相关。只回答'是'或'否'。"
            else:
                intent_prompt = f"""Determine if the following user question is related to cryptocurrency, digital currency, blockchain, Bitcoin, or Ethereum.

User question: {user_query}

Please answer only "yes" or "no", no other text."""
                system_prompt = "You are an intent analysis assistant that determines if user questions are related to cryptocurrency. Answer only 'yes' or 'no'."
            
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
            
            logger.debug(f"LLM crypto intent detection result: {is_related} (response: {answer})")
            return is_related
            
        except Exception as e:
            logger.warning(f"Error in LLM crypto intent detection: {e}, defaulting to True")
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
        Process cryptocurrency-related query from user.
        
        Args:
            user_query: User's cryptocurrency-related question or concern
            conversation_history: Optional conversation history (list of messages with role and content)
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing crypto advice or consultation response
        """
        logger.info(f"Processing crypto query: {user_query[:100]}...")
        
        # Detect language first
        language = self._detect_language(user_query)
        
        # Check if the query is related to cryptocurrency
        is_crypto_related = await self._detect_crypto_intent(
            user_query=user_query,
            language=language,
            api_key=api_key,
            base_url=base_url
        )
        
        # If not related to crypto, return a reminder
        if not is_crypto_related:
            if language == "zh":
                reminder = """我是一个加密货币咨询助手，专门帮助用户解答加密货币相关的问题。

我可以为您提供：
- 加密货币基础知识（比特币、以太坊等）
- 区块链技术原理和概念
- 加密货币投资建议和风险提示
- 市场行情分析和趋势解读
- DeFi、NFT、Web3 等新兴领域介绍
- 钱包使用和交易安全建议

重要提示：
- 我的建议仅供参考，不构成投资建议
- 加密货币投资有风险，请谨慎投资
- 我不会提供具体的买卖建议或价格预测
- 请做好风险管理，不要投入超过承受能力的资金

请在提问中描述您的加密货币相关问题，例如：
"什么是比特币？"
"如何选择加密货币钱包？"
"DeFi 是什么？"
"加密货币投资有哪些风险？" """
            else:
                reminder = """I am a cryptocurrency consultation assistant, specialized in helping users with cryptocurrency-related questions.

I can provide:
- Cryptocurrency basics (Bitcoin, Ethereum, etc.)
- Blockchain technology principles and concepts
- Cryptocurrency investment advice and risk warnings
- Market analysis and trend interpretation
- Introduction to emerging fields like DeFi, NFT, Web3
- Wallet usage and trading security advice

Important Notice:
- My advice is for reference only and does not constitute investment advice
- Cryptocurrency investment carries risks, please invest cautiously
- I will not provide specific buy/sell advice or price predictions
- Please manage risks properly and do not invest more than you can afford to lose

Please describe your cryptocurrency-related question, for example:
"What is Bitcoin?"
"How to choose a cryptocurrency wallet?"
"What is DeFi?"
"What are the risks of cryptocurrency investment?" """
            
            return {
                "answer": reminder
            }
        
        # Build context from conversation history if available
        context = ""
        if conversation_history:
            # Include recent messages for context
            recent_messages = conversation_history[-10:]  # Last 10 messages
            context = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_messages])
        
        # Build system prompt for crypto consultation
        if language == "zh":
            system_prompt = """你是一位专业、友善的加密货币咨询助手。你的职责是：

1. 提供加密货币相关的知识、信息和建议
2. 回答关于比特币、以太坊、区块链技术、DeFi、NFT、Web3 等方面的问题
3. 解释加密货币市场的基本概念和原理
4. 提供投资风险提示和教育性内容
5. 帮助用户理解加密货币生态系统

重要原则：
- 始终强调：你的建议仅供参考，不构成投资建议
- 必须提醒用户：加密货币投资有风险，请谨慎投资
- 不要提供具体的买卖建议或价格预测
- 提供科学、准确的信息
- 语气要专业、客观、教育性
- 强调风险管理和资金安全
- 鼓励用户进行自己的研究（DYOR - Do Your Own Research）

请用中文回答，语言要清晰易懂。"""
        else:
            system_prompt = """You are a professional and friendly cryptocurrency consultation assistant. Your responsibilities are:

1. Provide cryptocurrency-related knowledge, information, and advice
2. Answer questions about Bitcoin, Ethereum, blockchain technology, DeFi, NFT, Web3, etc.
3. Explain basic concepts and principles of the cryptocurrency market
4. Provide investment risk warnings and educational content
5. Help users understand the cryptocurrency ecosystem

Important Principles:
- Always emphasize: Your advice is for reference only and does not constitute investment advice
- Must remind users: Cryptocurrency investment carries risks, please invest cautiously
- Do not provide specific buy/sell advice or price predictions
- Provide scientific and accurate information
- Tone should be professional, objective, and educational
- Emphasize risk management and fund security
- Encourage users to do their own research (DYOR - Do Your Own Research)

Please answer in English, with clear and understandable language."""
        
        # Build user prompt
        history_section = f'对话历史：\n{context}' if context else ''
        if language == "zh":
            user_prompt = f"""用户问题：{user_query}

{history_section}

请根据用户的问题，提供专业、友善的加密货币相关建议。记住：
- 如果问题涉及投资建议，必须强调风险提示
- 不要提供具体的买卖建议或价格预测
- 提供实用的知识和信息
- 语气要专业、客观、教育性
- 强调你的建议仅供参考，不构成投资建议"""
        else:
            user_prompt = f"""User question: {user_query}

{history_section}

Please provide professional and friendly cryptocurrency-related advice based on the user's question. Remember:
- If the question involves investment advice, must emphasize risk warnings
- Do not provide specific buy/sell advice or price predictions
- Provide practical knowledge and information
- Tone should be professional, objective, and educational
- Emphasize that your advice is for reference only and does not constitute investment advice"""
        
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating crypto consultation (model: {model_name})")
        
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
            logger.info(f"Generated crypto consultation response (length: {len(answer)})")
            
            return {
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Error generating crypto consultation: {e}", exc_info=True)
            # Return a friendly error message instead of raising exception
            if language == "zh":
                error_message = "抱歉，生成加密货币建议时遇到了问题。"
                if "402" in str(e) or "Insufficient credits" in str(e):
                    error_message = "抱歉，API 服务暂时不可用（余额不足）。请稍后再试或联系管理员。"
                elif "401" in str(e) or "Unauthorized" in str(e):
                    error_message = "抱歉，API 认证失败。请检查 API key 是否正确。"
                else:
                    error_message = f"抱歉，生成加密货币建议时遇到了技术问题：{str(e)[:100]}"
            else:
                error_message = "Sorry, encountered an issue while generating crypto advice."
                if "402" in str(e) or "Insufficient credits" in str(e):
                    error_message = "Sorry, API service is temporarily unavailable (insufficient credits). Please try again later or contact the administrator."
                elif "401" in str(e) or "Unauthorized" in str(e):
                    error_message = "Sorry, API authentication failed. Please check if the API key is correct."
                else:
                    error_message = f"Sorry, encountered a technical issue while generating crypto advice: {str(e)[:100]}"
            
            return {
                "answer": error_message
            }

