"""Tarot Card Reading Agent for divination and guidance."""

from typing import Dict, Any, Optional, List
import random
import re
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class TarotAgent:
    """Tarot Card Reading Agent for divination and guidance based on user questions."""
    
    # Standard 78-card Tarot deck
    TAROT_DECK = [
        # Major Arcana (22 cards)
        {"name": "The Fool", "number": 0, "arcana": "major"},
        {"name": "The Magician", "number": 1, "arcana": "major"},
        {"name": "The High Priestess", "number": 2, "arcana": "major"},
        {"name": "The Empress", "number": 3, "arcana": "major"},
        {"name": "The Emperor", "number": 4, "arcana": "major"},
        {"name": "The Hierophant", "number": 5, "arcana": "major"},
        {"name": "The Lovers", "number": 6, "arcana": "major"},
        {"name": "The Chariot", "number": 7, "arcana": "major"},
        {"name": "Strength", "number": 8, "arcana": "major"},
        {"name": "The Hermit", "number": 9, "arcana": "major"},
        {"name": "Wheel of Fortune", "number": 10, "arcana": "major"},
        {"name": "Justice", "number": 11, "arcana": "major"},
        {"name": "The Hanged Man", "number": 12, "arcana": "major"},
        {"name": "Death", "number": 13, "arcana": "major"},
        {"name": "Temperance", "number": 14, "arcana": "major"},
        {"name": "The Devil", "number": 15, "arcana": "major"},
        {"name": "The Tower", "number": 16, "arcana": "major"},
        {"name": "The Star", "number": 17, "arcana": "major"},
        {"name": "The Moon", "number": 18, "arcana": "major"},
        {"name": "The Sun", "number": 19, "arcana": "major"},
        {"name": "Judgement", "number": 20, "arcana": "major"},
        {"name": "The World", "number": 21, "arcana": "major"},
        # Minor Arcana - Wands (14 cards)
        {"name": "Ace of Wands", "suit": "Wands", "number": 1, "arcana": "minor"},
        {"name": "Two of Wands", "suit": "Wands", "number": 2, "arcana": "minor"},
        {"name": "Three of Wands", "suit": "Wands", "number": 3, "arcana": "minor"},
        {"name": "Four of Wands", "suit": "Wands", "number": 4, "arcana": "minor"},
        {"name": "Five of Wands", "suit": "Wands", "number": 5, "arcana": "minor"},
        {"name": "Six of Wands", "suit": "Wands", "number": 6, "arcana": "minor"},
        {"name": "Seven of Wands", "suit": "Wands", "number": 7, "arcana": "minor"},
        {"name": "Eight of Wands", "suit": "Wands", "number": 8, "arcana": "minor"},
        {"name": "Nine of Wands", "suit": "Wands", "number": 9, "arcana": "minor"},
        {"name": "Ten of Wands", "suit": "Wands", "number": 10, "arcana": "minor"},
        {"name": "Page of Wands", "suit": "Wands", "rank": "Page", "arcana": "minor"},
        {"name": "Knight of Wands", "suit": "Wands", "rank": "Knight", "arcana": "minor"},
        {"name": "Queen of Wands", "suit": "Wands", "rank": "Queen", "arcana": "minor"},
        {"name": "King of Wands", "suit": "Wands", "rank": "King", "arcana": "minor"},
        # Minor Arcana - Cups (14 cards)
        {"name": "Ace of Cups", "suit": "Cups", "number": 1, "arcana": "minor"},
        {"name": "Two of Cups", "suit": "Cups", "number": 2, "arcana": "minor"},
        {"name": "Three of Cups", "suit": "Cups", "number": 3, "arcana": "minor"},
        {"name": "Four of Cups", "suit": "Cups", "number": 4, "arcana": "minor"},
        {"name": "Five of Cups", "suit": "Cups", "number": 5, "arcana": "minor"},
        {"name": "Six of Cups", "suit": "Cups", "number": 6, "arcana": "minor"},
        {"name": "Seven of Cups", "suit": "Cups", "number": 7, "arcana": "minor"},
        {"name": "Eight of Cups", "suit": "Cups", "number": 8, "arcana": "minor"},
        {"name": "Nine of Cups", "suit": "Cups", "number": 9, "arcana": "minor"},
        {"name": "Ten of Cups", "suit": "Cups", "number": 10, "arcana": "minor"},
        {"name": "Page of Cups", "suit": "Cups", "rank": "Page", "arcana": "minor"},
        {"name": "Knight of Cups", "suit": "Cups", "rank": "Knight", "arcana": "minor"},
        {"name": "Queen of Cups", "suit": "Cups", "rank": "Queen", "arcana": "minor"},
        {"name": "King of Cups", "suit": "Cups", "rank": "King", "arcana": "minor"},
        # Minor Arcana - Swords (14 cards)
        {"name": "Ace of Swords", "suit": "Swords", "number": 1, "arcana": "minor"},
        {"name": "Two of Swords", "suit": "Swords", "number": 2, "arcana": "minor"},
        {"name": "Three of Swords", "suit": "Swords", "number": 3, "arcana": "minor"},
        {"name": "Four of Swords", "suit": "Swords", "number": 4, "arcana": "minor"},
        {"name": "Five of Swords", "suit": "Swords", "number": 5, "arcana": "minor"},
        {"name": "Six of Swords", "suit": "Swords", "number": 6, "arcana": "minor"},
        {"name": "Seven of Swords", "suit": "Swords", "number": 7, "arcana": "minor"},
        {"name": "Eight of Swords", "suit": "Swords", "number": 8, "arcana": "minor"},
        {"name": "Nine of Swords", "suit": "Swords", "number": 9, "arcana": "minor"},
        {"name": "Ten of Swords", "suit": "Swords", "number": 10, "arcana": "minor"},
        {"name": "Page of Swords", "suit": "Swords", "rank": "Page", "arcana": "minor"},
        {"name": "Knight of Swords", "suit": "Swords", "rank": "Knight", "arcana": "minor"},
        {"name": "Queen of Swords", "suit": "Swords", "rank": "Queen", "arcana": "minor"},
        {"name": "King of Swords", "suit": "Swords", "rank": "King", "arcana": "minor"},
        # Minor Arcana - Pentacles (14 cards)
        {"name": "Ace of Pentacles", "suit": "Pentacles", "number": 1, "arcana": "minor"},
        {"name": "Two of Pentacles", "suit": "Pentacles", "number": 2, "arcana": "minor"},
        {"name": "Three of Pentacles", "suit": "Pentacles", "number": 3, "arcana": "minor"},
        {"name": "Four of Pentacles", "suit": "Pentacles", "number": 4, "arcana": "minor"},
        {"name": "Five of Pentacles", "suit": "Pentacles", "number": 5, "arcana": "minor"},
        {"name": "Six of Pentacles", "suit": "Pentacles", "number": 6, "arcana": "minor"},
        {"name": "Seven of Pentacles", "suit": "Pentacles", "number": 7, "arcana": "minor"},
        {"name": "Eight of Pentacles", "suit": "Pentacles", "number": 8, "arcana": "minor"},
        {"name": "Nine of Pentacles", "suit": "Pentacles", "number": 9, "arcana": "minor"},
        {"name": "Ten of Pentacles", "suit": "Pentacles", "number": 10, "arcana": "minor"},
        {"name": "Page of Pentacles", "suit": "Pentacles", "rank": "Page", "arcana": "minor"},
        {"name": "Knight of Pentacles", "suit": "Pentacles", "rank": "Knight", "arcana": "minor"},
        {"name": "Queen of Pentacles", "suit": "Pentacles", "rank": "Queen", "arcana": "minor"},
        {"name": "King of Pentacles", "suit": "Pentacles", "rank": "King", "arcana": "minor"},
    ]
    
    def __init__(self):
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        # Keep OpenAI client for backward compatibility
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = "google/gemini-2.5-pro"
        
        # Initialize LangChain ChatOpenAI (default)
        self.langchain_llm = ChatOpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url,
            model="google/gemini-2.5-pro",
            temperature=0.7  # Higher temperature for more creative interpretations
        )
    
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
            model_name = "google/gemini-2.5-pro"
        else:
            # Use default AIHubMix
            client_base_url = settings.aihubmix_base_url
            model_name = "google/gemini-2.5-pro"
        
        llm_kwargs = {
            "api_key": api_key or settings.aihubmix_api_key,
            "base_url": client_base_url,
            "model": model_name,
            "temperature": 0.7  # Higher temperature for creative tarot readings
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
    
    def _draw_cards(self, num_cards: int = 3) -> List[Dict[str, Any]]:
        """
        Draw random cards from the tarot deck.
        
        Args:
            num_cards: Number of cards to draw
            
        Returns:
            List of drawn cards with orientation (upright/reversed)
        """
        drawn_cards = random.sample(self.TAROT_DECK, num_cards)
        result = []
        for card in drawn_cards:
            # Randomly determine if card is upright or reversed (50% chance)
            is_reversed = random.choice([True, False])
            card_info = card.copy()
            card_info["orientation"] = "reversed" if is_reversed else "upright"
            result.append(card_info)
        return result
    
    def _extract_question(self, user_query: str, language: str) -> str:
        """
        Extract the user's question from their query.
        For simple version, just return the query as-is or extract the question part.
        
        Args:
            user_query: User's input
            language: Language of the query
            
        Returns:
            Extracted question
        """
        # Simple extraction - look for question patterns
        if language == "zh":
            # Remove common prefixes
            patterns_to_remove = [
                r'^帮我.*?算',
                r'^我想.*?占卜',
                r'^请.*?占卜',
                r'^帮我.*?占卜',
            ]
            question = user_query
            for pattern in patterns_to_remove:
                question = re.sub(pattern, '', question, flags=re.IGNORECASE)
            return question.strip() or user_query
        else:
            # English patterns
            patterns_to_remove = [
                r'^please.*?read',
                r'^can you.*?read',
                r'^i want.*?reading',
                r'^help me.*?read',
            ]
            question = user_query
            for pattern in patterns_to_remove:
                question = re.sub(pattern, '', question, flags=re.IGNORECASE)
            return question.strip() or user_query
    
    async def query(
        self,
        user_query: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process tarot reading query from natural language.
        Simple version: single interaction, automatic 3-card spread.
        
        Args:
            user_query: User's question or query
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing tarot reading result
        """
        logger.info(f"Processing tarot query: {user_query[:100]}...")
        
        # Detect language
        language = self._detect_language(user_query)
        
        # Check if query is related to tarot
        if not self._is_tarot_related(user_query, language):
            if language == "zh":
                reminder = """我是一个塔罗牌占卜助手，专门帮助用户进行塔罗牌占卜和指引。

我可以为您：
- 解答关于爱情、事业、学业、人际关系等各方面的问题
- 使用三张牌牌阵（过去-现在-未来）为您解读
- 提供详细的牌面含义和指引建议

请直接告诉我您想要占卜的问题，例如：
"我想知道我的感情运势如何？"
"我的事业发展会怎样？"
"我应该如何改善人际关系？" """
            else:
                reminder = """I am a Tarot card reading assistant, specialized in helping users with tarot divination and guidance.

I can help you with:
- Questions about love, career, studies, relationships, and more
- Three-card spread readings (Past-Present-Future)
- Detailed card interpretations and guidance

Just tell me your question, for example:
"What does my love life look like?"
"How will my career develop?"
"What should I do to improve my relationships?" """
            
            return {
                "answer": reminder
            }
        
        # Extract question
        question = self._extract_question(user_query, language)
        
        # Draw 3 cards for Past-Present-Future spread
        drawn_cards = self._draw_cards(3)
        
        # Perform reading
        return await self.read(
            question=question,
            cards=drawn_cards,
            spread_type="three_card",
            language=language,
            api_key=api_key,
            base_url=base_url
        )
    
    def _is_tarot_related(self, text: str, language: str) -> bool:
        """
        Check if the query is related to tarot reading.
        
        Args:
            text: User's query
            language: Language of the query
            
        Returns:
            True if related to tarot, False otherwise
        """
        if language == "zh":
            tarot_keywords = [
                "塔罗", "占卜", "运势", "感情", "爱情", "事业", "学业",
                "人际关系", "未来", "指引", "建议", "如何", "怎样",
                "会怎样", "怎么样", "如何", "怎么办"
            ]
        else:
            tarot_keywords = [
                "tarot", "reading", "fortune", "love", "career", "relationship",
                "future", "guidance", "advice", "how", "what", "will",
                "should", "question", "wondering"
            ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in tarot_keywords)
    
    async def read(
        self,
        question: str,
        cards: List[Dict[str, Any]],
        spread_type: str = "three_card",
        language: str = "zh",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform tarot reading with given cards.
        
        Args:
            question: User's question
            cards: List of drawn cards
            spread_type: Type of spread (default: "three_card")
            language: Language for response
            api_key: Optional API key
            base_url: Optional base URL
            
        Returns:
            Dictionary containing tarot reading result
        """
        logger.info(f"Performing tarot reading: question={question[:50]}, cards={[c['name'] for c in cards]}, language={language}")
        
        # Build card information string
        if spread_type == "three_card":
            positions = ["过去 (Past)", "现在 (Present)", "未来 (Future)"] if language == "zh" else ["Past", "Present", "Future"]
        else:
            positions = [f"位置 {i+1}" for i in range(len(cards))] if language == "zh" else [f"Position {i+1}" for i in range(len(cards))]
        
        cards_info = []
        for i, card in enumerate(cards):
            card_name = card["name"]
            orientation = card["orientation"]
            if language == "zh":
                orientation_text = "逆位" if orientation == "reversed" else "正位"
                cards_info.append(f"{positions[i]}: {card_name} ({orientation_text})")
            else:
                cards_info.append(f"{positions[i]}: {card_name} ({orientation})")
        
        cards_text = "\n".join(cards_info)
        
        # Build prompt
        if language == "zh":
            prompt = f"""请为以下塔罗牌占卜进行解读：

用户问题：{question}

抽到的牌：
{cards_text}

请按照以下格式进行详细解读：
1. 整体解读：简要说明这次占卜的整体含义
2. 各张牌的解读：
   - {positions[0]}：详细解释这张牌的含义，结合用户的问题
   - {positions[1]}：详细解释这张牌的含义，结合用户的问题
   - {positions[2]}：详细解释这张牌的含义，结合用户的问题
3. 综合指引：将三张牌联系起来，给出综合的建议和指引

注意：
- 如果牌是逆位，请说明逆位的特殊含义
- 要结合用户的具体问题来解读
- 语气要温和、神秘，但也要积极正面
- 提供实用的建议和指引
- 控制回答长度在800-1200字之间"""
            
            system_prompt = "你是一位经验丰富的塔罗牌占卜师，精通塔罗牌的各种牌阵和解读方法。你的解读风格既神秘又温暖，能够给用户带来启发和指引。"
        else:
            prompt = f"""Please provide a detailed tarot reading interpretation:

User's Question: {question}

Cards Drawn:
{cards_text}

Please provide a detailed interpretation in the following format:
1. Overall Reading: Briefly explain the overall meaning of this reading
2. Individual Card Interpretations:
   - {positions[0]}: Detailed explanation of this card's meaning in relation to the user's question
   - {positions[1]}: Detailed explanation of this card's meaning in relation to the user's question
   - {positions[2]}: Detailed explanation of this card's meaning in relation to the user's question
3. Comprehensive Guidance: Connect all three cards together and provide overall advice and guidance

Notes:
- If a card is reversed, explain the special meaning of the reversed position
- Connect the interpretation to the user's specific question
- Use a warm, mysterious but positive tone
- Provide practical advice and guidance
- Keep the response between 800-1200 words"""
            
            system_prompt = "You are an experienced tarot card reader, skilled in various tarot spreads and interpretation methods. Your reading style is both mysterious and warm, able to provide inspiration and guidance to users."
        
        try:
            # Use LangChain for reading
            langchain_llm = self._get_langchain_llm(api_key=api_key, base_url=base_url, max_tokens=2000)
            
            # Build prompt template
            reading_prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{prompt}")
            ])
            
            # Format and invoke
            messages = reading_prompt_template.format_messages(prompt=prompt)
            response = await langchain_llm.ainvoke(messages)
            
            reading_text = response.content
            logger.info(f"Generated tarot reading (length: {len(reading_text)})")
            
            # Format response
            response_text = reading_text
            
            # Add card information at the beginning
            if language == "zh":
                header = f"🔮 塔罗牌占卜结果\n\n"
                header += f"您的问题：{question}\n\n"
                header += f"抽到的牌：\n{cards_text}\n\n"
                header += "=" * 40 + "\n\n"
            else:
                header = f"🔮 Tarot Reading Result\n\n"
                header += f"Your Question: {question}\n\n"
                header += f"Cards Drawn:\n{cards_text}\n\n"
                header += "=" * 40 + "\n\n"
            
            response_text = header + response_text
            
            return {
                "answer": response_text,
                "question": question,
                "cards": cards,
                "spread_type": spread_type
            }
        except Exception as e:
            logger.error(f"Error generating tarot reading: {e}", exc_info=True)
            # Return a friendly error message
            error_message = "抱歉，进行塔罗牌占卜时遇到了问题。" if language == "zh" else "Sorry, I encountered an issue while performing the tarot reading."
            if "402" in str(e) or "Insufficient credits" in str(e):
                error_message = "抱歉，API 服务暂时不可用（余额不足）。请稍后再试或联系管理员。" if language == "zh" else "Sorry, API service is temporarily unavailable (insufficient credits). Please try again later or contact the administrator."
            elif "401" in str(e) or "Unauthorized" in str(e):
                error_message = "抱歉，API 认证失败。请检查 API key 是否正确。" if language == "zh" else "Sorry, API authentication failed. Please check if the API key is correct."
            else:
                error_message = f"抱歉，进行塔罗牌占卜时遇到了技术问题：{str(e)[:100]}" if language == "zh" else f"Sorry, I encountered a technical issue: {str(e)[:100]}"
            
            return {
                "answer": error_message
            }

