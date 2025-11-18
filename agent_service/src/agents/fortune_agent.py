"""Fortune Telling Agent for predicting tomorrow's fortune."""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import re
import json
from openai import OpenAI
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
    
    async def extract_info(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract name, birth year, and zodiac sign from natural language input.
        
        Args:
            user_query: User's natural language input
            conversation_history: Optional conversation history (list of messages with role and content)
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing extracted information and completeness status
        """
        logger.info(f"Extracting information from query: {user_query[:100]}...")
        
        # Build context from conversation history if available
        context = ""
        if conversation_history:
            # Include recent messages for context
            recent_messages = conversation_history[-5:]  # Last 5 messages
            context = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_messages])
        
        # Build prompt for information extraction
        extract_prompt = f"""请从以下文本中提取用户的姓名、出生年份和星座信息。

用户输入：{user_query}
{f'对话历史：{context}' if context else ''}

请仔细提取以下信息：
1. 姓名（name）：用户的名字，可能出现在"我叫"、"我是"、"姓名叫"、"名字是"等表达中
2. 出生年份（birth_year）：4位数字的年份，如1990、2000等，可能出现在"出生"、"年出生"等表达中
3. 星座（zodiac_sign）：十二星座之一，可能是中文（白羊座、金牛座等）或英文（Aries、Taurus等）

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

请以JSON格式返回，如果找到了信息就填写，找不到就写null：
{{
    "name": "提取的姓名或null",
    "birth_year": 提取的年份数字或null,
    "zodiac_sign": "提取的星座（英文）或null"
}}

只返回JSON，不要其他文字。"""
        
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Extracting information (model: {model_name})")
        
        try:
            # Get LLM client
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个信息提取助手，擅长从自然语言中提取结构化信息。请严格按照要求返回JSON格式。"
                    },
                    {
                        "role": "user",
                        "content": extract_prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more accurate extraction
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content.strip()
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
            
            # Check completeness
            missing = []
            if not extracted_info.get("name"):
                missing.append("name")
            if not extracted_info.get("birth_year"):
                missing.append("birth_year")
            if not extracted_info.get("zodiac_sign"):
                missing.append("zodiac_sign")
            
            extracted_info["missing_info"] = missing
            extracted_info["is_complete"] = len(missing) == 0
            
            logger.info(f"Extracted info: name={extracted_info.get('name')}, birth_year={extracted_info.get('birth_year')}, zodiac_sign={extracted_info.get('zodiac_sign')}, missing={missing}")
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting information: {e}", exc_info=True)
            # Try manual extraction as fallback when LLM fails
            logger.info("LLM extraction failed, trying manual extraction as fallback")
            extracted_info = self._manual_extract(user_query)
            
            # Normalize zodiac sign to English
            if extracted_info.get("zodiac_sign"):
                extracted_info["zodiac_sign"] = self._normalize_zodiac_sign(extracted_info["zodiac_sign"])
            
            # Check completeness
            missing = []
            if not extracted_info.get("name"):
                missing.append("name")
            if not extracted_info.get("birth_year"):
                missing.append("birth_year")
            if not extracted_info.get("zodiac_sign"):
                missing.append("zodiac_sign")
            
            extracted_info["missing_info"] = missing
            extracted_info["is_complete"] = len(missing) == 0
            
            logger.info(f"Manual extraction result: name={extracted_info.get('name')}, birth_year={extracted_info.get('birth_year')}, zodiac_sign={extracted_info.get('zodiac_sign')}, missing={missing}")
            
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
        # First, extract information from the query
        extracted_info = await self.extract_info(
            user_query=user_query,
            conversation_history=conversation_history,
            api_key=api_key,
            base_url=base_url
        )
        
        # If information is incomplete, return reminder
        if not extracted_info.get("is_complete"):
            missing = extracted_info.get("missing_info", [])
            missing_chinese = {
                "name": "姓名",
                "birth_year": "出生年份",
                "zodiac_sign": "星座"
            }
            missing_list = [missing_chinese.get(m, m) for m in missing]
            
            reminder = f"为了给您预测明天的运势，我还需要以下信息：{', '.join(missing_list)}。\n\n请告诉我：\n"
            if "name" in missing:
                reminder += "- 您的姓名\n"
            if "birth_year" in missing:
                reminder += "- 您的出生年份（例如：1990）\n"
            if "zodiac_sign" in missing:
                reminder += "- 您的星座（例如：白羊座、Aries等）\n"
            
            # Format as text response (consistent with other agents)
            response_text = reminder
            if extracted_info.get("name"):
                response_text = f"已获取信息：姓名 {extracted_info.get('name')}\n\n" + response_text
            if extracted_info.get("birth_year"):
                response_text = f"已获取信息：出生年份 {extracted_info.get('birth_year')}\n\n" + response_text
            if extracted_info.get("zodiac_sign"):
                response_text = f"已获取信息：星座 {extracted_info.get('zodiac_sign')}\n\n" + response_text
            
            return {
                "answer": response_text
            }
        
        # If complete, proceed with prediction
        return await self.predict(
            name=extracted_info["name"],
            birth_year=extracted_info["birth_year"],
            zodiac_sign=extracted_info["zodiac_sign"],
            api_key=api_key,
            base_url=base_url
        )
    
    async def predict(
        self,
        name: str,
        birth_year: int,
        zodiac_sign: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict tomorrow's fortune based on user's information.
        
        Args:
            name: User's name
            birth_year: User's birth year
            zodiac_sign: User's zodiac sign
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Dictionary containing fortune prediction and related information
        """
        logger.info(f"Predicting fortune for {name} (born {birth_year}, {zodiac_sign})")
        
        # Calculate tomorrow's date
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y年%m月%d日")
        
        # Calculate age
        current_year = datetime.now().year
        age = current_year - birth_year
        
        # Build prompt for fortune telling
        prompt = f"""请为以下用户预测明天的运势：

姓名：{name}
出生年份：{birth_year}年（今年{age}岁）
星座：{zodiac_sign}
预测日期：{tomorrow_str}

请根据用户的姓名、年龄和星座，预测明天的运势。请提供：
1. 整体运势预测（包括事业、爱情、健康、财运等方面）
2. 幸运数字（3-5个数字）
3. 幸运颜色
4. 明日建议

请用中文回答，语气要友好、积极，但也要保持一定的神秘感。预测要具体但不过于绝对。"""
        
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating fortune prediction (model: {model_name})")
        
        try:
            # Get LLM client (OpenRouter if api_key provided, otherwise AIHubMix)
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位经验丰富的占星师和命理师，擅长根据姓名、出生年份和星座预测运势。你的预测风格既神秘又积极，能够给用户带来希望和指导。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,  # Higher temperature for more creative predictions
                max_tokens=500  # Allow more tokens for detailed predictions
            )
            
            prediction_text = response.choices[0].message.content
            logger.info(f"Generated fortune prediction (length: {len(prediction_text)})")
            
            # Try to extract lucky numbers and color from the prediction
            lucky_numbers = self._extract_lucky_numbers(prediction_text)
            lucky_color = self._extract_lucky_color(prediction_text)
            advice = self._extract_advice(prediction_text)
            
            # Format response as text (consistent with other agents)
            # The prediction_text already contains all the information, but we can enhance it
            response_text = prediction_text
            
            # Add extracted structured info if available and not already in text
            if lucky_numbers and f"幸运数字" not in prediction_text:
                response_text += f"\n\n幸运数字：{', '.join(map(str, lucky_numbers))}"
            if lucky_color and f"幸运颜色" not in prediction_text:
                response_text += f"\n幸运颜色：{lucky_color}"
            if advice and f"建议" not in prediction_text:
                response_text += f"\n建议：{advice}"
            
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

