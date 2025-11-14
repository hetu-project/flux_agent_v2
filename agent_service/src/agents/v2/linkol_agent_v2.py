"""Linkol Agent V2 for querying Linkol-related information and KOL data with cheaper models and RAG-based tweet search."""

from typing import List, Dict, Any, Optional
import httpx
from openai import OpenAI
from src.config import get_settings
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.services.linkol import LinkolService
from src.repositories.project_content_repository import ProjectContentRepository
from src.repositories.tweet_repository import TweetRepository
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class LinkolAgentV2:
    """Linkol Agent V2 for querying Linkol-related information and KOL data with cheaper models and RAG-based tweet search."""
    
    def __init__(
        self,
        project_content_repo: Optional[ProjectContentRepository] = None,
        tweet_repo: Optional[TweetRepository] = None
    ):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.project_content_repo = project_content_repo
        self.tweet_repo = tweet_repo
        self.linkol_service = LinkolService()
        
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        # Use cheaper model: gpt-4o-mini instead of grok
        self.chat_model = "gpt-4o-mini"
        self._cached_project_names: Optional[List[str]] = None
    
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
    
    def _search_tweets(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search tweets using RAG (vector similarity search).
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant tweets
        """
        if not self.tweet_repo:
            logger.debug("Tweet repository not available")
            return []
        
        try:
            # Generate query embedding
            query_vector = self.embedding.embed_text(query)
            
            # Search tweets (no project filter for Linkol queries)
            results = self.tweet_repo.search(
                query_vector=query_vector,
                top_k=top_k,
                min_score=0.6  # Minimum similarity threshold
            )
            
            logger.info(f"Found {len(results)} relevant tweets for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error searching tweets: {e}", exc_info=True)
            return []
    
    def _search_linkol_content(self, user_question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for Linkol-related content in project_content collection.
        
        Args:
            user_question: User's question
            top_k: Number of results to return
            
        Returns:
            List of relevant content items
        """
        if not self.project_content_repo:
            logger.debug("Project content repository not available")
            return []
        
        try:
            logger.debug(f"Searching for Linkol-related content")
            # Search in project_content collection, filter by project_name = "Linkol"
            results = self.project_content_repo.search(
                query=user_question,
                project_name="Linkol",  # Filter for Linkol project
                top_k=top_k,
                min_score=0.6
            )
            
            logger.info(f"Found {len(results)} Linkol-related content items")
            return results
        except Exception as e:
            logger.error(f"Error searching Linkol content: {e}")
            return []
    
    def _extract_screen_name(self, user_question: str) -> Optional[str]:
        """
        Extract Twitter screen name from user question using regex.
        
        Args:
            user_question: User's question
            
        Returns:
            Screen name (without @) if found, None otherwise
        """
        import re
        
        # Pattern to match @username
        pattern = r'@([a-zA-Z0-9_]+)'
        matches = re.findall(pattern, user_question)
        
        if matches:
            screen_name = matches[0]
            logger.debug(f"Extracted screen name from @: {screen_name}")
            return screen_name
        
        # Check if question mentions a specific username pattern without @
        question_lower = user_question.lower()
        valuation_keywords = ["估值", "价格", "价钱", "price", "valuation", "值多少钱"]
        
        for keyword in valuation_keywords:
            if keyword in question_lower:
                # Extract text before the keyword
                before_keyword = question_lower.split(keyword)[0].strip()
                # Try to find the last valid username pattern
                parts = before_keyword.split()
                for part in reversed(parts):
                    part_clean = part.strip('的').strip()
                    # Match username pattern (alphanumeric + underscore only)
                    if re.match(r'^[a-zA-Z0-9_]+$', part_clean) and len(part_clean) > 2:
                        logger.debug(f"Extracted screen name from fallback: {part_clean}")
                        return part_clean
        
        return None
    
    async def _get_top_kol_price(self) -> Optional[Dict[str, Any]]:
        """
        Get the price of the top-ranked KOL.
        
        Returns:
            Dict with KOL info and price, or None if error
        """
        try:
            logger.info("Fetching hot KOLs list...")
            # Step 1: Get top 20 hot KOLs
            hot_kols_result = await self.linkol_service.get_hot_kols()
            
            if hot_kols_result.get("code") != 200:
                logger.error(f"Failed to get hot KOLs: {hot_kols_result.get('msg')}")
                return None
            
            kols_list = hot_kols_result.get("data", {}).get("list", [])
            if not kols_list:
                logger.warning("No KOLs found in hot KOLs list")
                return None
            
            # Step 2: Get the top-ranked KOL (first in list)
            top_kol = kols_list[0]
            screen_name = top_kol.get("screen_name")
            
            if not screen_name:
                logger.error("Top KOL has no screen_name")
                return None
            
            logger.info(f"Top KOL: @{screen_name}")
            
            # Step 3: Get price for top KOL
            logger.info(f"Fetching price for @{screen_name}...")
            price_result = await self.linkol_service.get_kol_price(screen_name=screen_name)
            
            if price_result.get("code") != 200:
                logger.error(f"Failed to get KOL price: {price_result.get('msg')}")
                return None
            
            price = price_result.get("data", {}).get("price")
            
            return {
                "kol": top_kol,
                "price": price,
                "screen_name": screen_name
            }
        except Exception as e:
            logger.error(f"Error getting top KOL price: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def query(
        self,
        user_question: str,
        top_k: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the Linkol agent with intelligent intent classification.
        
        Logic:
        1. Check if user is asking for specific KOL valuation
        2. Check if user is asking for general KOL data
        3. Search for Linkol-related content and tweets using RAG
        4. Generate answer using cheaper LLM model
        
        Args:
            user_question: User's question
            top_k: Number of relevant documents to retrieve
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Agent response with answer, sources, and KOL data
        """
        logger.info(f"Processing Linkol query: {user_question[:100]}...")
        
        # Get LLM client (OpenRouter if api_key provided, otherwise AIHubMix)
        llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
        # Select model based on API provider (same logic as v1)
        model_name = settings.openrouter_model if api_key else self.chat_model
        
        question_lower = user_question.lower()
        
        # Check for valuation keywords
        valuation_keywords = [
            "估值", "价格", "价钱", "多少钱", "值多少", "valuation", 
            "price", "pricing", "cost", "worth", "值", "定价"
        ]
        
        is_valuation_query = any(keyword in question_lower for keyword in valuation_keywords)
        
        # Check for specific user valuation
        if is_valuation_query:
            screen_name = self._extract_screen_name(user_question)
            
            if screen_name:
                # Get price for specific screen name
                logger.info(f"Fetching price for @{screen_name}...")
                try:
                    price_result = await self.linkol_service.get_kol_price(screen_name=screen_name)
                    
                    if price_result.get("code") == 200:
                        price = price_result.get("data", {}).get("price")
                        answer = f"@{screen_name} 的当前估值是 **${price:.2f}**\n\n这个估值是基于该 KOL 最近 20 篇原创推文的数据计算得出的。"
                        
                        return {
                            "answer": answer,
                            "sources": [],
                            "kol_data": {"screen_name": screen_name, "price": price},
                            "num_sources": 0
                        }
                    else:
                        return {
                            "answer": f"抱歉，无法获取 @{screen_name} 的估值信息。可能是该用户不在系统中或数据不可用。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                except Exception as e:
                    logger.error(f"Error getting KOL price: {e}")
                    return {
                        "answer": f"抱歉，获取 @{screen_name} 的估值信息时发生错误。请稍后再试。",
                        "sources": [],
                        "kol_data": None,
                        "num_sources": 0
                    }
            else:
                # General valuation query - get top KOLs
                logger.info("Processing general valuation query - getting top KOLs")
                try:
                    hot_kols_result = await self.linkol_service.get_hot_kols()
                    
                    if hot_kols_result.get("code") != 200:
                        return {
                            "answer": "抱歉，无法获取 KOL 数据。请稍后再试。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                    
                    kols_list = hot_kols_result.get("data", {}).get("list", [])
                    if not kols_list:
                        return {
                            "answer": "目前没有可用的 KOL 数据。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                    
                    # Get valuations for top 5-10 KOLs
                    top_kols_with_prices = []
                    max_kols = min(10, len(kols_list))
                    
                    for i, kol in enumerate(kols_list[:max_kols], 1):
                        screen_name = kol.get("screen_name")
                        if screen_name:
                            try:
                                price_result = await self.linkol_service.get_kol_price(screen_name=screen_name)
                                if price_result.get("code") == 200:
                                    price = price_result.get("data", {}).get("price")
                                    kol_info = {
                                        "rank": i,
                                        "screen_name": screen_name,
                                        "name": kol.get("name", "N/A"),
                                        "followers": kol.get("followers_count", 0),
                                        "price": price
                                    }
                                    top_kols_with_prices.append(kol_info)
                            except Exception as e:
                                logger.warning(f"Failed to get price for @{screen_name}: {e}")
                                continue
                    
                    if not top_kols_with_prices:
                        return {
                            "answer": "抱歉，无法获取 KOL 的估值信息。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                    
                    # Format answer
                    answer = "以下是部分热门 KOL 的估值信息：\n\n"
                    for kol_info in top_kols_with_prices:
                        answer += f"{kol_info['rank']}. @{kol_info['screen_name']} ({kol_info['name']})\n"
                        answer += f"   粉丝数: {kol_info['followers']:,}\n"
                        answer += f"   估值: **${kol_info['price']:.2f}**\n\n"
                    
                    answer += "注：估值是基于每位 KOL 最近 20 篇原创推文的数据计算得出的。"
                    
                    return {
                        "answer": answer,
                        "sources": [],
                        "kol_data": {"kols": top_kols_with_prices},
                        "num_sources": 0
                    }
                except Exception as e:
                    logger.error(f"Error processing general valuation: {e}")
                    raise
        
        # For other queries, search content and tweets using RAG
        relevant_content = self._search_linkol_content(user_question, top_k=top_k)
        relevant_tweets = self._search_tweets(user_question, top_k=top_k)
        
        sources = []
        
        # Format sources from content
        for item in relevant_content[:top_k]:
            source_info = {
                "type": item.get("content_type", "unknown"),
                "content": item.get("content", "")[:200] + "..." if len(item.get("content", "")) > 200 else item.get("content", ""),
                "score": item.get("score", 0)
            }
            if item.get("title"):
                source_info["title"] = item.get("title")
            if item.get("author"):
                source_info["author"] = item.get("author")
            if item.get("source_url"):
                source_info["url"] = item.get("source_url")
            sources.append(source_info)
        
        # Add tweets to sources
        for tweet in relevant_tweets:
            source_info = {
                "type": "tweet",
                "content": tweet.get("text", "")[:200] + "..." if len(tweet.get("text", "")) > 200 else tweet.get("text", ""),
                "author": tweet.get("author", ""),
                "created_at": tweet.get("created_at", ""),
                "score": tweet.get("score", 0)
            }
            sources.append(source_info)
        
        # Get top KOL data for context
        kol_data = await self._get_top_kol_price()
        
        # Build context
        content_context = ""
        if relevant_content:
            content_context = "\n\nRelevant Linkol content:\n"
            for i, item in enumerate(relevant_content[:top_k], 1):
                content_type = item.get("content_type", "content")
                content_text = item.get("content", "")[:300]
                if item.get("title"):
                    content_context += f"\n[{i}] {item['title']} ({content_type}): {content_text}\n"
                else:
                    content_context += f"\n[{i}] ({content_type}): {content_text}\n"
        
        tweets_context = ""
        if relevant_tweets:
            tweets_context = "\n\nRelevant tweets from Twitter:\n"
            for i, tweet in enumerate(relevant_tweets[:top_k], 1):
                tweet_text = tweet.get("text", "")[:300]
                author = tweet.get("author", "Unknown")
                created_at = tweet.get("created_at", "")
                tweets_context += f"\n[{i}] @{author} ({created_at}): {tweet_text}\n"
        
        kol_context = ""
        if kol_data:
            kol = kol_data.get("kol", {})
            price = kol_data.get("price")
            kol_context = f"""

Top-ranked KOL Information:
- Name: {kol.get('name', 'N/A')} (@{kol_data.get('screen_name', 'N/A')})
- Followers: {kol.get('followers_count', 0):,}
- Total Tweets: {kol.get('total_tweet_count', 0):,}
- Description: {kol.get('description', 'N/A')}
- Current Price: ${price:.2f} (based on last 20 original tweets)
"""
        
        # Generate answer using LLM
        prompt = f"""Introduce Linkol to the user based on the following information.{content_context}{tweets_context}{kol_context}

User question: {user_question}

Please introduce Linkol in a professional yet friendly tone. Focus on providing clear, informative explanations about what Linkol is and how it works. Incorporate the relevant content, tweets, and KOL pricing information naturally into your introduction."""

        logger.debug(f"Generating LLM response (model: {model_name})")
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable Linkol introducer who helps users learn about Linkol. Your tone is professional yet friendly, clear and informative. You focus on explaining what Linkol is, how it works, and its key features. If users ask about your identity, identify yourself as a Linkol introducer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300  # Limit to ~200 English words
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response (length: {len(answer)})")
            
            return {
                "answer": answer,
                "sources": sources,
                "kol_data": kol_data,
                "num_sources": len(sources)
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

