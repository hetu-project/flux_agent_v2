"""Linkol Agent for querying Linkol-related information and KOL data."""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.config import get_settings
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.services.linkol import LinkolService
from src.repositories.project_content_repository import ProjectContentRepository
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class LinkolAgent:
    """Linkol Agent for querying Linkol-related information and KOL data."""
    
    def __init__(
        self,
        project_content_repo: Optional[ProjectContentRepository] = None
    ):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.project_content_repo = project_content_repo
        self.linkol_service = LinkolService()
        
        # Initialize LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = settings.chat_model
    
    def _is_linkol_related(self, user_question: str) -> bool:
        """
        Analyze if user's question is related to Linkol.
        
        Args:
            user_question: User's question
            
        Returns:
            True if related to Linkol, False otherwise
        """
        logger.debug(f"Analyzing Linkol intent: {user_question[:50]}...")
        
        # Keywords that indicate Linkol-related queries
        linkol_keywords = [
            "linkol", "kol", "influencer", "twitter influencer",
            "推文价格", "kol价格", "热门kol", "top kol",
            "influence", "social media", "twitter user"
        ]
        
        question_lower = user_question.lower()
        
        # Check if question contains Linkol keywords
        for keyword in linkol_keywords:
            if keyword in question_lower:
                logger.info(f"Detected Linkol-related query (keyword: {keyword})")
                return True
        
        # Use LLM for more nuanced intent detection
        try:
            prompt = f"""Analyze if the following user question is related to Linkol, KOL (Key Opinion Leader), Twitter influencers, or social media influencer pricing.

User question: {user_question}

Respond with only "yes" or "no"."""
            
            response = self.llm.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes user intent. Respond with only 'yes' or 'no'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().lower()
            is_related = "yes" in answer
            
            if is_related:
                logger.info("LLM detected Linkol-related query")
            else:
                logger.debug("LLM determined query is not Linkol-related")
            
            return is_related
        except Exception as e:
            logger.error(f"Error in LLM intent analysis: {e}")
            # Fallback to keyword matching
            return False
    
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
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query the Linkol agent.
        
        Logic:
        1. Analyze if user's question is related to Linkol
        2. If related, search for Linkol-related content in project_content
        3. Get top-ranked KOL and their price from Linkol API
        4. Generate response combining content and KOL data
        
        Args:
            user_question: User's question
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Agent response with answer, sources, and KOL data
        """
        logger.info(f"Processing Linkol query: {user_question[:100]}...")
        
        # Part 1: Intent analysis
        is_linkol_related = self._is_linkol_related(user_question)
        
        # If not related to Linkol, directly use LLM to answer
        if not is_linkol_related:
            logger.info("Query is not Linkol-related, using LLM to answer directly")
            try:
                response = self.llm.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Provide clear and informative answers to user questions."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                answer = response.choices[0].message.content
                logger.info(f"Generated LLM response (length: {len(answer)})")
                
                return {
                    "answer": answer,
                    "sources": [],
                    "kol_data": None,
                    "num_sources": 0
                }
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                raise
        
        # Search for Linkol-related content
        relevant_content = []
        sources = []
        
        if self.project_content_repo:
            relevant_content = self._search_linkol_content(user_question, top_k=top_k)
            
            # Format sources
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
        
        # Part 2: Get top KOL price from Linkol API
        logger.info("Fetching top KOL data from Linkol API...")
        kol_data = await self._get_top_kol_price()
        
        # Build context for LLM
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
        
        kol_context = ""
        if kol_data:
            kol = kol_data.get("kol", {})
            price = kol_data.get("price")
            kol_context = f"""

Top-ranked KOL Information:
- Name: {kol.get('name', 'N/A')} (@{kol_data.get('screen_name', 'N/A')})
- Followers: {kol.get('followers_count', 0):,}
- Total Tweets: {kol.get('total_tweet_count', 0):,}
- Total Likes: {kol.get('like_count', 0):,}
- Description: {kol.get('description', 'N/A')}
- Current Price: ${price:.2f} (based on last 20 original tweets)
"""
        
        # Generate answer using LLM
        prompt = f"""Introduce Linkol to the user based on the following information.{content_context}{kol_context}

User question: {user_question}

Please introduce Linkol in a professional yet friendly tone. Focus on providing clear, informative explanations about what Linkol is and how it works. Your tone should be approachable and easy to understand, but maintain professionalism. Think of yourself as a knowledgeable guide introducing an interesting project to someone who wants to learn about it. Incorporate the relevant content and KOL pricing information naturally into your introduction."""

        logger.debug(f"Generating LLM response (model: {self.chat_model})")
        try:
            response = self.llm.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable project introducer who introduces Linkol to users. Your tone is professional yet friendly, clear and informative. You focus on explaining what Linkol is, how it works, and its key features. You speak in a way that is approachable and easy to understand, but you maintain a professional demeanor. You are enthusiastic about the project but not overly casual."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
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