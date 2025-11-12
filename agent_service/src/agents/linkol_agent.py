"""Linkol Agent for querying Linkol-related information and KOL data."""

from typing import List, Dict, Any, Optional
import httpx
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
        
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = settings.chat_model
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
    
    def _get_all_project_names(self) -> List[str]:
        """
        Get all unique project names from the database.
        Caches the result to avoid repeated queries.
        
        Returns:
            List of project names (excluding "Linkol")
        """
        # Return cached result if available
        if self._cached_project_names is not None:
            return self._cached_project_names
        
        try:
            if not self.project_content_repo:
                logger.debug("Project content repository not available, cannot get project names")
                self._cached_project_names = []
                return []
            
            # Use Qdrant client directly to get all unique project names
            collection_name = "project_content"
            
            # Scroll through all points to get unique project names
            project_names = set()
            offset = None
            
            while True:
                result = self.qdrant.client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset
                )
                
                points, next_offset = result
                
                if not points:
                    break
                
                # Extract unique project names
                for point in points:
                    payload = point.payload
                    project_name = payload.get("project_name")
                    if project_name and project_name != "Linkol":
                        project_names.add(project_name)
                
                # Check if we've reached the end
                if next_offset is None:
                    break
                offset = next_offset
            
            project_names_list = list(project_names)
            self._cached_project_names = project_names_list
            logger.debug(f"Found {len(project_names_list)} unique project names: {project_names_list}")
            return project_names_list
            
        except Exception as e:
            logger.error(f"Error getting project names: {e}")
            self._cached_project_names = []
            return []
    
    def _check_for_other_projects(self, user_question: str) -> Optional[str]:
        """
        Check if user's question mentions other projects (excluding Linkol) using vector similarity search.
        
        Uses vector embedding to find semantically similar projects rather than simple string matching.
        
        Args:
            user_question: User's question
            
        Returns:
            Name of mentioned project if found with high similarity, None otherwise
        """
        try:
            # Use vector similarity search instead of string matching
            # Search in project_content collection to find projects with high semantic similarity
            
            if not self.project_content_repo:
                logger.debug("Project content repository not available")
                return None
            
            # Get all unique project names (excluding Linkol) using vector search
            # Search for content that matches the user question semantically
            # We'll search with a filter to exclude Linkol and see what other projects match
            
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
            
            # Get all project names first
            project_names = self._get_all_project_names()
            if not project_names:
                logger.debug("No other projects found in database")
                return None
            
            # Use vector search to find the most semantically similar project content
            # Search across all projects except Linkol
            try:
                # Generate embedding for user question
                query_vector = self.embedding.embed_text(user_question)
                
                # Create filter to exclude Linkol and only search other projects
                filter_query = Filter(
                    must=[
                        FieldCondition(
                            key="project_name",
                            match=MatchAny(any=project_names)  # Only search non-Linkol projects
                        )
                    ]
                )
                
                # Search for most similar content
                results = self.qdrant.search(
                    collection_name="project_content",
                    query_vector=query_vector,
                    limit=5,  # Top 5 results
                    filter_query=filter_query
                )
                
                if not results:
                    logger.debug("No similar project content found for user question")
                    return None
                
                # Filter results by minimum similarity threshold (0.7 for cosine similarity)
                min_score = 0.7
                filtered_results = [r for r in results if r.score >= min_score]
                
                if not filtered_results:
                    logger.debug(f"Top result score: {results[0].score:.3f} (below threshold {min_score})")
                    return None
                
                # Get the top result and check its project name
                top_result = filtered_results[0]
                top_project = top_result.payload.get("project_name")
                
                # Only return if project is not Linkol (already filtered out by filter_query, but double-check)
                if top_project and top_project != "Linkol":
                    logger.info(f"Detected semantically similar project: {top_project} (score: {top_result.score:.3f})")
                    return top_project
                
                logger.debug(f"Top result project: {top_project} (ignored)")
                return None
                
            except Exception as e:
                logger.error(f"Error in vector search for projects: {e}")
                # Fallback to string matching if vector search fails
                if project_names:
                    return self._check_for_other_projects_fallback(user_question, list(project_names))
                return None
            
        except Exception as e:
            logger.error(f"Error checking for other projects: {e}")
            return None
    
    def _check_for_other_projects_fallback(self, user_question: str, project_names: List[str]) -> Optional[str]:
        """
        Fallback method using simple string matching if vector search fails.
        
        Args:
            user_question: User's question
            project_names: List of project names to check
            
        Returns:
            Name of mentioned project if found, None otherwise
        """
        question_lower = user_question.lower()
        
        for project_name in project_names:
            if project_name.lower() in question_lower:
                logger.info(f"Detected mention of other project (fallback): {project_name}")
                return project_name
        
        return None
    
    def _has_valuation_intent_for_user(self, user_question: str, llm_client: Optional[OpenAI] = None, model_name: Optional[str] = None) -> bool:
        """
        Check if user's question has intent to get valuation for a specific user.
        
        Args:
            user_question: User's question
            
        Returns:
            True if user wants to get valuation for a specific user, False otherwise
        """
        question_lower = user_question.lower()
        
        # Valuation keywords
        valuation_keywords = [
            "估值", "价格", "价钱", "多少钱", "值多少", "valuation", 
            "price", "pricing", "cost", "worth", "值", "定价"
        ]
        
        # Check if question contains valuation keywords
        has_valuation = any(keyword in question_lower for keyword in valuation_keywords)
        
        if not has_valuation:
            return False
        
        # Use LLM to determine if user wants valuation for a specific user
        try:
            llm = llm_client if llm_client else self.llm
            model = model_name if model_name else self.chat_model
            prompt = f"""Analyze if the user's question is asking for valuation/price of a SPECIFIC Twitter user (not general valuation).

User question: {user_question}

Respond with only "yes" if the user is asking for a specific user's valuation, or "no" if they're asking for general valuation information.

Examples:
- "@username 的估值" -> yes
- "vis_eth 的价格" -> yes  
- "KOL估值是多少" -> no
- "热门KOL的估值" -> no"""

            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes user intent. Respond with only 'yes' or 'no'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().lower()
            has_specific_intent = "yes" in answer
            
            logger.debug(f"LLM determined valuation intent for specific user: {has_specific_intent}")
            return has_specific_intent
        except Exception as e:
            logger.error(f"Error in LLM valuation intent check: {e}")
            # Fallback: assume true if has valuation keywords
            return has_valuation
    
    def _extract_screen_name(self, user_question: str, llm_client: Optional[OpenAI] = None, model_name: Optional[str] = None) -> Optional[str]:
        """
        Extract Twitter screen name from user question.
        First tries regex, then falls back to LLM extraction if regex fails.
        
        Args:
            user_question: User's question
            
        Returns:
            Screen name (without @) if found, None otherwise
        """
        import re
        
        # Step 1: Try regex extraction first
        # Pattern to match @username (only alphanumeric and underscore, stop at non-word characters)
        pattern = r'@([a-zA-Z0-9_]+)'
        matches = re.findall(pattern, user_question)
        
        if matches:
            screen_name = matches[0]
            logger.debug(f"Extracted screen name from @: {screen_name}")
            return screen_name
        
        # Check if question mentions a specific username pattern without @
        # e.g., "vis_eth 的估值", "dada81505550664的价格", "为dada81505550664估价"
        question_lower = user_question.lower()
        
        # Try to find username pattern before valuation keywords
        # Match patterns like: "username 的估值", "username 的价格", "为username估价"
        username_patterns = [
            r'([a-zA-Z0-9_]+)\s*[的]?\s*(?:估值|价格|价钱|price|valuation|值多少钱)',  # "username 的估值"
            r'为\s*([a-zA-Z0-9_]+)\s*(?:估值|估价|价格|价钱)',  # "为username估价"
            r'(?:给|为|查)\s*([a-zA-Z0-9_]+)\s*(?:的)?\s*(?:估值|估价|价格|价钱)',  # "给username估价"
        ]
        
        for username_pattern in username_patterns:
            match = re.search(username_pattern, user_question, re.IGNORECASE)
            if match:
                potential_name = match.group(1)
                if len(potential_name) > 2 and re.match(r'^[a-zA-Z0-9_]+$', potential_name):
                    logger.debug(f"Extracted potential screen name from text: {potential_name}")
                    return potential_name
        
        # Fallback regex: try to extract from beginning of question before valuation keywords
        valuation_keywords = ["估值", "价格", "价钱", "price", "valuation", "值多少钱", "现在的估值"]
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
        
        # Step 2: If regex failed, check if user has valuation intent for a specific user
        has_specific_intent = self._has_valuation_intent_for_user(user_question, llm_client=llm_client, model_name=model_name)
        if not has_specific_intent:
            logger.debug("No specific user valuation intent detected, skipping LLM extraction")
            return None
        
        # Step 3: Use LLM to extract screen name
        logger.info("Regex extraction failed, using LLM to extract screen name")
        try:
            llm = llm_client if llm_client else self.llm
            model = model_name if model_name else self.chat_model
            prompt = f"""Extract the Twitter username (screen name) from the user's question. 
The username should be without @ symbol. Return only the username, nothing else.

User question: {user_question}

Examples:
- "你知道@vis_eth现在的估值吗？" -> vis_eth
- "vis_eth 的估值是多少" -> vis_eth
- "我想知道 dada81505550664 的价格" -> dada81505550664
- "KOL估值是多少" -> (no username, return empty)

Return only the username if found, or return "none" if no username is mentioned."""

            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts Twitter usernames. Return only the username (without @) or 'none' if no username is found."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Clean the answer - remove quotes, whitespace, and check if it's valid
            answer = answer.strip('"\'`').strip()
            
            # Check if LLM said "none" or similar
            if answer.lower() in ["none", "no", "n/a", "not found", ""]:
                logger.debug("LLM determined no username found")
                return None
            
            # Validate extracted username (alphanumeric + underscore, reasonable length)
            if re.match(r'^[a-zA-Z0-9_]+$', answer) and 2 <= len(answer) <= 50:
                logger.info(f"LLM extracted screen name: {answer}")
                return answer
            else:
                logger.warning(f"LLM extracted invalid username format: {answer}")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM screen name extraction: {e}")
            return None
        
        return None
    
    def _classify_intent(self, user_question: str, llm_client: Optional[OpenAI] = None, model_name: Optional[str] = None) -> tuple[str, Optional[str]]:
        """
        Classify user's question intent for Linkol-related queries.
        
        Args:
            user_question: User's question
            
        Returns:
            Tuple of (intent_type, screen_name)
            Intent type: "valuation_specific", "valuation_general", "data_query", "introduction", "action", "general", or "unrelated"
            Screen name: Extracted Twitter screen name if found, None otherwise
        """
        logger.debug(f"Classifying intent: {user_question[:50]}...")
        
        question_lower = user_question.lower()
        
        # Check if related to Linkol first
        is_related = self._is_linkol_related(user_question, llm_client=llm_client, model_name=model_name)
        if not is_related:
            return ("unrelated", None)
        
        # Valuation keywords
        valuation_keywords = [
            "估值", "价格", "价钱", "多少钱", "值多少", "valuation", 
            "price", "pricing", "cost", "worth", "值", "定价"
        ]
        
        # Check for valuation intent first
        is_valuation_query = any(keyword in question_lower for keyword in valuation_keywords)
        
        if is_valuation_query:
            # Try to extract specific screen name FIRST before classifying
            screen_name = self._extract_screen_name(user_question, llm_client=llm_client, model_name=model_name)
            
            if screen_name:
                logger.info(f"Detected valuation_specific intent for @{screen_name}")
                return ("valuation_specific", screen_name)
            else:
                # If no screen name extracted, double-check with intent analysis
                # This helps distinguish between general valuation queries and specific ones we couldn't extract
                has_specific_intent = self._has_valuation_intent_for_user(user_question, llm_client=llm_client, model_name=model_name)
                if has_specific_intent:
                    # User wants specific user but we couldn't extract - try LLM extraction again
                    logger.info("Detected specific user valuation intent but regex failed, will use LLM extraction in query handler")
                    return ("valuation_specific", None)  # Will be extracted again in query handler
                else:
                    logger.info("Detected valuation_general intent")
                    return ("valuation_general", None)
        
        # Data query keywords (need API data)
        data_query_keywords = [
            "top", "名单", "list", "排名", "rank", "热门", "hot",
            "top kol", "热门kol", "top 20", "top kol list"
        ]
        
        # Action/How-to keywords (need search)
        action_keywords = [
            "如何", "怎么", "怎样", "how to", "how do", "way to",
            "加入", "join", "使用", "use", "注册", "register",
            "开始", "start", "begin", "tutorial", "guide", "步骤"
        ]
        
        # Introduction keywords
        intro_keywords = [
            "什么", "什么是", "what is", "介绍", "introduce", "explain",
            "了解", "learn about", "tell me about"
        ]
        
        # Check for data query intent
        for keyword in data_query_keywords:
            if keyword in question_lower:
                logger.info(f"Detected data_query intent (keyword: {keyword})")
                return ("data_query", None)
        
        # Check for action intent
        for keyword in action_keywords:
            if keyword in question_lower:
                logger.info(f"Detected action intent (keyword: {keyword})")
                return ("action", None)
        
        # Check for introduction intent
        for keyword in intro_keywords:
            if keyword in question_lower:
                logger.info(f"Detected introduction intent (keyword: {keyword})")
                return ("introduction", None)
        
        # Use LLM for more nuanced classification
        try:
            llm = llm_client if llm_client else self.llm
            model = model_name if model_name else self.chat_model
            prompt = f"""Classify the user's question about Linkol into one of these categories:
1. "valuation_specific" - User asks about valuation/price of a specific Twitter user (e.g., "@username 的估值", "what is @user's price")
2. "valuation_general" - User asks about valuations/prices in general (e.g., "KOL估值是多少", "how much do KOLs cost")
3. "data_query" - User wants specific data/numbers/lists (e.g., "top KOL list", "hot KOLs")
4. "action" - User wants to know how to do something (e.g., "how to join", "how to use Linkol")
5. "introduction" - User wants to learn about Linkol (e.g., "what is Linkol", "introduce Linkol")
6. "general" - General Linkol-related question that doesn't fit above

User question: {user_question}

Respond with only one word: valuation_specific, valuation_general, data_query, action, introduction, or general."""
            
            response = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies user intent. Respond with only one word: valuation_specific, valuation_general, data_query, action, introduction, or general."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=20
            )
            
            answer = response.choices[0].message.content.strip().lower()
            intent = None
            
            for intent_type in ["valuation_specific", "valuation_general", "data_query", "action", "introduction", "general"]:
                if intent_type in answer:
                    intent = intent_type
                    break
            
            if intent:
                logger.info(f"LLM classified intent: {intent}")
                # If valuation intent, try to extract screen name
                if intent == "valuation_specific" or "valuation" in answer:
                    screen_name = self._extract_screen_name(user_question, llm_client=llm_client, model_name=model_name)
                    if screen_name:
                        return (intent, screen_name) if intent == "valuation_specific" else ("valuation_specific", screen_name)
                    else:
                        return ("valuation_general", None)
                return (intent, None)
            else:
                logger.debug("LLM classification unclear, defaulting to general")
                return ("general", None)
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {e}")
            return ("general", None)
    
    def _is_linkol_related(self, user_question: str, llm_client: Optional[OpenAI] = None, model_name: Optional[str] = None) -> bool:
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
            llm = llm_client if llm_client else self.llm
            model = model_name if model_name else self.chat_model
            prompt = f"""Analyze if the following user question is related to Linkol, KOL (Key Opinion Leader), Twitter influencers, or social media influencer pricing.

User question: {user_question}

Respond with only "yes" or "no"."""
            
            response = llm.chat.completions.create(
                model=model,
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
        top_k: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the Linkol agent with intelligent intent classification.
        
        Logic:
        1. Classify user intent (data_query, action, introduction, general, unrelated)
        2. For data_query: Directly call API and return concise data
        3. For action: Search database first, if no results, use LLM's Twitter search capability
        4. For introduction/general: Search database + API + comprehensive introduction
        5. For unrelated: Direct LLM answer with Linkol assistant identity
        
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
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        
        # Check if user's question mentions other projects using vector similarity
        other_project = self._check_for_other_projects(user_question)
        if other_project:
            logger.info(f"User question mentions other project: {other_project}")
            return {
                "answer": f"Questions about {other_project} should be directed to our parallel universe agent. Please visit the parallel universe agent for queries regarding {other_project}.",
                "sources": [],
                "kol_data": None,
                "num_sources": 0
            }
        
        # Part 1: Intent classification
        intent, screen_name = self._classify_intent(user_question, llm_client=llm_client, model_name=model_name)
        logger.info(f"Classified intent: {intent}, screen_name: {screen_name}")
        
        # If not related to Linkol, directly use LLM to answer
        if intent == "unrelated":
            logger.info("Query is not Linkol-related, using LLM to answer directly")
            try:
                # Build prompt with identity handling
                prompt = user_question
                
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for Linkol. You are a Linkol introducer, but you are also happy to answer other questions. If the user asks about your identity or who you are, you should identify yourself as a Linkol introducer while expressing that you're happy to help with various questions. For other questions, just provide helpful and informative answers directly."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300  # Limit to ~200 English words
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
        
        # Handle different intent types
        if intent == "valuation_specific":
            # For specific user valuation queries, directly call API
            logger.info(f"Processing valuation_specific intent for @{screen_name}")
            
            # If screen_name is None, try to extract it using LLM
            if not screen_name:
                logger.info("Screen name not extracted yet, trying LLM extraction")
                screen_name = self._extract_screen_name(user_question, llm_client=llm_client, model_name=model_name)
                
                if not screen_name:
                    return {
                        "answer": "抱歉，无法识别您要查询的 Twitter 用户名。请使用 @用户名 或 \"用户名 的估值\" 的格式，例如：@username 的估值，或 dada81505550664 的估值。",
                        "sources": [],
                        "kol_data": None,
                        "num_sources": 0
                    }
            
            # Get price for specific screen name
            logger.info(f"Fetching price for @{screen_name}...")
            try:
                    price_result = await self.linkol_service.get_kol_price(screen_name=screen_name)
                    
                    # Check for non-200 response code
                    if price_result.get("code") != 200:
                        error_code = price_result.get("code")
                        logger.warning(f"Failed to get KOL price for @{screen_name}: code={error_code}, msg={price_result.get('msg')}")
                        
                        # If 510 error (user not found), fallback to top KOLs with valuations
                        if error_code == 510:
                            # Execute fallback logic inline
                            logger.info(f"User @{screen_name} not found (510), fetching top KOLs as fallback")
                            
                            try:
                                # Get top KOLs and their valuations as reference
                                hot_kols_result = await self.linkol_service.get_hot_kols()
                                
                                if hot_kols_result.get("code") != 200:
                                    return {
                                        "answer": f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中）。同时无法获取热门 KOL 列表作为参考。",
                                        "sources": [],
                                        "kol_data": None,
                                        "num_sources": 0
                                    }
                                
                                kols_list = hot_kols_result.get("data", {}).get("list", [])
                                if not kols_list:
                                    return {
                                        "answer": f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中）。目前暂无可用的热门 KOL 数据作为参考。",
                                        "sources": [],
                                        "kol_data": None,
                                        "num_sources": 0
                                    }
                                
                                # Get valuations for top 5-10 KOLs as reference
                                top_kols_with_prices = []
                                max_kols = min(10, len(kols_list))
                                
                                for i, kol in enumerate(kols_list[:max_kols], 1):
                                    kol_screen_name = kol.get("screen_name")
                                    if kol_screen_name:
                                        try:
                                            kol_price_result = await self.linkol_service.get_kol_price(screen_name=kol_screen_name)
                                            if kol_price_result.get("code") == 200:
                                                price = kol_price_result.get("data", {}).get("price")
                                                kol_info = {
                                                    "rank": i,
                                                    "screen_name": kol_screen_name,
                                                    "name": kol.get("name", "N/A"),
                                                    "followers": kol.get("followers_count", 0),
                                                    "price": price
                                                }
                                                top_kols_with_prices.append(kol_info)
                                        except Exception as e:
                                            logger.warning(f"Failed to get price for @{kol_screen_name}: {e}")
                                            continue
                                
                                if not top_kols_with_prices:
                                    return {
                                        "answer": f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中）。同时无法获取其他 KOL 的估值作为参考。",
                                        "sources": [],
                                        "kol_data": None,
                                        "num_sources": 0
                                    }
                                
                                # Format answer with fallback information
                                answer = f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中或数据不可用）。\n\n"
                                answer += "以下是部分热门 KOL 的估值信息供您参考：\n\n"
                                for kol_info in top_kols_with_prices:
                                    answer += f"{kol_info['rank']}. @{kol_info['screen_name']} ({kol_info['name']})\n"
                                    answer += f"   粉丝数: {kol_info['followers']:,}\n"
                                    answer += f"   估值: **${kol_info['price']:.2f}**\n\n"
                                
                                answer += "注：估值是基于每位 KOL 最近 20 篇原创推文的数据计算得出的。"
                                
                                return {
                                    "answer": answer,
                                    "sources": [],
                                    "kol_data": {"kols": top_kols_with_prices, "requested_user": screen_name, "not_found": True},
                                    "num_sources": 0
                                }
                            except Exception as e:
                                logger.error(f"Error in fallback logic: {e}")
                                return {
                                    "answer": f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中）。同时获取参考数据时发生错误。",
                                    "sources": [],
                                    "kol_data": None,
                                    "num_sources": 0
                                }
                        else:
                            # Other errors
                            return {
                                "answer": f"抱歉，无法获取 @{screen_name} 的估值信息。可能是该用户不在系统中或数据不可用。",
                                "sources": [],
                                "kol_data": None,
                                "num_sources": 0
                            }
                    else:
                        # Success - return the price
                        price = price_result.get("data", {}).get("price")
                        answer = f"@{screen_name} 的当前估值是 **${price:.2f}**\n\n这个估值是基于该 KOL 最近 20 篇原创推文的数据计算得出的。"
                        
                        return {
                            "answer": answer,
                            "sources": [],
                            "kol_data": {"screen_name": screen_name, "price": price},
                            "num_sources": 0
                        }
                
            except httpx.HTTPStatusError as e:
                # Handle HTTP status errors (including 510)
                error_code = e.response.status_code
                logger.warning(f"HTTP error getting KOL price for @{screen_name}: status_code={error_code}")
                
                # If 510 error (user not found), fallback to top KOLs with valuations
                if error_code == 510:
                    logger.info(f"User @{screen_name} not found (510), fetching top KOLs as fallback")
                    
                    # Get top KOLs and their valuations as reference
                    hot_kols_result = await self.linkol_service.get_hot_kols()
                    
                    if hot_kols_result.get("code") != 200:
                        return {
                            "answer": f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中）。同时无法获取热门 KOL 列表作为参考。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                    
                    kols_list = hot_kols_result.get("data", {}).get("list", [])
                    if not kols_list:
                        return {
                            "answer": f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中）。目前暂无可用的热门 KOL 数据作为参考。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                    
                    # Get valuations for top 5-10 KOLs as reference
                    top_kols_with_prices = []
                    max_kols = min(10, len(kols_list))
                    
                    for i, kol in enumerate(kols_list[:max_kols], 1):
                        kol_screen_name = kol.get("screen_name")
                        if kol_screen_name:
                            try:
                                kol_price_result = await self.linkol_service.get_kol_price(screen_name=kol_screen_name)
                                if kol_price_result.get("code") == 200:
                                    price = kol_price_result.get("data", {}).get("price")
                                    kol_info = {
                                        "rank": i,
                                        "screen_name": kol_screen_name,
                                        "name": kol.get("name", "N/A"),
                                        "followers": kol.get("followers_count", 0),
                                        "price": price
                                    }
                                    top_kols_with_prices.append(kol_info)
                            except Exception as e:
                                logger.warning(f"Failed to get price for @{kol_screen_name}: {e}")
                                continue
                    
                    if not top_kols_with_prices:
                        return {
                            "answer": f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中）。同时无法获取其他 KOL 的估值作为参考。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                    
                    # Format answer with fallback information
                    answer = f"抱歉，无法获取 @{screen_name} 的估值信息（该用户不在系统中或数据不可用）。\n\n"
                    answer += "以下是部分热门 KOL 的估值信息供您参考：\n\n"
                    for kol_info in top_kols_with_prices:
                        answer += f"{kol_info['rank']}. @{kol_info['screen_name']} ({kol_info['name']})\n"
                        answer += f"   粉丝数: {kol_info['followers']:,}\n"
                        answer += f"   估值: **${kol_info['price']:.2f}**\n\n"
                    
                    answer += "注：估值是基于每位 KOL 最近 20 篇原创推文的数据计算得出的。"
                    
                    return {
                        "answer": answer,
                        "sources": [],
                        "kol_data": {"kols": top_kols_with_prices, "requested_user": screen_name, "not_found": True},
                        "num_sources": 0
                    }
                else:
                        # Other HTTP errors
                        logger.error(f"HTTP error {error_code} getting KOL price for @{screen_name}")
                        return {
                            "answer": f"抱歉，无法获取 @{screen_name} 的估值信息。可能是该用户不在系统中或数据不可用。",
                            "sources": [],
                            "kol_data": None,
                            "num_sources": 0
                        }
                
            except httpx.RequestError as e:
                # Handle request errors (network issues, timeouts, etc.)
                logger.error(f"Request error getting KOL price for @{screen_name}: {e}")
                return {
                    "answer": f"抱歉，网络请求失败，无法获取 @{screen_name} 的估值信息。请稍后再试。",
                    "sources": [],
                    "kol_data": None,
                    "num_sources": 0
                }
                
            except Exception as e:
                # Handle any other unexpected errors
                logger.error(f"Unexpected error getting KOL price for @{screen_name}: {e}")
                return {
                    "answer": f"抱歉，获取 @{screen_name} 的估值信息时发生错误。请稍后再试。",
                    "sources": [],
                    "kol_data": None,
                    "num_sources": 0
                }
        
        elif intent == "valuation_general":
            # For general valuation queries, get top KOLs and their valuations
            logger.info("Processing valuation_general intent - getting top KOLs and valuations")
            try:
                # Step 1: Get top KOLs
                hot_kols_result = await self.linkol_service.get_hot_kols()
                
                if hot_kols_result.get("code") != 200:
                    logger.error(f"Failed to get hot KOLs: {hot_kols_result.get('msg')}")
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
                
                # Step 2: Get valuations for top 5-10 KOLs
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
                logger.error(f"Error processing valuation_general: {e}")
                raise
        
        elif intent == "data_query":
            # For data queries, directly call API and return concise data
            logger.info("Processing data_query intent - calling API directly")
            try:
                # Get hot KOLs data
                hot_kols_result = await self.linkol_service.get_hot_kols()
                
                if hot_kols_result.get("code") != 200:
                    logger.error(f"Failed to get hot KOLs: {hot_kols_result.get('msg')}")
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
                
                # Format KOL list data
                kol_list_text = "以下是 Linkol 热门 KOL 榜单：\n\n"
                for i, kol in enumerate(kols_list[:20], 1):
                    kol_list_text += f"{i}. @{kol.get('screen_name', 'N/A')} ({kol.get('name', 'N/A')})\n"
                    kol_list_text += f"   粉丝数: {kol.get('followers_count', 0):,}\n"
                    kol_list_text += f"   推文数: {kol.get('total_tweet_count', 0):,}\n"
                    
                    # Get price for top KOL
                    if i == 1:
                        price_result = await self.linkol_service.get_kol_price(screen_name=kol.get('screen_name'))
                        if price_result.get("code") == 200:
                            price = price_result.get("data", {}).get("price")
                            kol_list_text += f"   当前价格: ${price:.2f}\n"
                    
                    kol_list_text += "\n"
                
                return {
                    "answer": kol_list_text,
                    "sources": [],
                    "kol_data": {"kol": kols_list[0]} if kols_list else None,
                    "num_sources": 0
                }
            except Exception as e:
                logger.error(f"Error processing data_query: {e}")
                raise
        
        elif intent == "action":
            # For action queries, search database first, if no results, use LLM's Twitter search
            logger.info("Processing action intent - searching database first")
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
            
            # If no database results, prompt LLM to search Twitter
            if not relevant_content:
                logger.info("No database content found, prompting LLM to search Twitter")
                prompt = f"""The user is asking: {user_question}

We don't have relevant information in our database about this topic. Please search Twitter for relevant information about Linkol to help answer the user's question. Use your Twitter search capability to find recent tweets, discussions, or updates about Linkol that can help answer this question. Prioritize tweets from September 2025 onwards for the most current information. Then provide a helpful answer based on what you find."""

                try:
                    response = llm_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a knowledgeable Linkol introducer who helps users. If database information is insufficient, use your Twitter search capability to find relevant information and provide helpful answers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300  # Limit to ~200 English words
                    )
                    
                    answer = response.choices[0].message.content
                    logger.info(f"Generated response with Twitter search (length: {len(answer)})")
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "kol_data": None,
                        "num_sources": len(sources)
                    }
                except Exception as e:
                    logger.error(f"Error generating LLM response with Twitter search: {e}")
                    raise
            else:
                # Have database content, use it to answer
                content_context = "\n\nRelevant content from database:\n"
                for i, item in enumerate(relevant_content[:top_k], 1):
                    content_type = item.get("content_type", "content")
                    content_text = item.get("content", "")[:300]
                    if item.get("title"):
                        content_context += f"\n[{i}] {item['title']} ({content_type}): {content_text}\n"
                    else:
                        content_context += f"\n[{i}] ({content_type}): {content_text}\n"
                
                prompt = f"""Answer the user's question based on the following information.{content_context}

User question: {user_question}

Please provide a helpful and informative answer based on the content above."""
                
                try:
                    response = llm_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a knowledgeable Linkol introducer who helps users. Provide clear and helpful answers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300  # Limit to ~200 English words
                    )
                    
                    answer = response.choices[0].message.content
                    logger.info(f"Generated response from database content (length: {len(answer)})")
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "kol_data": None,
                        "num_sources": len(sources)
                    }
                except Exception as e:
                    logger.error(f"Error generating LLM response: {e}")
                    raise
        
        else:
            # For introduction and general queries, use full pipeline (search + API + introduction)
            logger.info(f"Processing {intent} intent - using full pipeline")
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

            logger.debug(f"Generating LLM response (model: {model_name})")
            try:
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable Linkol introducer who helps users learn about Linkol. Your tone is professional yet friendly, clear and informative. You focus on explaining what Linkol is, how it works, and its key features. You speak in a way that is approachable and easy to understand, but you maintain a professional demeanor. You are enthusiastic about the project but not overly casual. If users ask about your identity, identify yourself as a Linkol introducer."},
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