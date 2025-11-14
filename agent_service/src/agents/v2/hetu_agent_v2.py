"""Hetu Agent V2 for querying Hetu Protocol information with cheaper models and RAG-based tweet search."""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.config import get_settings
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.repositories.project_repository import ProjectRepository
from src.repositories.project_content_repository import ProjectContentRepository
from src.repositories.tweet_repository import TweetRepository
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HetuAgentV2:
    """Hetu Agent V2 specialized in answering questions about Hetu Protocol with cheaper models and RAG-based tweet search."""
    
    PROJECT_NAME = "Hetu Protocol"  # Fixed project name
    
    def __init__(
        self,
        project_repo: Optional[ProjectRepository] = None,
        project_content_repo: Optional[ProjectContentRepository] = None,
        tweet_repo: Optional[TweetRepository] = None
    ):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.project_repo = project_repo
        self.project_content_repo = project_content_repo
        self.tweet_repo = tweet_repo
        
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        # Use cheaper model: gpt-4o-mini instead of grok
        self.chat_model = "gpt-4o-mini"
    
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
    
    def _get_hetu_project(self) -> Optional[Dict[str, Any]]:
        """
        Get Hetu Protocol project from database.
        
        Returns:
            Project dict with name and description if found, None otherwise
        """
        if not self.project_repo:
            logger.debug("Project repository not available")
            return None
        
        try:
            logger.debug(f"Getting project: {self.PROJECT_NAME}")
            project_obj = self.project_repo.get_by_name(self.PROJECT_NAME)
            
            if project_obj:
                project = {
                    "name": project_obj.name,
                    "description": project_obj.description
                }
                logger.info(f"Found project: {self.PROJECT_NAME}")
                return project
            else:
                logger.warning(f"Project '{self.PROJECT_NAME}' not found in database")
                return None
        except Exception as e:
            logger.error(f"Error getting Hetu project: {e}")
            return None
    
    def _search_tweets(self, query: str, project: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search tweets using RAG (vector similarity search).
        
        Args:
            query: Search query
            project: Optional project filter
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
            
            # Search tweets
            results = self.tweet_repo.search(
                query_vector=query_vector,
                project=project,
                top_k=top_k,
                min_score=0.6  # Minimum similarity threshold
            )
            
            logger.info(f"Found {len(results)} relevant tweets for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error searching tweets: {e}", exc_info=True)
            return []
    
    async def query(
        self,
        user_question: str,
        top_k: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the agent with a question about Hetu Protocol.
        
        Logic:
        1. Always use Hetu Protocol project information
        2. Search for relevant tweets using RAG
        3. Search for relevant content related to Hetu Protocol
        4. Include project info, tweets, and content in the prompt
        5. Use cheaper LLM model to generate answer
        
        Args:
            user_question: User's question
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Agent response with answer and sources
        """
        logger.info(f"Processing Hetu Protocol query: {user_question[:100]}...")
        
        # 1. Always get Hetu Protocol project
        found_project = self._get_hetu_project()
        
        # 2. Search for relevant tweets using RAG
        relevant_tweets = self._search_tweets(
            query=user_question,
            project=self.PROJECT_NAME,
            top_k=top_k
        )
        
        # 3. Search for relevant content from project_content collection
        sources = []
        relevant_content = []
        
        if found_project and self.project_content_repo:
            # Search for relevant content related to Hetu Protocol
            logger.debug(f"Searching for relevant content for project '{self.PROJECT_NAME}'")
            relevant_content = self.project_content_repo.search(
                query=user_question,
                project_name=self.PROJECT_NAME,
                top_k=top_k,
                min_score=0.6  # Minimum similarity threshold
            )
            
            # Format sources from relevant content
            for item in relevant_content[:top_k]:  # Limit to top_k
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
            
            logger.info(f"Found {len(sources)} relevant content items for project '{self.PROJECT_NAME}'")
        
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
        
        # 4. Build prompt with Hetu Protocol information, tweets, and content
        if found_project:
            # Include project information in the prompt
            project_info = f"Project name: {found_project['name']}"
            if found_project.get('description'):
                project_info += f"\nProject description: {found_project['description']}"
            
            # Add relevant content context if available
            content_context = ""
            if relevant_content:
                content_context = "\n\nRelevant content from papers and other sources:\n"
                for i, item in enumerate(relevant_content[:top_k], 1):
                    content_type = item.get("content_type", "content")
                    content_text = item.get("content", "")[:300]  # Limit length
                    if item.get("title"):
                        content_context += f"\n[{i}] {item['title']} ({content_type}): {content_text}\n"
                    else:
                        content_context += f"\n[{i}] ({content_type}): {content_text}\n"
            
            # Add relevant tweets context
            tweets_context = ""
            if relevant_tweets:
                tweets_context = "\n\nRelevant tweets from Twitter:\n"
                for i, tweet in enumerate(relevant_tweets[:top_k], 1):
                    tweet_text = tweet.get("text", "")[:300]
                    author = tweet.get("author", "Unknown")
                    created_at = tweet.get("created_at", "")
                    tweets_context += f"\n[{i}] @{author} ({created_at}): {tweet_text}\n"
            
            prompt = f"""Answer the user's question about Hetu Protocol based on the following project information, relevant content, and tweets.

{project_info}{content_context}{tweets_context}

User question: {user_question}

Please provide a helpful answer about Hetu Protocol based on the project info, relevant content, and tweets above. If the information is insufficient, please say so."""
        else:
            # Add relevant tweets context even if project not found
            tweets_context = ""
            if relevant_tweets:
                tweets_context = "\n\nRelevant tweets from Twitter:\n"
                for i, tweet in enumerate(relevant_tweets[:top_k], 1):
                    tweet_text = tweet.get("text", "")[:300]
                    author = tweet.get("author", "Unknown")
                    created_at = tweet.get("created_at", "")
                    tweets_context += f"\n[{i}] @{author} ({created_at}): {tweet_text}\n"
            
            prompt = f"""Answer the user's question about Hetu Protocol based on the following relevant tweets.{tweets_context}

User question: {user_question}

Please provide a helpful answer about Hetu Protocol based on the tweets above."""
        
        # 5. Generate answer using cheaper LLM model
        # Select model based on API provider (same logic as v1)
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating LLM response (model: {model_name})")
        try:
            # Get LLM client (OpenRouter if api_key provided, otherwise AIHubMix)
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable Hetu Protocol introducer who helps users learn about Hetu Protocol. Your role is to introduce and explain Hetu Protocol in a professional yet friendly manner. You focus on explaining what Hetu Protocol is, how it works, its key features, technical details, and applications. Your tone is clear, informative, and approachable. You are enthusiastic about Hetu Protocol but maintain professionalism. Always prioritize providing accurate information about Hetu Protocol based on the provided context."},
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
                "num_sources": len(sources)
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise