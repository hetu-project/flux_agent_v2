"""Hetu Agent for querying Hetu Protocol information."""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.config import get_settings
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.repositories.project_repository import ProjectRepository
from src.repositories.project_content_repository import ProjectContentRepository
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HetuAgent:
    """Hetu Agent specialized in answering questions about Hetu Protocol."""
    
    PROJECT_NAME = "Hetu Protocol"  # Fixed project name
    
    def __init__(
        self,
        project_repo: Optional[ProjectRepository] = None,
        project_content_repo: Optional[ProjectContentRepository] = None
    ):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.project_repo = project_repo
        self.project_content_repo = project_content_repo
        
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
        2. Search for relevant content related to Hetu Protocol
        3. Include project info and content in the prompt
        4. Use LLM to generate answer with Hetu Protocol introducer role
        
        Args:
            user_question: User's question
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Agent response with answer and sources
        """
        logger.info(f"Processing Hetu Protocol query: {user_question[:100]}...")
        
        # 1. Always get Hetu Protocol project
        found_project = self._get_hetu_project()
        
        # 2. Search for relevant content from project_content collection
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
        
        # 3. Build prompt with Hetu Protocol information
        if found_project:
            # Include project information in the prompt
            project_info = f"Project name: {found_project['name']}"
            if found_project.get('description'):
                project_info += f"\nProject description: {found_project['description']}"
            
            # Add relevant content context if available
            content_context = ""
            if relevant_content:
                content_context = "\n\nRelevant content from tweets, papers, and other sources:\n"
                for i, item in enumerate(relevant_content[:top_k], 1):
                    content_type = item.get("content_type", "content")
                    content_text = item.get("content", "")[:300]  # Limit length
                    if item.get("title"):
                        content_context += f"\n[{i}] {item['title']} ({content_type}): {content_text}\n"
                    else:
                        content_context += f"\n[{i}] ({content_type}): {content_text}\n"
            
            prompt = f"""Answer the user's question about Hetu Protocol based on the following project information and relevant content.

{project_info}{content_context}

User question: {user_question}

Please provide a helpful answer about Hetu Protocol based on the project info and relevant content above. Additionally, please search Twitter for relevant tweets, discussions, or updates about Hetu Protocol. Prioritize finding recent tweets from September 2025 onwards, as these will contain the most up-to-date information. Use your Twitter search capability to find the most relevant and recent tweets (especially from September 2025 and later) that can help answer the user's question. If the information from the database and Twitter is insufficient, please say so."""
        else:
            # If project not found in database, still answer but mention it
            prompt = f"""Answer the user's question about Hetu Protocol.

User question: {user_question}

Please provide a helpful answer about Hetu Protocol. Please search Twitter for relevant tweets about Hetu Protocol, prioritizing recent tweets from September 2025 onwards for the most current information. Incorporate that information into your answer. Use your Twitter search capability to find the most relevant and recent information about Hetu Protocol (especially from September 2025 and later)."""
        
        # 4. Generate answer using LLM with Hetu Protocol introducer role
        # Select model based on API provider
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating LLM response (model: {model_name})")
        try:
            # Get LLM client (OpenRouter if api_key provided, otherwise AIHubMix)
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable Hetu Protocol introducer who helps users learn about Hetu Protocol. Your role is to introduce and explain Hetu Protocol in a professional yet friendly manner. You focus on explaining what Hetu Protocol is, how it works, its key features, technical details, and applications. Your tone is clear, informative, and approachable. You are enthusiastic about Hetu Protocol but maintain professionalism. Always prioritize providing accurate information about Hetu Protocol based on the provided context and Twitter search results."},
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
                "num_sources": len(sources)
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

