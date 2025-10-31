"""RAG Agent for querying project information."""

from typing import List, Dict, Any, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from src.config import get_settings
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.repositories.project_repository import ProjectRepository
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class RAGAgent:
    """RAG Agent for querying and analyzing project information."""
    
    def __init__(self, project_repo: Optional[ProjectRepository] = None):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.project_repo = project_repo
        
        # Initialize LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = settings.chat_model
        self.collection_name = "twitter_tweets"
    
    def _judge_intent(self, user_question: str) -> bool:
        """
        Judge if the user's question is about a project.
        
        Args:
            user_question: User's question
            
        Returns:
            True if the question is about a project, False otherwise
        """
        logger.debug(f"Judging intent for question: {user_question[:50]}...")
        intent_prompt = f"""Determine whether the following user question is about a project. Answer ONLY with YES or NO.

User question: {user_question}

Answer:"""
        
        try:
            response = self.llm.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are an intent classification assistant. Answer ONLY with YES or NO."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip()
            is_about_project = answer.strip().upper().startswith("YES")
            logger.debug(f"Intent judgment result: {is_about_project}")
            return is_about_project
        except Exception as e:
            logger.warning(f"Failed to judge intent: {e}, defaulting to False")
            return False
    
    def _find_project(self, user_question: str) -> Optional[Dict[str, Any]]:
        """
        Find the most relevant project from database based on user question.
        
        Args:
            user_question: User's question
            
        Returns:
            Project dict with name and description if found, None otherwise
        """
        if not self.project_repo:
            logger.debug("Project repository not available")
            return None
        
        try:
            logger.debug(f"Searching for project matching: {user_question[:50]}...")
            # Use vector search to find the most relevant project (projects are shared, no user_id filter)
            results = self.project_repo.search(query=user_question, top_k=1, min_score=0.5)
            
            if results and len(results) > 0:
                project = results[0]
                logger.info(f"Found project: {project.get('name')} (score: {project.get('score', 0):.3f})")
                return project
            logger.debug("No project found matching the query")
            return None
        except Exception as e:
            logger.error(f"Error finding project: {e}")
            return None
    
    async def query(
        self,
        user_question: str,
        project: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query the agent with a question.
        
        Logic:
        1. Judge if the question is about a project
        2. If yes, search for the project in database
        3. If project found, include project info in the prompt to LLM
        4. If project not found or not about project, directly send question to LLM
        
        Args:
            user_question: User's question
            project: Optional project filter (deprecated, will be auto-detected)
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Agent response with answer and sources
        """
        logger.info(f"Processing query: {user_question[:100]}...")
        # 1. Judge intent: is the question about a project?
        is_about_project = self._judge_intent(user_question)
        
        found_project = None
        if is_about_project and self.project_repo:
            # 2. Try to find the project in database
            found_project = self._find_project(user_question)
        
        # 3. If project is explicitly provided, use it (for backward compatibility)
        if project:
            try:
                found_project_obj = self.project_repo.get_by_name(project) if self.project_repo else None
                if found_project_obj:
                    found_project = {
                        "name": found_project_obj.name,
                        "description": found_project_obj.description
                    }
            except Exception:
                pass
        
        # 4. Build prompt based on whether project is found
        if found_project:
            # Include project information in the prompt
            project_info = f"Project name: {found_project['name']}"
            if found_project.get('description'):
                project_info += f"\nProject description: {found_project['description']}"
            
            prompt = f"""Answer the user's question based on the following project information.

{project_info}

User question: {user_question}

Provide a helpful answer based on the project info. If the information is insufficient, say so."""
            
            # No sources from tweets when using project info
            sources = []
        else:
            # Directly send question to LLM without project context
            prompt = f"""Please answer the user's question.

User question: {user_question}

Provide a helpful, concise answer."""
            
            # No sources from tweets when not using project
            sources = []
        
        # 5. Generate answer using LLM
        logger.debug(f"Generating LLM response (model: {self.chat_model})")
        try:
            response = self.llm.chat.completions.create(
                model=self.chat_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers user questions clearly and concisely."},
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

