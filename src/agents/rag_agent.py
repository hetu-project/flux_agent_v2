"""RAG Agent for querying project information."""

from typing import List, Dict, Any, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from src.config import get_settings
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.repositories.project_repository import ProjectRepository
from src.repositories.project_content_repository import ProjectContentRepository
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class RAGAgent:
    """RAG Agent for querying and analyzing project information."""
    
    def __init__(
        self,
        project_repo: Optional[ProjectRepository] = None,
        project_content_repo: Optional[ProjectContentRepository] = None
    ):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.project_repo = project_repo
        self.project_content_repo = project_content_repo
        
        # Initialize LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = settings.chat_model
    
    def _find_project(self, user_question: str, min_score: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Find the most relevant project from database based on user question.
        
        This method uses vector similarity search to find projects. It only returns a project
        if the similarity score is high enough (>= min_score), ensuring the user's question
        is actually related to a specific project in the database.
        
        Args:
            user_question: User's question
            min_score: Minimum similarity score threshold (default: 0.6)
                       Higher threshold = more strict matching
            
        Returns:
            Project dict with name and description if found with sufficient similarity, None otherwise
        """
        if not self.project_repo:
            logger.debug("Project repository not available")
            return None
        
        try:
            logger.debug(f"Searching for project matching: {user_question[:50]}... (min_score={min_score})")
            # Use vector search to find the most relevant project
            # Higher min_score ensures we only match when user explicitly mentions/asks about a project
            results = self.project_repo.search(query=user_question, top_k=1, min_score=min_score)
            
            if results and len(results) > 0:
                project = results[0]
                score = project.get('score', 0)
                project_name = project.get('name')
                
                if not project_name:
                    logger.debug("Found project but no name, skipping")
                    return None
                
                # Check if project name (or significant words from it) appears in question
                question_lower = user_question.lower()
                project_name_lower = project_name.lower()
                
                # Check for exact project name match
                if project_name_lower in question_lower:
                    logger.info(f"Found project '{project_name}' (score: {score:.3f}) - exact name match")
                    return project
                
                # Check for partial match: split project name into words and check if any significant word appears
                # This handles cases like "Akasha" matching "Akasha Dao"
                project_words = [w for w in project_name_lower.split() if len(w) > 3]  # Only check words longer than 3 chars
                for word in project_words:
                    if word in question_lower:
                        logger.info(f"Found project '{project_name}' (score: {score:.3f}) - partial name match (word: {word})")
                        return project
                
                # If similarity is very high (>= 0.75), accept even if name not explicitly mentioned
                # This handles cases like "tell me about the fancy project" matching based on description
                if score >= 0.75:
                    logger.info(f"Found project '{project_name}' (score: {score:.3f}) - high similarity match")
                    return project
                else:
                    logger.debug(f"Project '{project_name}' found but similarity too low ({score:.3f} < 0.75) and name not mentioned")
                    return None
            
            logger.debug(f"No project found matching the query (similarity threshold: {min_score})")
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
        1. Search for matching projects in database using vector similarity
           - Only matches if user question mentions project name or has high similarity (>= 0.75)
           - Uses stricter similarity threshold (0.6) to ensure relevance
        2. If project found with sufficient similarity, include project info in the prompt
        3. If project not found or similarity too low, answer question directly without project context
        
        Args:
            user_question: User's question
            project: Optional project filter (deprecated, will be auto-detected)
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Agent response with answer and sources
        """
        logger.info(f"Processing query: {user_question[:100]}...")
        # 1. Try to find a matching project in database
        # We use vector search to find projects - if a project is found with sufficient similarity,
        # it means the user's question is about a specific project
        found_project = None
        if self.project_repo:
            found_project = self._find_project(user_question, min_score=0.6)
        
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
        
        # 4. Search for relevant content from project_content collection
        sources = []
        relevant_content = []
        
        if found_project and self.project_content_repo:
            # Search for relevant content (tweets, papers, etc.) related to the project
            logger.debug(f"Searching for relevant content for project '{found_project['name']}'")
            relevant_content = self.project_content_repo.search(
                query=user_question,
                project_name=found_project['name'],
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
            
            logger.info(f"Found {len(sources)} relevant content items for project '{found_project['name']}'")
        
        # 5. Build prompt based on whether project is found
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
            
            prompt = f"""Answer the user's question based on the following project information and relevant content.

{project_info}{content_context}

User question: {user_question}

Please provide a helpful answer based on the project info and relevant content above. Additionally, please search Twitter for relevant tweets, discussions, or updates about this project. Prioritize finding recent tweets from 2025 onwards, as these will contain the most up-to-date information. Use your Twitter search capability to find the most relevant and recent tweets (especially from 2025) that can help answer the user's question. If the information from the database and Twitter is insufficient, please say so."""
        else:
            # Directly send question to LLM without project context
            prompt = f"""Please answer the user's question.

User question: {user_question}

Please provide a helpful, concise answer. Please search Twitter for relevant tweets, prioritizing recent tweets from 2025 onwards for the most current information. Incorporate that information into your answer. Use your Twitter search capability to find the most relevant and recent information (especially from 2025)."""
        
        # 6. Generate answer using LLM
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

