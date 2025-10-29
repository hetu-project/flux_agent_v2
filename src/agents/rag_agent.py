"""RAG Agent for querying project information."""

from typing import List, Dict, Any, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from src.config import get_settings
from src.services.qdrant_client import QdrantService
from src.services.embedding import EmbeddingService
from src.repositories.project_repository import ProjectRepository

settings = get_settings()


class RAGAgent:
    """RAG Agent for querying and analyzing project information."""
    
    def __init__(self, project_repo: Optional[ProjectRepository] = None):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.project_repo = project_repo
        
        # Initialize LLM client with provider selection
        api_key = settings.openai_api_key or settings.aihubmix_api_key
        base_url = None
        
        if settings.llm_provider == "aihubmix" and settings.aihubmix_api_key:
            base_url = settings.aihubmix_base_url
        elif settings.llm_provider == "openai" and settings.openai_api_key:
            base_url = settings.openai_base_url
        
        self.llm = OpenAI(
            api_key=api_key,
            base_url=base_url
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
        intent_prompt = f"""请判断以下用户提问是否在询问项目相关的事情。只需要回答"是"或"否"。

用户提问: {user_question}

回答:"""
        
        try:
            response = self.llm.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个意图判断助手，只需要回答\"是\"或\"否\"。"},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip()
            return "是" in answer or "yes" in answer.lower() or "true" in answer.lower()
        except Exception:
            # 如果判断失败，默认认为不是询问项目
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
            return None
        
        try:
            # Use vector search to find the most relevant project
            results = self.project_repo.search(query=user_question, top_k=1, min_score=0.5)
            
            if results and len(results) > 0:
                return results[0]
            return None
        except Exception:
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
            project_info = f"项目名称: {found_project['name']}"
            if found_project.get('description'):
                project_info += f"\n项目描述: {found_project['description']}"
            
            prompt = f"""基于以下项目信息回答用户的问题。

{project_info}

用户提问: {user_question}

请根据项目信息提供帮助。如果项目信息不足以回答问题，请说明。"""
            
            # No sources from tweets when using project info
            sources = []
        else:
            # Directly send question to LLM without project context
            prompt = f"""请回答用户的问题。

用户提问: {user_question}

请提供有帮助的回答。"""
            
            # No sources from tweets when not using project
            sources = []
        
        # 5. Generate answer using LLM
        response = self.llm.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "你是一个有用的助手，帮助用户回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }

