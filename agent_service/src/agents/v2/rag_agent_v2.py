"""RAG Agent V2 for querying project information with cheaper models and RAG-based tweet search."""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
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


class RAGAgentV2:
    """RAG Agent V2 for querying and analyzing project information with cheaper models and RAG-based tweet search."""
    
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
        # Use cheaper model: gpt-3.5-turbo or gpt-4o-mini instead of grok
        self.chat_model = "gpt-4o-mini"  # Cheaper alternative to Grok
    
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
    
    def _find_project(self, user_question: str, min_score: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        Find the most relevant project from database based on user question.
        
        Strategy:
        1. First, try to find project by name using vector similarity search on project names
        2. If not found, fall back to vector similarity search on project descriptions
        
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
            
            # Step 1: Try to find project by name using vector similarity
            # Get all projects and search by name similarity
            all_projects = self.project_repo.get_all()
            question_lower = user_question.lower()
            
            # Generate embedding for the question
            question_embedding = self.embedding.embed_text(user_question)
            
            # Try to match project names using vector similarity
            best_name_match = None
            best_name_score = 0.0
            
            for project in all_projects:
                project_name = project.name
                if not project_name:
                    continue
                
                # Generate embedding for project name
                name_embedding = self.embedding.embed_text(project_name)
                
                # Convert to numpy arrays for similarity calculation
                question_vec = np.array(question_embedding)
                name_vec = np.array(name_embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(question_vec, name_vec) / (
                    np.linalg.norm(question_vec) * np.linalg.norm(name_vec)
                )
                
                # Also check if project name appears in question (exact or partial match)
                project_name_lower = project_name.lower()
                name_in_question = project_name_lower in question_lower
                
                # Check for partial match: split project name into words
                project_words = [w for w in project_name_lower.split() if len(w) > 2]
                partial_match = any(word in question_lower for word in project_words)
                
                # If name appears in question or similarity is high, consider it a match
                if name_in_question or partial_match or similarity >= min_score:
                    if similarity > best_name_score:
                        best_name_score = similarity
                        best_name_match = {
                            "name": project_name,
                            "description": project.description,
                            "score": float(similarity)
                        }
            
            # If we found a good name match, return it
            if best_name_match and (best_name_score >= min_score or 
                                    best_name_match["name"].lower() in question_lower):
                logger.info(f"Found project '{best_name_match['name']}' by name matching (score: {best_name_score:.3f})")
                return best_name_match
            
            # Step 2: Fall back to description-based vector search
            logger.debug("No project found by name matching, trying description-based search...")
            results = self.project_repo.search(query=user_question, top_k=5, min_score=min_score)
            
            if results and len(results) > 0:
                # Check all results to see if any project name appears in the question
                for project in results:
                    project_name = project.get('name')
                    if not project_name:
                        continue
                    
                    project_name_lower = project_name.lower()
                    score = project.get('score', 0)
                    
                    # If project name appears in question, prioritize it
                    if project_name_lower in question_lower:
                        logger.info(f"Found project '{project_name}' by description search with name match (score: {score:.3f})")
                        return project
                    
                    # Check for partial match
                    project_words = [w for w in project_name_lower.split() if len(w) > 2]
                    if any(word in question_lower for word in project_words):
                        logger.info(f"Found project '{project_name}' by description search with partial name match (score: {score:.3f})")
                        return project
                
                # If no name match found, return the top result if similarity is high enough
                top_project = results[0]
                top_score = top_project.get('score', 0)
                if top_score >= 0.75:
                    logger.info(f"Found project '{top_project.get('name')}' by description search (high similarity: {top_score:.3f})")
                    return top_project
                else:
                    logger.debug(f"Top project '{top_project.get('name')}' found but similarity too low ({top_score:.3f} < 0.75)")
            
            logger.debug(f"No project found matching the query (similarity threshold: {min_score})")
            return None
        except Exception as e:
            logger.error(f"Error finding project: {e}", exc_info=True)
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
    
    def _check_parallel_universe_count_intent(self, user_question: str) -> bool:
        """
        Check if user's question is asking about the number/count of parallel universe projects.
        
        Uses keyword matching to detect questions about parallel universe count, project count, etc.
        
        Args:
            user_question: User's question
            
        Returns:
            True if user is asking about project count, False otherwise
        """
        question_lower = user_question.lower()
        
        # Keywords related to counting projects
        count_keywords = [
            "数量", "多少", "几个", "count", "number", "how many",
            "平行网", "parallel", "projects", "项目数量",
            "有多少", "total", "共", "总共有"
        ]
        
        # Check if question contains count-related keywords
        has_count_keyword = any(keyword in question_lower for keyword in count_keywords)
        
        if not has_count_keyword:
            return False
        
        # Additional check: question should be related to projects/parallel universe
        project_related_keywords = [
            "平行网", "parallel", "project", "项目", "universe",
            "network", "生态系统", "ecosystem"
        ]
        
        has_project_keyword = any(keyword in question_lower for keyword in project_related_keywords)
        
        return has_project_keyword
    
    def _get_project_count(self) -> int:
        """
        Get the total number of projects in the database.
        
        Returns:
            Total number of projects
        """
        try:
            if not self.project_repo:
                logger.debug("Project repository not available")
                return 0
            
            # Use project repository to get all projects and count
            all_projects = self.project_repo.get_all()
            count = len(all_projects)
            logger.info(f"Found {count} projects in database")
            return count
        except Exception as e:
            logger.error(f"Error getting project count: {e}")
            return 0
    
    async def query(
        self,
        user_question: str,
        project: Optional[str] = None,
        top_k: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the agent with a question.
        
        Logic:
        1. Check if user is asking about parallel universe project count
        2. Search for matching projects in database using vector similarity
        3. Search for relevant tweets using RAG
        4. Search for relevant content from project_content collection
        5. Generate answer using cheaper LLM model
        
        Args:
            user_question: User's question
            project: Optional project filter (deprecated, will be auto-detected)
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Agent response with answer and sources
        """
        logger.info(f"Processing query: {user_question[:100]}...")
        
        # 1. Check if user is asking about parallel universe project count
        if self._check_parallel_universe_count_intent(user_question):
            logger.info("Detected parallel universe count intent")
            project_count = self._get_project_count()
            
            if project_count > 0:
                answer = f"Currently, there are {project_count} projects in the Hetu Parallel Universe ecosystem. The Hetu Parallel Universe is an ecosystem built by HETU using the FLUX points system, consisting of multiple parallel universe projects that work together to create a decentralized intelligence economy."
                
                return {
                    "answer": answer,
                    "sources": [],
                    "num_sources": 0,
                    "project_count": project_count
                }
            else:
                return {
                    "answer": "I'm unable to retrieve the current number of projects in the Hetu Parallel Universe. The database connection may be unavailable. Please try again later.",
                    "sources": [],
                    "num_sources": 0
                }
        
        # 2. Try to find a matching project in database
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
        
        # 4. Search for relevant tweets using RAG
        relevant_tweets = []
        if found_project:
            # Search tweets for the specific project
            relevant_tweets = self._search_tweets(
                query=user_question,
                project=found_project['name'],
                top_k=top_k
            )
        else:
            # Search tweets without project filter
            relevant_tweets = self._search_tweets(
                query=user_question,
                top_k=top_k
            )
        
        # 5. Search for relevant content from project_content collection
        sources = []
        relevant_content = []
        
        if found_project and self.project_content_repo:
            # Search for relevant content (papers, etc.) related to the project
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
        
        logger.info(f"Found {len(sources)} relevant sources (content: {len(relevant_content)}, tweets: {len(relevant_tweets)})")
        
        # 6. Build prompt based on whether project is found
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
            
            prompt = f"""Answer the user's question based on the following project information, relevant content, and tweets.

{project_info}{content_context}{tweets_context}

User question: {user_question}

Please provide a helpful answer based on the project info, relevant content, and tweets above. If the information is insufficient, please say so."""
        else:
            # Add relevant tweets context even if no project found
            tweets_context = ""
            if relevant_tweets:
                tweets_context = "\n\nRelevant tweets from Twitter:\n"
                for i, tweet in enumerate(relevant_tweets[:top_k], 1):
                    tweet_text = tweet.get("text", "")[:300]
                    author = tweet.get("author", "Unknown")
                    created_at = tweet.get("created_at", "")
                    tweets_context += f"\n[{i}] @{author} ({created_at}): {tweet_text}\n"
            
            prompt = f"""Please answer the user's question based on the following relevant tweets.{tweets_context}

User question: {user_question}

Please provide a helpful, concise answer based on the tweets above."""
        
        # 7. Generate answer using cheaper LLM model
        # Select model based on API provider (same logic as v1)
        model_name = settings.openrouter_model if api_key else self.chat_model
        logger.debug(f"Generating LLM response (model: {model_name})")
        try:
            # Get LLM client (OpenRouter if api_key provided, otherwise AIHubMix)
            llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable Hetu Parallel Universe introducer who helps users learn about projects in the parallel universe ecosystem. Hetu Parallel Universe is an ecosystem built by HETU using the FLUX points system, consisting of multiple parallel universe projects. Your role is to introduce and explain various projects within the Hetu Parallel Universe ecosystem. You can answer questions about any project in the parallel universe. Your tone is professional yet friendly, clear and informative. You focus on explaining what each project is, how it works, and how it relates to the Hetu Parallel Universe ecosystem. Always prioritize providing accurate information about projects in the parallel universe based on the provided context."},
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

