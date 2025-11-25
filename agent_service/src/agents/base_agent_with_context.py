"""Base agent class with context management support."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain.memory import ConversationBufferWindowMemory

from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.sql_models.agent import Agent
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.sql_models.conversation import Conversation

logger = get_logger(__name__)


class BaseAgentWithContext:
    """Base class for agents with context management support."""
    
    AGENT_NAME: str = ""  # Should be overridden by subclasses
    DEFAULT_MEMORY_WINDOW: int = 10  # Default memory window size
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize base agent with context support.
        
        Args:
            db_session: Optional async database session for context persistence
        """
        # Initialize LangChain Memory storage for different sessions
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}
        self._default_memory_window = self.DEFAULT_MEMORY_WINDOW
        
        # Database session for CRDT operations
        self.db_session = db_session
        self._agent_id: Optional[int] = None
    
    def _get_or_create_memory(self, session_id: Optional[str] = None) -> Optional[ConversationBufferWindowMemory]:
        """
        Get or create a memory instance for a specific session.
        If no session_id is provided, returns None (no memory storage).
        
        Args:
            session_id: Optional session identifier. If None, returns None
            
        Returns:
            ConversationBufferWindowMemory instance for the session, or None if no session_id
        """
        # If no session_id provided, don't use memory
        if not session_id:
            return None
        
        # Create memory if it doesn't exist for this session
        if session_id not in self._memories:
            self._memories[session_id] = ConversationBufferWindowMemory(
                k=self._default_memory_window,
                return_messages=True,
                memory_key="chat_history"
            )
            logger.debug(f"Created new memory for session: {session_id}")
        
        return self._memories[session_id]
    
    def _load_conversation_history_to_memory(
        self, 
        conversation_history: Optional[List[Dict[str, str]]],
        session_id: Optional[str] = None
    ):
        """
        Load conversation history into LangChain memory for a specific session.
        Only loads if session_id is provided.
        
        Args:
            conversation_history: List of messages with role and content
            session_id: Optional session identifier. If None, no memory is stored.
        """
        # If no session_id, don't store memory
        if not session_id:
            return
        
        if not conversation_history:
            return
        
        # Get memory for this session
        memory = self._get_or_create_memory(session_id)
        if not memory:
            return
        
        # Clear existing memory to avoid duplicates when loading from external history
        # This ensures we use the provided history as the source of truth
        memory.clear()
        
        # Load history into memory
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                memory.chat_memory.add_user_message(content)
            elif role == "assistant":
                memory.chat_memory.add_ai_message(content)
            elif role == "system":
                # System messages are handled separately in prompts
                pass
        
        logger.debug(f"Loaded {len(conversation_history)} messages into memory for session: {session_id}")
    
    def clear_session_memory(self, session_id: str):
        """
        Clear memory for a specific session.
        
        Args:
            session_id: Session identifier to clear
        """
        if session_id in self._memories:
            self._memories[session_id].clear()
            logger.debug(f"Cleared memory for session: {session_id}")
    
    def get_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self._memories)
    
    async def _get_agent_id(self) -> Optional[int]:
        """
        Get agent ID from database by name.
        
        Returns:
            Agent ID if found, None otherwise
        """
        if self._agent_id:
            return self._agent_id
        
        if not self.db_session:
            logger.warning("Database session not available, cannot get agent_id")
            return None
        
        if not self.AGENT_NAME:
            logger.warning("AGENT_NAME not set for agent")
            return None
        
        try:
            stmt = select(Agent).where(Agent.name == self.AGENT_NAME)
            result = await self.db_session.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if agent:
                self._agent_id = agent.id
                logger.info(f"Found agent_id={self._agent_id} for agent '{self.AGENT_NAME}'")
                return self._agent_id
            else:
                logger.warning(f"Agent '{self.AGENT_NAME}' not found in database")
                return None
        except Exception as e:
            logger.error(f"Error getting agent_id: {e}", exc_info=True)
            return None
    
    async def _get_or_create_conversation(
        self,
        session_id: Optional[str],
        user_id: Optional[int]
    ) -> Optional["Conversation"]:
        """
        Get or create conversation for the session.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID
            
        Returns:
            Conversation instance or None
        """
        if not self.db_session or not session_id:
            return None
        
        try:
            agent_id = await self._get_agent_id()
            if not agent_id:
                return None
            
            # Get or create conversation
            conv_repo = ConversationRepository(self.db_session)
            conversation = await conv_repo.get_by_session_id(
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id
            )
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error getting or creating conversation: {e}", exc_info=True)
            return None
    
    async def _load_conversation_from_db(
        self,
        session_id: Optional[str],
        user_id: Optional[int]
    ) -> Optional[List[Dict[str, str]]]:
        """
        Load conversation history from database.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID
            
        Returns:
            List of messages in format [{"role": "...", "content": "..."}] or None
        """
        conversation = await self._get_or_create_conversation(session_id, user_id)
        if not conversation:
            return None
        
        try:
            # Load messages
            msg_repo = MessageRepository(self.db_session)
            messages = await msg_repo.get_recent_messages(
                conversation_id=conversation.id,
                limit=self._default_memory_window
            )
            
            # Convert to format expected by LangChain
            history = []
            for msg in messages:
                history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            logger.info(f"Loaded {len(history)} messages from database for session: {session_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error loading conversation from database: {e}", exc_info=True)
            return None
    
    async def _save_message_to_db(
        self,
        session_id: Optional[str],
        user_id: Optional[int],
        role: str,
        content: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save message to database.
        
        Args:
            session_id: Session identifier
            user_id: Optional user ID
            role: Message role (user, assistant, system)
            content: Message content
            extra_metadata: Optional metadata
            
        Returns:
            True if saved successfully, False otherwise
        """
        conversation = await self._get_or_create_conversation(session_id, user_id)
        if not conversation:
            return False
        
        try:
            # Save message
            msg_repo = MessageRepository(self.db_session)
            await msg_repo.create(
                conversation_id=conversation.id,
                role=role,
                content=content,
                extra_metadata=extra_metadata
            )
            
            # Commit the transaction
            await self.db_session.commit()
            logger.debug(f"Saved {role} message to database for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving message to database: {e}", exc_info=True)
            await self.db_session.rollback()
            return False

