"""Service layer for chat history business logic."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from src.repositories.user_repository import UserRepository
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.sql_models.agent import Agent
from src.sql_models.conversation import Conversation
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatHistoryService:
    """Service for chat history business operations."""
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize chat history service.
        
        Args:
            db_session: Async database session
        """
        self.db_session = db_session
        self.user_repo = UserRepository(db_session)
        self.conv_repo = ConversationRepository(db_session)
        self.msg_repo = MessageRepository(db_session)
    
    async def get_chat_history(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get chat history for a user.
        
        Args:
            user_id: User client_id
            limit: Maximum number of conversations to return
            offset: Offset for pagination
            agent_name: Optional agent name to filter by
            
        Returns:
            Dictionary containing:
                - conversations: List of conversations with messages
                - total: Total number of conversations
                - limit: Limit used
                - offset: Offset used
        """
        # Get or create user
        user = await self.user_repo.get_or_create_by_client_id(user_id)
        user_id_int = user.id
        
        # Get agent_id if agent_name is provided
        agent_id = None
        agent_name_map = {}
        if agent_name:
            stmt = select(Agent).where(Agent.name == agent_name)
            result = await self.db_session.execute(stmt)
            agent = result.scalar_one_or_none()
            if agent:
                agent_id = agent.id
                agent_name_map[agent.id] = agent.name
            else:
                logger.warning(f"Agent '{agent_name}' not found")
        
        # Get conversations for user
        conversations = await self._get_conversations(
            user_id_int=user_id_int,
            agent_id=agent_id
        )
        
        # Load agent names for all conversations
        agent_ids = set(conv.agent_id for conv in conversations)
        if agent_ids:
            stmt = select(Agent).where(Agent.id.in_(agent_ids))
            result = await self.db_session.execute(stmt)
            agents = result.scalars().all()
            for agent in agents:
                agent_name_map[agent.id] = agent.name
        
        # Get messages for each conversation
        conversation_list = []
        for conv in conversations:
            messages = await self.msg_repo.get_by_conversation(
                conversation_id=conv.id,
                limit=None,  # Get all messages
                offset=0
            )
            
            # Convert to dict format
            message_list = []
            for msg in messages:
                message_list.append({
                    "id": msg.id,
                    "conversation_id": msg.conversation_id,
                    "role": msg.role,
                    "content": msg.content,
                    "sequence": msg.sequence,
                    "created_at": msg.created_at.isoformat() if msg.created_at else "",
                    "extra_metadata": msg.extra_metadata
                })
            
            conversation_list.append({
                "id": conv.id,
                "agent_id": conv.agent_id,
                "agent_name": agent_name_map.get(conv.agent_id),
                "title": conv.title,
                "status": conv.status,
                "created_at": conv.created_at.isoformat() if conv.created_at else "",
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
                "messages": message_list
            })
        
        # Apply pagination to conversations (after loading all messages)
        total = len(conversation_list)
        paginated_conversations = conversation_list[offset:offset + limit]
        
        logger.info(
            f"Retrieved chat history for user_id={user_id}: "
            f"{len(paginated_conversations)} conversations (total: {total})"
        )
        
        return {
            "conversations": paginated_conversations,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    
    async def _get_conversations(
        self,
        user_id_int: int,
        agent_id: Optional[int] = None
    ) -> List[Conversation]:
        """
        Get conversations for a user, optionally filtered by agent.
        
        Args:
            user_id_int: User ID (integer)
            agent_id: Optional agent ID to filter by
            
        Returns:
            List of Conversation instances
        """
        if agent_id:
            conversations = await self.conv_repo.get_by_user_and_agent(
                user_id=user_id_int,
                agent_id=agent_id,
                status="active",
                limit=1000  # Get all conversations, we'll paginate later
            )
        else:
            # Get all conversations for user across all agents
            stmt = select(Conversation).where(
                and_(
                    Conversation.user_id == user_id_int,
                    Conversation.status == "active"
                )
            ).order_by(Conversation.updated_at.desc())
            result = await self.db_session.execute(stmt)
            conversations = list(result.scalars().all())
        
        return conversations

