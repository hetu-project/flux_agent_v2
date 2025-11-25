"""Repository for Conversation CRUD operations with CRDT support."""

from typing import List, Optional, Dict, Any
from sqlalchemy import select, and_, or_, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone

from src.sql_models.conversation import Conversation
from src.sql_models.agent import Agent
from src.sql_models.user import User
from src.crdt.conversation_crdt import ConversationCRDT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationRepository:
    """Repository for Conversation database operations with CRDT support."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async database session
        """
        self.session = session
    
    async def create(
        self,
        agent_id: int,
        user_id: Optional[int] = None,
        title: Optional[str] = None,
        status: str = "active"
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            agent_id: Agent ID
            user_id: Optional user ID
            title: Optional conversation title
            status: Conversation status (default: "active")
            
        Returns:
            Created Conversation instance
        """
        conversation = Conversation(
            agent_id=agent_id,
            user_id=user_id,
            title=title,
            status=status
        )
        self.session.add(conversation)
        await self.session.flush()
        await self.session.refresh(conversation)
        logger.info(f"Created conversation: id={conversation.id}, agent_id={agent_id}, user_id={user_id}")
        return conversation
    
    async def get_by_id(self, conversation_id: int) -> Optional[Conversation]:
        """
        Get conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation instance or None
        """
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_session_id(
        self,
        session_id: str,
        agent_id: int,
        user_id: Optional[int] = None
    ) -> Optional[Conversation]:
        """
        Get or create conversation by session_id.
        Uses session_id stored in title field as a lookup key.
        
        Args:
            session_id: Session identifier
            agent_id: Agent ID
            user_id: Optional user ID
            
        Returns:
            Conversation instance
        """
        # Store session_id in title field with a prefix for identification
        session_title = f"Session: {session_id[:200]}" if session_id else None
        
        # Try to find existing conversation by session_id (stored in title)
        stmt = select(Conversation).where(
            and_(
                Conversation.agent_id == agent_id,
                Conversation.status == "active",
                Conversation.title == session_title
            )
        )
        
        if user_id:
            stmt = stmt.where(Conversation.user_id == user_id)
        
        result = await self.session.execute(stmt.order_by(Conversation.updated_at.desc()))
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            # Create new conversation with session_id in title
            conversation = await self.create(
                agent_id=agent_id,
                user_id=user_id,
                title=session_title,
                status="active"
            )
        
        return conversation
    
    async def get_by_user_and_agent(
        self,
        user_id: Optional[int],
        agent_id: int,
        status: str = "active",
        limit: int = 10
    ) -> List[Conversation]:
        """
        Get conversations by user and agent.
        
        Args:
            user_id: Optional user ID
            agent_id: Agent ID
            status: Conversation status filter
            limit: Maximum number of results
            
        Returns:
            List of Conversation instances
        """
        stmt = select(Conversation).where(
            and_(
                Conversation.agent_id == agent_id,
                Conversation.status == status
            )
        )
        
        if user_id:
            stmt = stmt.where(Conversation.user_id == user_id)
        
        stmt = stmt.order_by(Conversation.updated_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def update(
        self,
        conversation_id: int,
        title: Optional[str] = None,
        status: Optional[str] = None,
        expired_at: Optional[datetime] = None
    ) -> Optional[Conversation]:
        """
        Update conversation.
        
        Args:
            conversation_id: Conversation ID
            title: Optional new title
            status: Optional new status
            expired_at: Optional expiration timestamp
            
        Returns:
            Updated Conversation instance or None
        """
        conversation = await self.get_by_id(conversation_id)
        if not conversation:
            return None
        
        if title is not None:
            conversation.title = title
        if status is not None:
            conversation.status = status
        if expired_at is not None:
            conversation.expired_at = expired_at
        
        conversation.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        await self.session.refresh(conversation)
        logger.info(f"Updated conversation: id={conversation_id}")
        return conversation
    
    async def merge(self, conversation_data: Dict[str, Any]) -> Conversation:
        """
        Merge conversation data using CRDT operations.
        Creates new conversation if ID doesn't exist, otherwise updates.
        
        Args:
            conversation_data: Conversation data dictionary
            
        Returns:
            Merged Conversation instance
        """
        conv_id = conversation_data.get('id')
        
        if conv_id:
            # Try to get existing conversation
            existing = await self.get_by_id(conv_id)
            if existing:
                # Merge with existing
                existing_dict = ConversationCRDT.to_dict(existing)
                merged_data = ConversationCRDT.merge(existing_dict, conversation_data)
                
                # Update fields
                for key, value in merged_data.items():
                    if key != 'id' and hasattr(existing, key):
                        setattr(existing, key, value)
                
                existing.updated_at = datetime.utcnow()
                await self.session.flush()
                await self.session.refresh(existing)
                logger.info(f"Merged conversation: id={conv_id}")
                return existing
        
        # Create new conversation
        return await self.create(
            agent_id=conversation_data['agent_id'],
            user_id=conversation_data.get('user_id'),
            title=conversation_data.get('title'),
            status=conversation_data.get('status', 'active')
        )
    
    async def delete(self, conversation_id: int) -> bool:
        """
        Soft delete conversation by setting status to 'deleted'.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if deleted, False if not found
        """
        conversation = await self.get_by_id(conversation_id)
        if not conversation:
            return False
        
        conversation.status = "deleted"
        conversation.updated_at = datetime.utcnow()
        await self.session.flush()
        logger.info(f"Deleted conversation: id={conversation_id}")
        return True

