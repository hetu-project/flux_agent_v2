"""Repository for Message CRUD operations with CRDT support."""

from typing import List, Optional, Dict, Any
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from src.sql_models.message import Message
from src.crdt.message_crdt import MessageCRDT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MessageRepository:
    """Repository for Message database operations with CRDT support."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async database session
        """
        self.session = session
    
    async def create(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sequence: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Create a new message.
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            sequence: Optional sequence number (auto-incremented if not provided)
            extra_metadata: Optional metadata dictionary
            
        Returns:
            Created Message instance
        """
        # Auto-increment sequence if not provided
        if sequence is None:
            sequence = await self.get_next_sequence(conversation_id)
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sequence=sequence,
            extra_metadata=extra_metadata
        )
        self.session.add(message)
        await self.session.flush()
        await self.session.refresh(message)
        logger.debug(f"Created message: id={message.id}, conversation_id={conversation_id}, sequence={sequence}")
        return message
    
    async def get_by_id(self, message_id: int) -> Optional[Message]:
        """
        Get message by ID.
        
        Args:
            message_id: Message ID
            
        Returns:
            Message instance or None
        """
        stmt = select(Message).where(Message.id == message_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_conversation(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """
        Get messages by conversation ID, ordered by sequence.
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of results
            offset: Offset for pagination
            
        Returns:
            List of Message instances
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.asc())
            .offset(offset)
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_next_sequence(self, conversation_id: int) -> int:
        """
        Get the next sequence number for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Next sequence number
        """
        stmt = (
            select(func.max(Message.sequence))
            .where(Message.conversation_id == conversation_id)
        )
        result = await self.session.execute(stmt)
        max_sequence = result.scalar() or 0
        return max_sequence + 1
    
    async def get_recent_messages(
        self,
        conversation_id: int,
        limit: int = 15
    ) -> List[Message]:
        """
        Get recent messages from a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Number of recent messages to retrieve
            
        Returns:
            List of Message instances (most recent first, then reversed for chronological order)
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages = list(result.scalars().all())
        # Reverse to get chronological order
        messages.reverse()
        return messages
    
    async def merge(self, message_data: Dict[str, Any]) -> Message:
        """
        Merge message data using CRDT operations.
        Creates new message if ID doesn't exist, otherwise updates.
        
        Args:
            message_data: Message data dictionary
            
        Returns:
            Merged Message instance
        """
        msg_id = message_data.get('id')
        
        if msg_id:
            # Try to get existing message
            existing = await self.get_by_id(msg_id)
            if existing:
                # Merge with existing
                existing_dict = MessageCRDT.to_dict(existing)
                merged_data = MessageCRDT.merge(existing_dict, message_data)
                
                # Update fields
                for key, value in merged_data.items():
                    if key != 'id' and hasattr(existing, key):
                        setattr(existing, key, value)
                
                await self.session.flush()
                await self.session.refresh(existing)
                logger.debug(f"Merged message: id={msg_id}")
                return existing
        
        # Create new message
        return await self.create(
            conversation_id=message_data['conversation_id'],
            role=message_data['role'],
            content=message_data['content'],
            sequence=message_data.get('sequence'),
            extra_metadata=message_data.get('extra_metadata')
        )
    
    async def batch_create(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Message]:
        """
        Create multiple messages in batch.
        
        Args:
            messages: List of message dictionaries with conversation_id, role, content
            
        Returns:
            List of created Message instances
        """
        created_messages = []
        
        for msg_data in messages:
            message = await self.create(
                conversation_id=msg_data['conversation_id'],
                role=msg_data['role'],
                content=msg_data['content'],
                sequence=msg_data.get('sequence'),
                extra_metadata=msg_data.get('extra_metadata')
            )
            created_messages.append(message)
        
        await self.session.flush()
        logger.info(f"Batch created {len(created_messages)} messages")
        return created_messages
    
    async def delete_by_conversation(self, conversation_id: int) -> int:
        """
        Delete all messages in a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Number of messages deleted
        """
        stmt = select(Message).where(Message.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        messages = result.scalars().all()
        
        count = 0
        for message in messages:
            await self.session.delete(message)
            count += 1
        
        await self.session.flush()
        logger.info(f"Deleted {count} messages for conversation: id={conversation_id}")
        return count

