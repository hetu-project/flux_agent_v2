"""Repository for User CRUD operations."""

from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.sql_models.user import User
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserRepository:
    """Repository for User database operations."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async database session
        """
        self.session = session
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User instance or None
        """
        stmt = select(User).where(User.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_client_id(self, client_id: str) -> Optional[User]:
        """
        Get user by client_id.
        
        Args:
            client_id: Client identifier
            
        Returns:
            User instance or None
        """
        stmt = select(User).where(User.client_id == client_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_or_create_by_client_id(self, client_id: str) -> User:
        """
        Get user by client_id, or create if not exists.
        
        Args:
            client_id: Client identifier
            
        Returns:
            User instance
        """
        user = await self.get_by_client_id(client_id)
        if user:
            return user
        
        # Create new user
        user = User(client_id=client_id)
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        logger.info(f"Created new user: id={user.id}, client_id={client_id}")
        return user

