"""Database service for PostgreSQL connection management."""

from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass

# Global engine instances
_sync_engine: Engine | None = None
_async_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_sync_engine() -> Engine:
    """
    Get or create synchronous SQLAlchemy engine.
    
    Returns:
        Synchronous SQLAlchemy engine instance
    """
    global _sync_engine
    if _sync_engine is None:
        logger.info(f"Creating PostgreSQL sync engine: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
        _sync_engine = create_engine(
            settings.postgres_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,  # Connection pool size
            max_overflow=20,  # Max overflow connections
            echo=False,  # Set to True for SQL query logging
        )
        logger.info("PostgreSQL sync engine created successfully")
    return _sync_engine


def get_async_engine() -> AsyncEngine:
    """
    Get or create asynchronous SQLAlchemy engine.
    
    Returns:
        Asynchronous SQLAlchemy engine instance
    """
    global _async_engine
    if _async_engine is None:
        logger.info(f"Creating PostgreSQL async engine: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
        _async_engine = create_async_engine(
            settings.postgres_url_async,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,  # Connection pool size
            max_overflow=20,  # Max overflow connections
            echo=False,  # Set to True for SQL query logging
        )
        logger.info("PostgreSQL async engine created successfully")
    return _async_engine


def get_async_session_maker() -> async_sessionmaker[AsyncSession]:
    """
    Get or create async session maker.
    
    Returns:
        Async session maker instance
    """
    global _async_session_maker
    if _async_session_maker is None:
        engine = get_async_engine()
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_maker


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """
    Get a synchronous database session (context manager).
    
    Usage:
        with get_sync_session() as session:
            # Use session
            pass
    
    Yields:
        Synchronous database session
    """
    engine = get_sync_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an asynchronous database session (context manager).
    
    Usage:
        async with get_async_session() as session:
            # Use session
            pass
    
    Yields:
        Asynchronous database session
    """
    session_maker = get_async_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_database_connection() -> bool:
    """
    Check if database connection is available.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        from sqlalchemy import text
        async with get_async_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
        logger.info("Database connection check successful")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def close_database_connections():
    """Close all database connections."""
    global _sync_engine, _async_engine, _async_session_maker
    
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        logger.info("Async database engine closed")
    
    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
        logger.info("Sync database engine closed")
    
    _async_session_maker = None

