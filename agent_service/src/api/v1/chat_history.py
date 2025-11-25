"""Chat history API routes."""

from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.chat_schema import GetChatHistoryRequest, GetChatHistoryResponse, ChatHistoryConversation, ChatHistoryMessage
from src.services.database import get_async_session
from src.services.chat_history_service import ChatHistoryService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat-history"])


@router.post("/history", response_model=GetChatHistoryResponse)
async def get_chat_history(
    request: GetChatHistoryRequest,
):
    """
    Get chat history for a user.
    
    Retrieves all conversations and messages for a specific user, with pagination support.
    Optionally filters by agent name.
    
    Request format:
    {
        "user_id": "user-client-id",
        "limit": 20,
        "offset": 0,
        "agent_name": "company"  // Optional
    }
    
    Returns:
    {
        "conversations": [
            {
                "id": 1,
                "agent_id": 1,
                "agent_name": "company",
                "title": "Session: abc123",
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "messages": [
                    {
                        "id": 1,
                        "conversation_id": 1,
                        "role": "user",
                        "content": "你好",
                        "sequence": 0,
                        "created_at": "2024-01-01T00:00:00Z",
                        "extra_metadata": null
                    }
                ]
            }
        ],
        "total": 1,
        "limit": 20,
        "offset": 0
    }
    """
    async with get_async_session() as db_session:
        try:
            # Initialize service
            service = ChatHistoryService(db_session)
            
            # Get chat history
            result = await service.get_chat_history(
                user_id=request.user_id,
                limit=request.limit,
                offset=request.offset,
                agent_name=request.agent_name
            )
            
            # Convert to response schema
            conversations = []
            for conv_data in result["conversations"]:
                messages = [
                    ChatHistoryMessage(**msg_data)
                    for msg_data in conv_data["messages"]
                ]
                conversations.append(ChatHistoryConversation(
                    id=conv_data["id"],
                    agent_id=conv_data["agent_id"],
                    agent_name=conv_data["agent_name"],
                    title=conv_data["title"],
                    status=conv_data["status"],
                    created_at=conv_data["created_at"],
                    updated_at=conv_data["updated_at"],
                    messages=messages
                ))
            
            return GetChatHistoryResponse(
                conversations=conversations,
                total=result["total"],
                limit=result["limit"],
                offset=result["offset"]
            )
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

