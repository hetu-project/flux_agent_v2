"""Company schemas for API requests and responses."""

# Reuse chat schemas for consistency
from .chat_schema import ChatRequest, ChatResponse

# Export for convenience
CompanyRequest = ChatRequest
CompanyResponse = ChatResponse

