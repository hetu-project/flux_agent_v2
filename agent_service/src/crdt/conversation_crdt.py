"""CRDT operations for Conversation entities."""

from typing import Optional, Dict, Any
from datetime import datetime
from src.crdt.crdt_operations import CRDTOperations
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationCRDT:
    """CRDT operations for Conversation synchronization."""
    
    @staticmethod
    def merge(conv1: Dict[str, Any], conv2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two conversation dictionaries using CRDT strategies.
        
        Args:
            conv1: First conversation data
            conv2: Second conversation data
            
        Returns:
            Merged conversation data
        """
        merged = {}
        
        # ID: Use the one that exists, prefer conv1 if both exist
        merged['id'] = conv1.get('id') or conv2.get('id')
        
        # Foreign keys: Use LWW (prefer conv2 if both exist)
        merged['agent_id'] = conv2.get('agent_id') or conv1.get('agent_id')
        merged['user_id'] = conv2.get('user_id') or conv1.get('user_id')
        
        # Title: LWW based on updated_at
        ts1 = conv1.get('updated_at')
        ts2 = conv2.get('updated_at')
        merged['title'] = CRDTOperations.merge_strings_lww(
            conv1.get('title'),
            conv2.get('title'),
            ts1,
            ts2
        )
        
        # Status: LWW based on updated_at
        merged['status'] = CRDTOperations.merge_strings_lww(
            conv1.get('status', 'active'),
            conv2.get('status', 'active'),
            ts1,
            ts2
        )
        
        # Timestamps: Use latest
        merged['created_at'] = CRDTOperations.merge_timestamps(
            conv1.get('created_at'),
            conv2.get('created_at')
        )
        merged['updated_at'] = CRDTOperations.merge_timestamps(
            conv1.get('updated_at'),
            conv2.get('updated_at')
        )
        merged['expired_at'] = CRDTOperations.merge_timestamps(
            conv1.get('expired_at'),
            conv2.get('expired_at')
        )
        
        return merged
    
    @staticmethod
    def to_dict(conversation) -> Dict[str, Any]:
        """
        Convert a Conversation SQLAlchemy model to dictionary.
        
        Args:
            conversation: Conversation model instance
            
        Returns:
            Dictionary representation
        """
        return {
            'id': conversation.id,
            'agent_id': conversation.agent_id,
            'user_id': conversation.user_id,
            'title': conversation.title,
            'status': conversation.status,
            'created_at': conversation.created_at,
            'updated_at': conversation.updated_at,
            'expired_at': conversation.expired_at,
        }

