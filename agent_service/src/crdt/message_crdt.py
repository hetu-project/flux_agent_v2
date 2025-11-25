"""CRDT operations for Message entities."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.crdt.crdt_operations import CRDTOperations
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MessageCRDT:
    """CRDT operations for Message synchronization."""
    
    @staticmethod
    def merge(message1: Dict[str, Any], message2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two message dictionaries using CRDT strategies.
        
        Args:
            message1: First message data
            message2: Second message data
            
        Returns:
            Merged message data
        """
        merged = {}
        
        # ID: Use the one that exists, prefer message1 if both exist
        merged['id'] = message1.get('id') or message2.get('id')
        
        # Foreign key: Use LWW
        merged['conversation_id'] = message2.get('conversation_id') or message1.get('conversation_id')
        
        # Role: LWW based on created_at
        ts1 = message1.get('created_at')
        ts2 = message2.get('created_at')
        merged['role'] = CRDTOperations.merge_strings_lww(
            message1.get('role'),
            message2.get('role'),
            ts1,
            ts2
        )
        
        # Content: LWW based on created_at
        merged['content'] = CRDTOperations.merge_strings_lww(
            message1.get('content'),
            message2.get('content'),
            ts1,
            ts2
        )
        
        # Sequence: Use maximum (for ordering)
        seq1 = message1.get('sequence', 0)
        seq2 = message2.get('sequence', 0)
        merged['sequence'] = max(seq1, seq2)
        
        # Metadata: Merge JSON
        merged['extra_metadata'] = CRDTOperations.merge_json_metadata(
            message1.get('extra_metadata'),
            message2.get('extra_metadata')
        )
        
        # Timestamp: Use latest
        merged['created_at'] = CRDTOperations.merge_timestamps(
            message1.get('created_at'),
            message2.get('created_at')
        )
        
        return merged
    
    @staticmethod
    def merge_list(messages1: List[Dict[str, Any]], 
                   messages2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge two lists of messages, preserving order and removing duplicates.
        
        Args:
            messages1: First list of messages
            messages2: Second list of messages
            
        Returns:
            Merged and sorted list of messages
        """
        # Create a dictionary keyed by (conversation_id, sequence) or id
        merged_dict = {}
        
        # Add messages from first list
        for msg in messages1:
            key = msg.get('id') or (msg.get('conversation_id'), msg.get('sequence', 0))
            if key not in merged_dict:
                merged_dict[key] = msg
            else:
                # Merge if duplicate
                merged_dict[key] = MessageCRDT.merge(merged_dict[key], msg)
        
        # Add messages from second list
        for msg in messages2:
            key = msg.get('id') or (msg.get('conversation_id'), msg.get('sequence', 0))
            if key not in merged_dict:
                merged_dict[key] = msg
            else:
                # Merge if duplicate
                merged_dict[key] = MessageCRDT.merge(merged_dict[key], msg)
        
        # Convert back to list and sort by sequence
        result = list(merged_dict.values())
        result.sort(key=lambda x: (x.get('conversation_id', 0), x.get('sequence', 0)))
        
        return result
    
    @staticmethod
    def to_dict(message) -> Dict[str, Any]:
        """
        Convert a Message SQLAlchemy model to dictionary.
        
        Args:
            message: Message model instance
            
        Returns:
            Dictionary representation
        """
        return {
            'id': message.id,
            'conversation_id': message.conversation_id,
            'role': message.role,
            'content': message.content,
            'sequence': message.sequence,
            'extra_metadata': message.extra_metadata,
            'created_at': message.created_at,
        }

