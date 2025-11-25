"""CRDT operations for conflict-free data synchronization."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CRDTOperations:
    """Base CRDT operations for conflict-free data merging."""
    
    @staticmethod
    def merge_timestamps(ts1: Optional[datetime], ts2: Optional[datetime]) -> Optional[datetime]:
        """
        Merge two timestamps using Last-Write-Wins (LWW) strategy.
        
        Args:
            ts1: First timestamp
            ts2: Second timestamp
            
        Returns:
            The later timestamp
        """
        if ts1 is None:
            return ts2
        if ts2 is None:
            return ts1
        return max(ts1, ts2)
    
    @staticmethod
    def merge_strings_lww(str1: Optional[str], str2: Optional[str], 
                          ts1: Optional[datetime], ts2: Optional[datetime]) -> Optional[str]:
        """
        Merge two strings using Last-Write-Wins (LWW) strategy.
        
        Args:
            str1: First string value
            str2: Second string value
            ts1: Timestamp for first value
            ts2: Timestamp for second value
            
        Returns:
            The string with the later timestamp
        """
        if str1 is None:
            return str2
        if str2 is None:
            return str1
        
        # Use timestamps to determine winner
        if ts1 and ts2:
            return str2 if ts2 > ts1 else str1
        # If no timestamps, prefer non-empty value
        return str2 if str2 else str1
    
    @staticmethod
    def merge_integers_lww(int1: Optional[int], int2: Optional[int],
                           ts1: Optional[datetime], ts2: Optional[datetime]) -> Optional[int]:
        """
        Merge two integers using Last-Write-Wins (LWW) strategy.
        
        Args:
            int1: First integer value
            int2: Second integer value
            ts1: Timestamp for first value
            ts2: Timestamp for second value
            
        Returns:
            The integer with the later timestamp
        """
        if int1 is None:
            return int2
        if int2 is None:
            return int1
        
        if ts1 and ts2:
            return int2 if ts2 > ts1 else int1
        return int2
    
    @staticmethod
    def merge_sequences_append(seq1: List[Any], seq2: List[Any]) -> List[Any]:
        """
        Merge two sequences by appending and deduplicating.
        For messages, we maintain order and remove duplicates based on ID.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Merged sequence with duplicates removed
        """
        # For messages, we want to preserve order and remove duplicates
        # Use a dict to track seen items by their unique identifier
        seen = {}
        result = []
        
        # Process first sequence
        for item in seq1:
            if isinstance(item, dict):
                item_id = item.get('id') or item.get('sequence')
                if item_id and item_id not in seen:
                    seen[item_id] = True
                    result.append(item)
            else:
                result.append(item)
        
        # Process second sequence
        for item in seq2:
            if isinstance(item, dict):
                item_id = item.get('id') or item.get('sequence')
                if item_id and item_id not in seen:
                    seen[item_id] = True
                    result.append(item)
            else:
                if item not in result:
                    result.append(item)
        
        return result
    
    @staticmethod
    def merge_json_metadata(meta1: Optional[Dict[str, Any]], 
                           meta2: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Merge two JSON metadata dictionaries.
        Uses Last-Write-Wins for conflicting keys.
        
        Args:
            meta1: First metadata dictionary
            meta2: Second metadata dictionary
            
        Returns:
            Merged metadata dictionary
        """
        if meta1 is None:
            return meta2
        if meta2 is None:
            return meta1
        
        # Start with meta1, then update with meta2 (LWW)
        merged = meta1.copy()
        merged.update(meta2)
        return merged

