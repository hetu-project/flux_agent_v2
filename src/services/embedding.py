"""Embedding service for generating vector embeddings."""

from typing import List
from openai import OpenAI
from src.config import get_settings

settings = get_settings()


class EmbeddingService:
    """Service for generating embeddings using OpenAI or AIHubMix."""
    
    def __init__(self):
        # Support both OpenAI and AIHubMix
        api_key = settings.openai_api_key or settings.aihubmix_api_key
        base_url = None
        
        if settings.llm_provider == "aihubmix" and settings.aihubmix_api_key:
            base_url = settings.aihubmix_base_url
        elif settings.llm_provider == "openai" and settings.openai_api_key:
            base_url = settings.openai_base_url
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = settings.embedding_model
        self.dimension = 1536  # OpenAI ada-002 dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # OpenAI API supports up to 2048 inputs per batch
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

