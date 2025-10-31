"""Embedding service for generating vector embeddings."""

from typing import List
from openai import OpenAI
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using AIHubMix."""
    
    def __init__(self):
        # Use AIHubMix for embeddings
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.client = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.model = settings.embedding_model
        self.dimension = 1536  # OpenAI ada-002 dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        logger.debug(f"Generating embedding for text (length: {len(text)})")
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            logger.debug("Embedding generated successfully")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        # OpenAI API supports up to 2048 inputs per batch
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch)} texts)")
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise
        
        logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
        return all_embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

