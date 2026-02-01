"""
Embedding Generator

Wraps sentence-transformers for generating text embeddings.
Supports batch processing and GPU acceleration.
"""

from typing import List
import logging
import torch
from sentence_transformers import SentenceTransformer

from config.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = 32
    ):
        """
        Initialize embedding model.

        Args:
            model_name: Name of sentence-transformers model (default from config)
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Successfully loaded model with embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def encode(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            logger.warning("Empty text list provided for encoding")
            return []

        try:
            # Encode in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )

            # Convert numpy arrays to lists for ChromaDB
            embeddings_list = embeddings.tolist()

            logger.debug(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def encode_query(self, query: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text
            normalize: Normalize embedding to unit length

        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.encode([query], show_progress=False, normalize=normalize)
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self):
        return f"EmbeddingGenerator(model={self.model_name}, device={self.device}, dim={self.get_embedding_dimension()})"
