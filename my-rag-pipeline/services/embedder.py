import logging
from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Local sentence-transformers based embedder."""

    def __init__(self):
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer '%s': %s", EMBEDDING_MODEL_NAME, e)
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Return a single embedding vector for `text` as a list of floats."""
        try:
            emb = self.model.encode(text, show_progress_bar=False)
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as e:
            logger.exception("Error computing embedding for text: %s", e)
            return []

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        # Force sentence-transformers to utilize vector batching optimizations
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
