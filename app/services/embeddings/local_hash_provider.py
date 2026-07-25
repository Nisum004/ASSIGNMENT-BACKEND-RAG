import hashlib
import math

from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic lightweight embeddings for tests and offline smoke runs."""

    def __init__(self) -> None:
        self.dimension = settings.vector_dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

