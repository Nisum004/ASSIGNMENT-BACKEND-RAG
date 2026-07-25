import httpx

from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        if not settings.huggingface_api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY is required for Hugging Face embeddings.")
        self.url = settings.huggingface_embedding_url
        self.headers = {
            "Authorization": f"Bearer {settings.huggingface_api_key}",
            "Content-Type": "application/json",
        }

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text]))[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self.url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
            response.raise_for_status()
        payload = response.json()
        return self._normalize_response(payload)

    def _normalize_response(self, payload: object) -> list[list[float]]:
        if not isinstance(payload, list):
            raise TypeError("Unexpected Hugging Face embedding response.")

        first = payload[0] if payload else None
        is_token_level = (
            isinstance(first, list) and first and isinstance(first[0], list)
        )
        if is_token_level:
            return [self._mean_pool(tokens) for tokens in payload]  # type: ignore[arg-type]

        return [[float(value) for value in vector] for vector in payload]  # type: ignore[union-attr]

    def _mean_pool(self, token_vectors: list[list[float]]) -> list[float]:
        if not token_vectors:
            return [0.0] * settings.vector_dimension
        dimensions = len(token_vectors[0])
        return [
            sum(token_vector[index] for token_vector in token_vectors) / len(token_vectors)
            for index in range(dimensions)
        ]
