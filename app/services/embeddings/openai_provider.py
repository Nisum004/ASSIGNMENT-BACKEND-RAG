from openai import AsyncOpenAI

from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

