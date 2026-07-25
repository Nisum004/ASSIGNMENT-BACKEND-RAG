from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider


def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "huggingface":
        from app.services.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider

        return HuggingFaceEmbeddingProvider()
    if settings.embedding_provider == "local_hash":
        from app.services.embeddings.local_hash_provider import LocalHashEmbeddingProvider

        return LocalHashEmbeddingProvider()
    from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider()
