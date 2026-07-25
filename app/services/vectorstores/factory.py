from app.core.config import settings
from app.services.vectorstores.base import VectorStore


def get_vector_store() -> VectorStore:
    if settings.vector_provider == "pinecone":
        from app.services.vectorstores.pinecone_store import PineconeVectorStore

        return PineconeVectorStore()
    if settings.vector_provider == "qdrant":
        from app.services.vectorstores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore()
    if settings.vector_provider == "weaviate":
        from app.services.vectorstores.weaviate_store import WeaviateVectorStore

        return WeaviateVectorStore()
    if settings.vector_provider == "milvus":
        from app.services.vectorstores.milvus_store import MilvusVectorStore

        return MilvusVectorStore()
    raise ValueError(f"Unsupported vector provider: {settings.vector_provider}")
