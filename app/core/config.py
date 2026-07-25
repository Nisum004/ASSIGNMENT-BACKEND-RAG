from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'rag.db'}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Assignment Backend RAG"
    environment: str = "local"
    api_prefix: str = "/api/v1"

    database_url: str = DEFAULT_DATABASE_URL
    redis_url: str = "redis://localhost:6379/0"

    vector_provider: Literal["pinecone", "qdrant", "weaviate", "milvus"] = "pinecone"
    vector_dimension: int = 384

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "rag-documents"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "rag_documents"

    weaviate_url: str = "http://localhost:8080"
    weaviate_collection_name: str = "RagDocument"

    milvus_uri: str = "http://localhost:19530"
    milvus_collection_name: str = "rag_documents"

    openai_api_key: str | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    huggingface_api_key: str | None = None
    huggingface_embedding_url: str = (
        "https://router.huggingface.co/hf-inference/models/"
        "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    )
    embedding_provider: Literal["huggingface", "openai", "local_hash"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: Literal["openai", "groq"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"

    default_top_k: int = 5
    max_upload_mb: int = 20
    chat_memory_ttl_seconds: int = 60 * 60 * 24 * 7
    supported_content_types: set[str] = Field(
        default_factory=lambda: {"application/pdf", "text/plain"}
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
