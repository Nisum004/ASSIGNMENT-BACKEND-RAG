from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.services.vectorstores.base import VectorDocument, VectorStore


class PineconeVectorStore(VectorStore):
    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is required when VECTOR_PROVIDER=pinecone.")
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._ensure_index()
        self.index = self.client.Index(self.index_name)

    def _ensure_index(self) -> None:
        indexes = self.client.list_indexes()
        existing_names = set(indexes.names()) if hasattr(indexes, "names") else {
            getattr(index, "name", index["name"]) for index in indexes
        }
        if self.index_name in existing_names:
            return
        self.client.create_index(
            name=self.index_name,
            dimension=settings.vector_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )

    def upsert(self, items: list[tuple[str, list[float], dict[str, str | int]]]) -> None:
        if not items:
            return
        vectors = [
            {"id": vector_id, "values": embedding, "metadata": metadata}
            for vector_id, embedding, metadata in items
        ]
        self.index.upsert(vectors=vectors)

    def query(self, vector: list[float], top_k: int) -> list[VectorDocument]:
        result = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        matches = getattr(result, "matches", []) or []
        documents: list[VectorDocument] = []
        for match in matches:
            metadata = dict(match.metadata or {})
            documents.append(
                VectorDocument(
                    id=match.id,
                    text=str(metadata.get("text", "")),
                    score=float(match.score or 0.0),
                    metadata=metadata,
                )
            )
        return documents
