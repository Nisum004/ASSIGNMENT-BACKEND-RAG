from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.services.vectorstores.base import VectorDocument, VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection_name
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.vector_dimension, distance=Distance.COSINE
                ),
            )

    def upsert(self, items: list[tuple[str, list[float], dict[str, str | int]]]) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=vector_id, vector=embedding, payload=metadata)
                for vector_id, embedding, metadata in items
            ],
        )

    def query(self, vector: list[float], top_k: int) -> list[VectorDocument]:
        matches = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
        )
        return [
            VectorDocument(
                id=str(match.id),
                text=str((match.payload or {}).get("text", "")),
                score=float(match.score),
                metadata=dict(match.payload or {}),
            )
            for match in matches
        ]

