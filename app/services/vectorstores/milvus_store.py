from pymilvus import MilvusClient

from app.core.config import settings
from app.services.vectorstores.base import VectorDocument, VectorStore


class MilvusVectorStore(VectorStore):
    def __init__(self) -> None:
        self.client = MilvusClient(uri=settings.milvus_uri)
        self.collection_name = settings.milvus_collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=settings.vector_dimension,
                metric_type="COSINE",
            )

    def upsert(self, items: list[tuple[str, list[float], dict[str, str | int]]]) -> None:
        data = [
            {"id": vector_id, "vector": embedding, **metadata}
            for vector_id, embedding, metadata in items
        ]
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def query(self, vector: list[float], top_k: int) -> list[VectorDocument]:
        result = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["text", "document_id", "chunk_index", "filename"],
        )
        matches = result[0] if result else []
        documents: list[VectorDocument] = []
        for match in matches:
            entity = dict(match.get("entity", {}))
            documents.append(
                VectorDocument(
                    id=str(match.get("id")),
                    text=str(entity.get("text", "")),
                    score=float(match.get("distance", 0.0)),
                    metadata=entity,
                )
            )
        return documents

