import weaviate

from app.core.config import settings
from app.services.vectorstores.base import VectorDocument, VectorStore


class WeaviateVectorStore(VectorStore):
    def __init__(self) -> None:
        self.client = weaviate.connect_to_local(host=settings.weaviate_url.replace("http://", ""))
        self.collection_name = settings.weaviate_collection_name

    def upsert(self, items: list[tuple[str, list[float], dict[str, str | int]]]) -> None:
        collection = self.client.collections.get(self.collection_name)
        with collection.batch.dynamic() as batch:
            for vector_id, embedding, metadata in items:
                batch.add_object(properties=metadata, uuid=vector_id, vector=embedding)

    def query(self, vector: list[float], top_k: int) -> list[VectorDocument]:
        collection = self.client.collections.get(self.collection_name)
        result = collection.query.near_vector(near_vector=vector, limit=top_k)
        documents: list[VectorDocument] = []
        for item in result.objects:
            metadata = dict(item.properties or {})
            documents.append(
                VectorDocument(
                    id=str(item.uuid),
                    text=str(metadata.get("text", "")),
                    score=0.0,
                    metadata=metadata,
                )
            )
        return documents

