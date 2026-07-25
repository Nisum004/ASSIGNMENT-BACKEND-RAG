from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories.document_repository import DocumentRepository
from app.schemas.ingestion import ChunkStrategy, IngestionResponse
from app.services.chunking.factory import get_chunker
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.factory import get_embedding_provider
from app.services.text_extractor import TextExtractor
from app.services.vectorstores.base import VectorStore
from app.services.vectorstores.factory import get_vector_store


class DocumentIngestionService:
    def __init__(
        self,
        db: Session,
        extractor: TextExtractor | None = None,
        embeddings: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.repository = DocumentRepository(db)
        self.extractor = extractor or TextExtractor()
        self.embeddings = embeddings or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()

    async def ingest(self, file: UploadFile, strategy: ChunkStrategy) -> IngestionResponse:
        self._validate_file(file)
        text = await self.extractor.extract(file)
        chunks = get_chunker(strategy).chunk(text)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="No extractable text found in the uploaded document.",
            )

        document = self.repository.create_document(
            filename=file.filename or "uploaded-document",
            content_type=file.content_type or "application/octet-stream",
            chunk_strategy=strategy.value,
        )
        vectors = await self.embeddings.embed_documents(chunks)
        vector_items: list[tuple[str, list[float], dict[str, str | int]]] = []
        chunk_rows: list[tuple[str, int, str]] = []

        for index, (chunk, embedding) in enumerate(zip(chunks, vectors, strict=True)):
            vector_id = str(uuid4())
            metadata = {
                "document_id": document.id,
                "chunk_index": index,
                "filename": document.filename,
                "chunk_strategy": strategy.value,
                "text": chunk,
            }
            vector_items.append((vector_id, embedding, metadata))
            chunk_rows.append((vector_id, index, chunk))

        self.vector_store.upsert(vector_items)
        self.repository.add_chunks(document.id, chunk_rows)
        self.repository.commit()

        return IngestionResponse(
            document_id=document.id,
            filename=document.filename,
            chunk_strategy=strategy,
            chunks_stored=len(chunks),
        )

    def _validate_file(self, file: UploadFile) -> None:
        suffix = Path(file.filename or "").suffix.lower()
        has_supported_type = file.content_type in settings.supported_content_types
        has_supported_extension = suffix in {".pdf", ".txt"}
        if not has_supported_type and not has_supported_extension:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only .pdf and .txt files are supported.",
            )
