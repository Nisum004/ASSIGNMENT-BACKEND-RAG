from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.schemas.ingestion import ChunkStrategy
from app.services.ingestion_service import DocumentIngestionService
from tests.fakes import FakeEmbeddingProvider, FakeVectorStore


def _text_upload(content: bytes, filename: str = "notes.txt") -> UploadFile:
    return UploadFile(
        BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": "text/plain"}),
    )


@pytest.mark.asyncio
async def test_ingest_stores_document_chunks_and_vectors(db_session) -> None:
    vector_store = FakeVectorStore()
    service = DocumentIngestionService(
        db_session,
        embeddings=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )
    content = ("This is a sentence about backend RAG systems. " * 50).encode("utf-8")

    response = await service.ingest(file=_text_upload(content), strategy=ChunkStrategy.recursive)

    assert response.chunks_stored > 1
    assert response.document_id > 0
    assert len(vector_store.upserted) == response.chunks_stored
    stored_metadata = vector_store.upserted[0][2]
    assert stored_metadata["document_id"] == response.document_id
    assert stored_metadata["filename"] == "notes.txt"


@pytest.mark.asyncio
async def test_ingest_generates_uuid_vector_ids_for_backend_portability(db_session) -> None:
    import uuid

    vector_store = FakeVectorStore()
    service = DocumentIngestionService(
        db_session,
        embeddings=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )
    content = b"Short document body for a single chunk."

    await service.ingest(file=_text_upload(content), strategy=ChunkStrategy.fixed)

    vector_id = vector_store.upserted[0][0]
    uuid.UUID(vector_id)  # raises ValueError if not a canonical UUID string


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_file_type(db_session) -> None:
    service = DocumentIngestionService(
        db_session,
        embeddings=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )
    upload = UploadFile(
        BytesIO(b"binary"),
        filename="image.png",
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.ingest(file=upload, strategy=ChunkStrategy.fixed)

    assert exc_info.value.status_code == 415


@pytest.mark.asyncio
async def test_ingest_rejects_empty_document(db_session) -> None:
    service = DocumentIngestionService(
        db_session,
        embeddings=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.ingest(file=_text_upload(b"   \n\n  "), strategy=ChunkStrategy.fixed)

    assert exc_info.value.status_code == 422
