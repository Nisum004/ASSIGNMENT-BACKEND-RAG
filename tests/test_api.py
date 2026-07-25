from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.services.vectorstores.base import VectorDocument
from tests.fakes import (
    FakeBookingMemory,
    FakeChatMemory,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeVectorStore,
)


@pytest.fixture()
def client(db_session, monkeypatch) -> Iterator[TestClient]:
    def override_get_db() -> Iterator:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        "app.services.ingestion_service.get_embedding_provider", FakeEmbeddingProvider
    )
    monkeypatch.setattr(
        "app.services.ingestion_service.get_vector_store", lambda: FakeVectorStore()
    )

    retrieved = [
        VectorDocument(
            id="doc-1",
            text="Backend RAG constraints go here.",
            score=0.9,
            metadata={"document_id": 1, "chunk_index": 0, "filename": "notes.txt"},
        )
    ]
    monkeypatch.setattr(
        "app.services.chat_service.get_embedding_provider", FakeEmbeddingProvider
    )
    monkeypatch.setattr(
        "app.services.chat_service.get_vector_store",
        lambda: FakeVectorStore(query_results=retrieved),
    )
    monkeypatch.setattr("app.services.chat_service.RedisChatMemory", FakeChatMemory)
    monkeypatch.setattr("app.services.chat_service.RedisBookingMemory", FakeBookingMemory)
    monkeypatch.setattr(
        "app.services.chat_service.get_llm_provider",
        lambda: FakeLLMProvider(answer="Based on the context, here is the answer."),
    )

    # Not entered as a context manager: the app's lifespan would create the real
    # sqlite file configured in settings, which the overridden get_db bypasses anyway.
    test_client = TestClient(app)
    yield test_client

    app.dependency_overrides.clear()


def test_health_check(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_endpoint_returns_chunk_count(client: TestClient) -> None:
    files = {"file": ("notes.txt", b"Backend RAG assignment content. " * 30, "text/plain")}

    response = client.post(
        "/api/v1/documents/ingest?chunk_strategy=fixed",
        files=files,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["chunk_strategy"] == "fixed"
    assert body["chunks_stored"] > 0


def test_ingest_endpoint_rejects_unsupported_file(client: TestClient) -> None:
    files = {"file": ("image.png", b"binary", "image/png")}

    response = client.post("/api/v1/documents/ingest", files=files)

    assert response.status_code == 415


def test_chat_endpoint_returns_answer_and_sources(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "api-session", "message": "What does the document say?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Based on the context, here is the answer."
    assert body["sources"][0]["filename"] == "notes.txt"
    assert body["booking_pending"] is False
