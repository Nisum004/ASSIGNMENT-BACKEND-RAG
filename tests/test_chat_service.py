import pytest

from app.models.booking import Booking
from app.schemas.chat import ChatRequest
from app.services.chat_service import ConversationalRAGService
from tests.fakes import (
    FakeBookingMemory,
    FakeChatMemory,
    FakeEmbeddingProvider,
    FakeLLMProvider,
    FakeVectorStore,
)


def _service(db_session, llm: FakeLLMProvider, booking_memory: FakeBookingMemory | None = None):
    return ConversationalRAGService(
        db_session,
        embeddings=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        memory=FakeChatMemory(),
        booking_memory=booking_memory or FakeBookingMemory(),
        llm=llm,
    )


@pytest.mark.asyncio
async def test_chat_answers_without_booking_intent(db_session) -> None:
    llm = FakeLLMProvider(answer="The document says X.")
    service = _service(db_session, llm)

    response = await service.chat(
        ChatRequest(session_id="s1", message="What does the document say?")
    )

    assert response.answer == "The document says X."
    assert response.booking is None
    assert response.booking_pending is False
    assert response.booking_id is None
    assert llm.generate_answer_calls == ["What does the document say?"]


@pytest.mark.asyncio
async def test_chat_multi_turn_booking_slot_filling(db_session) -> None:
    booking_memory = FakeBookingMemory()
    llm = FakeLLMProvider(
        booking_responses=[
            {"name": "Ada Lovelace", "email": "ada@example.com", "date": None, "time": None},
            {"name": None, "email": None, "date": "2026-08-12", "time": "14:00"},
        ]
    )
    service = _service(db_session, llm, booking_memory)

    first = await service.chat(
        ChatRequest(
            session_id="s1",
            message="Book an interview for Ada Lovelace, ada@example.com",
        )
    )
    assert first.booking is None
    assert first.booking_pending is True
    assert "date" in first.answer and "time" in first.answer

    second = await service.chat(
        ChatRequest(session_id="s1", message="August 12 at 14:00")
    )

    assert second.booking is not None
    assert second.booking.name == "Ada Lovelace"
    assert second.booking.date == "2026-08-12"
    assert second.booking_id is not None
    assert second.booking_pending is False
    assert db_session.query(Booking).count() == 1
    # A booking turn must never also run the unrelated document Q&A path —
    # that's what produced the contradictory "I can't schedule that" + confirmation replies.
    assert llm.generate_answer_calls == []


@pytest.mark.asyncio
async def test_chat_rejects_invalid_email_without_crashing(db_session) -> None:
    llm = FakeLLMProvider(
        booking_responses=[
            {"name": "Bob", "email": "not-an-email", "date": "2026-01-01", "time": "10:00"},
        ]
    )
    service = _service(db_session, llm)

    response = await service.chat(
        ChatRequest(session_id="s1", message="Book me in as Bob, not-an-email, Jan 1 10:00")
    )

    assert response.booking is None
    assert response.booking_pending is True
    assert db_session.query(Booking).count() == 0
    assert llm.generate_answer_calls == []
