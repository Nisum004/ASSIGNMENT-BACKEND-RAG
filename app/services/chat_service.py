from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories.booking_repository import BookingRepository
from app.schemas.chat import BookingDetails, BookingDraft, ChatRequest, ChatResponse
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.factory import get_embedding_provider
from app.services.llm.base import LLMProvider
from app.services.llm.factory import get_llm_provider
from app.services.memory.base import BookingMemory, ChatMemory, ChatMessage
from app.services.memory.redis_memory import RedisBookingMemory, RedisChatMemory
from app.services.vectorstores.base import VectorDocument, VectorStore
from app.services.vectorstores.factory import get_vector_store


@dataclass(frozen=True)
class BookingOutcome:
    """for booking intent."""

    booking: BookingDetails | None = None
    follow_up: str | None = None

    @property
    def is_pending(self) -> bool:
        return self.follow_up is not None


class ConversationalRAGService:
    def __init__(
        self,
        db: Session,
        embeddings: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        memory: ChatMemory | None = None,
        booking_memory: BookingMemory | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.embeddings = embeddings or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()
        self.memory = memory or RedisChatMemory()
        self.booking_memory = booking_memory or RedisBookingMemory()
        self.llm = llm or get_llm_provider()
        self.booking_repository = BookingRepository(db)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        history = await self.memory.get_messages(request.session_id)

        outcome = await self._process_booking(request.session_id, request.message, history)
        booking_id = None
        retrieved: list[VectorDocument] = []

        if outcome.booking:
            stored = self.booking_repository.create(request.session_id, outcome.booking)
            booking_id = stored.id
            answer = (
                f"Interview booked for {outcome.booking.name} on "
                f"{outcome.booking.date} at {outcome.booking.time}."
            )
        elif outcome.is_pending:
            answer = outcome.follow_up or ""
        else:
            retrieved = await self._retrieve(
                request.message, request.top_k or settings.default_top_k
            )
            context = self._format_context(retrieved)
            answer = await self.llm.generate_answer(
                question=request.message,
                history=history,
                context=context,
            )

        await self.memory.append(
            request.session_id, ChatMessage(role="user", content=request.message)
        )
        await self.memory.append(request.session_id, ChatMessage(role="assistant", content=answer))

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            sources=self._sources(retrieved),
            booking=outcome.booking,
            booking_id=booking_id,
            booking_pending=outcome.is_pending,
        )

    async def _retrieve(self, question: str, top_k: int) -> list[VectorDocument]:
        query_vector = await self.embeddings.embed_query(question)
        return self.vector_store.query(query_vector, top_k=top_k)

    async def _process_booking(
        self,
        session_id: str,
        message: str,
        history: list[ChatMessage],
    ) -> BookingOutcome:
        draft = await self.booking_memory.get_draft(session_id) or BookingDraft()
        extracted = await self.llm.extract_booking(message, history, draft)
        if not extracted:
            return BookingOutcome()

        merged = BookingDraft(
            name=extracted.get("name") or draft.name,
            email=extracted.get("email") or draft.email,
            date=extracted.get("date") or draft.date,
            time=extracted.get("time") or draft.time,
        )

        if not merged.is_complete():
            await self.booking_memory.save_draft(session_id, merged)
            missing = " and ".join(merged.missing_fields())
            return BookingOutcome(
                follow_up=f"To complete the interview booking, could you share your {missing}?"
            )

        try:
            booking_details = BookingDetails.model_validate(merged.model_dump())
        except ValidationError:
            await self.booking_memory.save_draft(session_id, merged)
            return BookingOutcome(
                follow_up=(
                    "One of the booking details doesn't look right — could you confirm your "
                    "name, email, date, and time again?"
                )
            )

        await self.booking_memory.clear_draft(session_id)
        return BookingOutcome(booking=booking_details)

    def _format_context(self, documents: list[VectorDocument]) -> str:
        blocks = []
        for index, document in enumerate(documents, start=1):
            filename = document.metadata.get("filename", "unknown")
            blocks.append(f"[{index}] file={filename} score={document.score:.4f}\n{document.text}")
        return "\n\n".join(blocks)

    def _sources(self, documents: list[VectorDocument]) -> list[dict[str, str | int]]:
        sources: list[dict[str, str | int]] = []
        for document in documents:
            sources.append(
                {
                    "vector_id": document.id,
                    "document_id": int(document.metadata.get("document_id", 0)),
                    "chunk_index": int(document.metadata.get("chunk_index", 0)),
                    "filename": str(document.metadata.get("filename", "unknown")),
                }
            )
        return sources
