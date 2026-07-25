"""In-memory fakes for the external-facing provider interfaces, used to keep unit
tests deterministic and network-free."""

from app.schemas.chat import BookingDraft
from app.services.embeddings.base import EmbeddingProvider
from app.services.llm.base import LLMProvider
from app.services.memory.base import BookingMemory, ChatMemory, ChatMessage
from app.services.vectorstores.base import VectorDocument, VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        return [float(len(text) % (index + 2)) for index in range(self.dimension)]


class FakeVectorStore(VectorStore):
    def __init__(self, query_results: list[VectorDocument] | None = None) -> None:
        self.upserted: list[tuple[str, list[float], dict[str, str | int]]] = []
        self.query_results = query_results or []

    def upsert(self, items: list[tuple[str, list[float], dict[str, str | int]]]) -> None:
        self.upserted.extend(items)

    def query(self, vector: list[float], top_k: int) -> list[VectorDocument]:
        return self.query_results[:top_k]


class FakeLLMProvider(LLMProvider):
    def __init__(
        self,
        answer: str = "This is the answer.",
        booking_responses: list[dict[str, str | None] | None] | None = None,
    ) -> None:
        self.answer = answer
        self._booking_responses = list(booking_responses or [])
        self.generate_answer_calls: list[str] = []
        self.extract_booking_calls: list[tuple[str, BookingDraft | None]] = []

    async def generate_answer(self, question: str, history: list[ChatMessage], context: str) -> str:
        self.generate_answer_calls.append(question)
        return self.answer

    async def extract_booking(
        self,
        message: str,
        history: list[ChatMessage],
        draft: BookingDraft | None,
    ) -> dict[str, str | None] | None:
        self.extract_booking_calls.append((message, draft))
        if not self._booking_responses:
            return None
        return self._booking_responses.pop(0)


class FakeChatMemory(ChatMemory):
    def __init__(self) -> None:
        self.messages: dict[str, list[ChatMessage]] = {}

    async def append(self, session_id: str, message: ChatMessage) -> None:
        self.messages.setdefault(session_id, []).append(message)

    async def get_messages(self, session_id: str, limit: int = 12) -> list[ChatMessage]:
        return self.messages.get(session_id, [])[-limit:]


class FakeBookingMemory(BookingMemory):
    def __init__(self) -> None:
        self.drafts: dict[str, BookingDraft] = {}

    async def get_draft(self, session_id: str) -> BookingDraft | None:
        return self.drafts.get(session_id)

    async def save_draft(self, session_id: str, draft: BookingDraft) -> None:
        self.drafts[session_id] = draft

    async def clear_draft(self, session_id: str) -> None:
        self.drafts.pop(session_id, None)
