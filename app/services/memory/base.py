from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.schemas.chat import BookingDraft


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ChatMemory(ABC):
    @abstractmethod
    async def append(self, session_id: str, message: ChatMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_messages(self, session_id: str, limit: int = 12) -> list[ChatMessage]:
        raise NotImplementedError


class BookingMemory(ABC):
    """Stores in-progress interview booking slots between chat turns."""

    @abstractmethod
    async def get_draft(self, session_id: str) -> BookingDraft | None:
        raise NotImplementedError

    @abstractmethod
    async def save_draft(self, session_id: str, draft: BookingDraft) -> None:
        raise NotImplementedError

    @abstractmethod
    async def clear_draft(self, session_id: str) -> None:
        raise NotImplementedError

