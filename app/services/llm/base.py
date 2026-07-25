from abc import ABC, abstractmethod

from app.schemas.chat import BookingDraft
from app.services.memory.base import ChatMessage


class LLMProvider(ABC):
    @abstractmethod
    async def generate_answer(
        self,
        question: str,
        history: list[ChatMessage],
        context: str,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def extract_booking(
        self,
        message: str,
        history: list[ChatMessage],
        draft: BookingDraft | None,
    ) -> dict[str, str | None] | None:
        """Detect booking intent and extract any slot values the latest message updates.

        Returns None when the message is unrelated to booking. Otherwise returns a dict
        with an "intent" key ("book" or "none") plus any of name/email/date/time the
        message newly provided (missing/unmentioned slots are omitted or null).
        """
        raise NotImplementedError

