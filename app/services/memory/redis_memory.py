import json

from redis.asyncio import Redis

from app.core.config import settings
from app.schemas.chat import BookingDraft
from app.services.memory.base import BookingMemory, ChatMemory, ChatMessage


class RedisChatMemory(ChatMemory):
    def __init__(self) -> None:
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)

    async def append(self, session_id: str, message: ChatMessage) -> None:
        key = self._key(session_id)
        payload = json.dumps({"role": message.role, "content": message.content})
        await self.redis.rpush(key, payload)
        await self.redis.expire(key, settings.chat_memory_ttl_seconds)

    async def get_messages(self, session_id: str, limit: int = 12) -> list[ChatMessage]:
        key = self._key(session_id)
        raw_items = await self.redis.lrange(key, -limit, -1)
        messages: list[ChatMessage] = []
        for raw_item in raw_items:
            item = json.loads(raw_item)
            messages.append(ChatMessage(role=item["role"], content=item["content"]))
        return messages

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}:messages"


class RedisBookingMemory(BookingMemory):

    def __init__(self) -> None:
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)

    async def get_draft(self, session_id: str) -> BookingDraft | None:
        raw = await self.redis.get(self._key(session_id))
        if raw is None:
            return None
        return BookingDraft.model_validate_json(raw)

    async def save_draft(self, session_id: str, draft: BookingDraft) -> None:
        await self.redis.set(
            self._key(session_id),
            draft.model_dump_json(),
            ex=settings.chat_memory_ttl_seconds,
        )

    async def clear_draft(self, session_id: str) -> None:
        await self.redis.delete(self._key(session_id))

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}:booking_draft"

