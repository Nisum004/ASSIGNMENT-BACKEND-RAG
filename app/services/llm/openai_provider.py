import json

from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.chat import BookingDraft
from app.services.llm.base import LLMProvider
from app.services.memory.base import ChatMessage


class OpenAILLMProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI LLM calls.")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    async def generate_answer(
        self,
        question: str,
        history: list[ChatMessage],
        context: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise RAG assistant. Answer from the supplied context when "
                    "possible. If the answer is not in context, say what is missing. Keep the "
                    "conversation natural and use chat history only to resolve references."
                ),
            },
            {"role": "system", "content": f"Retrieved context:\n{context or 'No context found.'}"},
        ]
        messages.extend({"role": msg.role, "content": msg.content} for msg in history)
        messages.append({"role": "user", "content": question})
        response = await self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""

    async def extract_booking(
        self,
        message: str,
        history: list[ChatMessage],
        draft: BookingDraft | None,
    ) -> dict[str, str | None] | None:
        history_text = "\n".join(f"{item.role}: {item.content}" for item in history[-8:])
        draft_text = json.dumps(draft.model_dump() if draft else {})
        response = await self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You track interview booking requests across a multi-turn conversation. "
                        "Some booking slots (name, email, date, time) may already be collected — "
                        "they are given as the current draft. Read the latest message and decide: "
                        "1) intent: \"book\" if the user is requesting, continuing, or confirming "
                        "an interview booking (including just supplying a missing slot you asked "
                        'for); otherwise "none". 2) For each of name/email/date/time, return the '
                        "value ONLY if the latest message newly states or updates it; otherwise "
                        "return null for that field — do not repeat values already present in the "
                        "draft. Return strict JSON with keys intent, name, email, date, time."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current draft:\n{draft_text}\n\nHistory:\n{history_text}\n\n"
                        f"Latest:\n{message}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if parsed.get("intent") != "book":
            return None
        return {key: parsed.get(key) for key in ("name", "email", "date", "time")}
