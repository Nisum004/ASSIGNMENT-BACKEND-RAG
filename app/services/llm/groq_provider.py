from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.openai_provider import OpenAILLMProvider


class GroqLLMProvider(OpenAILLMProvider):
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")
        self.client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self.model = settings.llm_model
