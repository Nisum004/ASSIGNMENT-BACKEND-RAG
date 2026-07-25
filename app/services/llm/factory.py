from app.services.llm.base import LLMProvider


def get_llm_provider() -> LLMProvider:
    from app.core.config import settings

    if settings.llm_provider == "groq":
        from app.services.llm.groq_provider import GroqLLMProvider

        return GroqLLMProvider()

    from app.services.llm.openai_provider import OpenAILLMProvider

    return OpenAILLMProvider()
