"""Central configuration, loaded from environment variables / .env."""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "groq:llama-3.3-70b-versatile")

    linkup_api_key: str = os.getenv("LINKUP_API_KEY", "")
    linkup_depth: str = os.getenv("LINKUP_DEPTH", "standard")
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    search_snippet_chars: int = int(os.getenv("SEARCH_SNIPPET_CHARS", "500"))
    fetch_max_chars: int = int(os.getenv("FETCH_MAX_CHARS", "3000"))

    sendgrid_api_key: str = os.getenv("SENDGRID_API_KEY", "")
    sendgrid_from_email: str = os.getenv("SENDGRID_FROM_EMAIL", "")

    max_startups: int = int(os.getenv("MAX_STARTUPS", "5"))
    deep_dive_concurrency: int = int(os.getenv("DEEP_DIVE_CONCURRENCY", "2"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model():
    """Chat model shared by all agents. Retries automatically on 429 rate-limit errors
    (important on free tiers such as Groq or Gemini)."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(get_settings().llm_model, temperature=0, max_retries=6)
