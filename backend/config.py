"""Central configuration, loaded from environment variables / .env."""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "groq:openai/gpt-oss-120b")

    linkup_api_key: str = os.getenv("LINKUP_API_KEY", "")
    linkup_depth: str = os.getenv("LINKUP_DEPTH", "standard")
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    search_snippet_chars: int = int(os.getenv("SEARCH_SNIPPET_CHARS", "500"))
    fetch_max_chars: int = int(os.getenv("FETCH_MAX_CHARS", "3000"))

    max_startups: int = int(os.getenv("MAX_STARTUPS", "4"))          # used only if the user gave no number
    max_startups_limit: int = int(os.getenv("MAX_STARTUPS_LIMIT", "10"))  # most a user can ask for

    # ---- cost control (Linkup is pay-per-call) ----
    max_searches_discovery: int = int(os.getenv("MAX_SEARCHES_DISCOVERY", "4"))
    max_searches_per_startup: int = int(os.getenv("MAX_SEARCHES_PER_STARTUP", "2"))
    max_page_reads_per_startup: int = int(os.getenv("MAX_PAGE_READS_PER_STARTUP", "1"))
    linkup_cache: bool = os.getenv("LINKUP_CACHE", "true").lower() in ("1", "true", "yes")
    cache_dir: str = os.getenv("CACHE_DIR", ".cache")
    deep_dive_concurrency: int = int(os.getenv("DEEP_DIVE_CONCURRENCY", "2"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model():
    """Chat model shared by all agents. Retries automatically on 429 rate-limit errors
    (important on free tiers such as Groq or Gemini)."""
    from langchain.chat_models import init_chat_model

    model = get_settings().llm_model
    extra = {}
    if "gpt-oss" in model:
        # gpt-oss is a reasoning model: low effort = faster and far fewer tokens (free tier = 8K tokens/min)
        extra["reasoning_effort"] = "low"
    return init_chat_model(model, temperature=0, max_retries=6, **extra)
