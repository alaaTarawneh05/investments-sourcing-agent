"""Central configuration, loaded from environment variables / .env."""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "google_genai:gemini-3.6-flash")

    linkup_api_key: str = os.getenv("LINKUP_API_KEY", "")
    linkup_depth: str = os.getenv("LINKUP_DEPTH", "standard")

    sendgrid_api_key: str = os.getenv("SENDGRID_API_KEY", "")
    sendgrid_from_email: str = os.getenv("SENDGRID_FROM_EMAIL", "")

    max_startups: int = int(os.getenv("MAX_STARTUPS", "8"))
    deep_dive_concurrency: int = int(os.getenv("DEEP_DIVE_CONCURRENCY", "3"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
