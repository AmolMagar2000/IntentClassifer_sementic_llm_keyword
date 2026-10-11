from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Intent Classifier Demo"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    GEMINI_API_KEY: str = ""
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GEMINI_LLM_MODEL: str = "gemini-3.5-flash"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "intent_utterances"

    SEMANTIC_THRESHOLD: float = 0.70
    LLM_THRESHOLD: float = 0.70
    TOP_K: int = 3
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = (
        "http://127.0.0.1:5500,http://localhost:5500,"
        "http://127.0.0.1:5501,http://localhost:5501"
    )

    @property
    def intents_file(self) -> Path:
        return self.BASE_DIR / "data" / "intents.json"

    @property
    def frontend_dir(self) -> Path:
        return self.BASE_DIR / "frontend"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
