import os
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base configuration — shared across all environments."""

    # App
    APP_NAME = os.getenv("APP_NAME", "NyrVexa")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    SECRET_KEY = os.getenv("SECRET_KEY")

    # Database
    SQLALCHEMY_DATABASE_URI = (
        os.getenv("SQLALCHEMY_DATABASE_URI")
        or os.getenv("DATABASE_URL")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,        # Auto-reconnect if connection drops
        "pool_recycle": 300,          # Recycle connections every 5 minutes
        "pool_size": 10,              # Max 10 connections in pool
        "max_overflow": 20,           # Allow 20 extra connections under load
    }


    # AI API Keys
    GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
    DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY")
    GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
    MISTRAL_API_KEY     = os.getenv("MISTRAL_API_KEY")
    GROQ_INTERVIEW_API_KEY    = os.getenv("GROQ_INTERVIEW_API_KEY") or os.getenv("GROQ_API_KEY_2")
    GEMINI_INTERVIEW_API_KEY  = os.getenv("GEMINI_INTERVIEW_API_KEY") or os.getenv("GEMINI_API_KEY_2")
    MISTRAL_INTERVIEW_API_KEY = os.getenv("MISTRAL_INTERVIEW_API_KEY") or os.getenv("MISTRAL_API_KEY_2")

    # AI Model Names
    GROQ_MODEL_FAST     = "llama-3.3-70b-versatile"
    GROQ_MODEL_FALLBACK = "llama3-8b-8192"
    DEEPSEEK_MODEL      = "deepseek-chat"
    DEEPSEEK_CODER      = "deepseek-coder"
    GEMINI_MODEL        = "gemini-2.5-flash"
    MISTRAL_MODEL       = "mistral-small-latest"

    # Chat Config
    MAX_CHAT_HISTORY    = 10   # last N messages sent as context
    MAX_TOKENS_RESPONSE = 1024
    CHAT_TEMPERATURE    = 0.7


    # JWT
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        hours=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_HOURS", 1))
    )
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(
        days=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES_DAYS", 30))
    )

    # Google OAuth
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

    # CORS
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # Rate Limiting
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per day, 50 per hour")
    RATELIMIT_STORAGE_URL = os.getenv("REDIS_URL") or os.getenv("RATELIMIT_STORAGE_URL", "memory://")

      # Razorpay
    RAZORPAY_KEY_ID         = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET     = os.getenv("RAZORPAY_KEY_SECRET", "")
    RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    # Team admin (job posting dashboard) — /teamadmin on frontend
    TEAM_ADMIN_EMAIL    = os.getenv("TEAM_ADMIN_EMAIL", "team@nyrvexa.in").strip()
    TEAM_ADMIN_PASSWORD = os.getenv("TEAM_ADMIN_PASSWORD", "").strip()
    TEAM_ADMIN_ALERT_EMAIL = os.getenv("TEAM_ADMIN_ALERT_EMAIL", TEAM_ADMIN_EMAIL).strip()
    # Local dev only: fixed OTP when email is not configured (never set in production)
    TEAM_ADMIN_DEV_OTP   = os.getenv("TEAM_ADMIN_DEV_OTP", "").strip()


    # AI Engine Phase 1 & 2
    # Safe numeric helpers: fall back to default on non-numeric values and clamp to [min_val, max_val].
    @staticmethod
    def _safe_int(value: str, default: int, min_val: int, max_val: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_val, min(parsed, max_val))

    @staticmethod
    def _safe_float(value: str, default: float, min_val: float, max_val: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(min_val, min(parsed, max_val))

    AI_ENGINE_ENABLED = os.getenv("AI_ENGINE_ENABLED", "false").lower() in ("true", "1", "yes")
    AI_ENGINE_SEARCH_PROVIDER = os.getenv("AI_ENGINE_SEARCH_PROVIDER", "duckduckgo")
    AI_ENGINE_FETCH_TIMEOUT_SECONDS = _safe_int.__func__(
        os.getenv("AI_ENGINE_FETCH_TIMEOUT_SECONDS", "5"),
        default=5, min_val=1, max_val=30
    )
    AI_ENGINE_MAX_FETCH_WORKERS = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_FETCH_WORKERS", "4"),
        default=4, min_val=1, max_val=10
    )

    # AI Engine Phase 2 Retrieval & Evidence Context Budget
    AI_ENGINE_MAX_EVIDENCE_ITEMS = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_EVIDENCE_ITEMS", "10"),
        default=10, min_val=1, max_val=50
    )
    AI_ENGINE_MAX_EVIDENCE_CHARS = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_EVIDENCE_CHARS", "12000"),
        default=12000, min_val=500, max_val=100000
    )
    AI_ENGINE_MAX_CHUNK_CHARS = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_CHUNK_CHARS", "1000"),
        default=1000, min_val=100, max_val=5000
    )
    AI_ENGINE_CHUNK_OVERLAP_CHARS = _safe_int.__func__(
        os.getenv("AI_ENGINE_CHUNK_OVERLAP_CHARS", "100"),
        default=100, min_val=0, max_val=500
    )
    AI_ENGINE_MAX_CANDIDATES = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_CANDIDATES", "20"),
        default=20, min_val=1, max_val=100
    )
    AI_ENGINE_MIN_RELEVANCE_SCORE = _safe_float.__func__(
        os.getenv("AI_ENGINE_MIN_RELEVANCE_SCORE", "0.10"),
        default=0.10, min_val=0.0, max_val=1.0
    )

    # AI Engine Phase 3 Reasoning & Generation
    AI_ENGINE_LLM_ENABLED = os.getenv("AI_ENGINE_LLM_ENABLED", "false").lower() in ("true", "1", "yes")
    AI_ENGINE_LLM_PROVIDER = os.getenv("AI_ENGINE_LLM_PROVIDER", "gemini").lower()
    AI_ENGINE_LLM_MODEL = os.getenv("AI_ENGINE_LLM_MODEL", "gemini-2.5-flash")
    AI_ENGINE_OLLAMA_BASE_URL = os.getenv("AI_ENGINE_OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
    AI_ENGINE_LLM_TIMEOUT_SECONDS = _safe_int.__func__(
        os.getenv("AI_ENGINE_LLM_TIMEOUT_SECONDS", "30"),
        default=30, min_val=5, max_val=120
    )
    AI_ENGINE_LLM_MAX_TOKENS = _safe_int.__func__(
        os.getenv("AI_ENGINE_LLM_MAX_TOKENS", "1024"),
        default=1024, min_val=128, max_val=4096
    )
    AI_ENGINE_LLM_MAX_RETRIES = _safe_int.__func__(
        os.getenv("AI_ENGINE_LLM_MAX_RETRIES", "2"),
        default=2, min_val=0, max_val=5
    )
    AI_ENGINE_LLM_TEMPERATURE = _safe_float.__func__(
        os.getenv("AI_ENGINE_LLM_TEMPERATURE", "0.2"),
        default=0.2, min_val=0.0, max_val=1.0
    )

    # AI Engine Phase 4 Document Ingestion & Knowledge Base
    AI_ENGINE_DOCUMENTS_ENABLED = os.getenv("AI_ENGINE_DOCUMENTS_ENABLED", "true").lower() in ("true", "1", "yes")
    AI_ENGINE_MAX_DOC_SIZE_BYTES = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_DOC_SIZE_BYTES", "5242880"),
        default=5242880, min_val=1048576, max_val=20971520
    )
    AI_ENGINE_MAX_DOCS_PER_USER = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_DOCS_PER_USER", "10"),
        default=10, min_val=1, max_val=50
    )
    AI_ENGINE_MAX_DOC_CHUNKS_PER_USER = _safe_int.__func__(
        os.getenv("AI_ENGINE_MAX_DOC_CHUNKS_PER_USER", "250"),
        default=250, min_val=50, max_val=2000
    )

    # Interview AI Config
    INTERVIEW_SESSION_EXPIRY = 3600  # 1 hour in seconds
    MAX_INTERVIEW_QUESTIONS = 10
    FILLER_WORDS = [
        "um", "uh", "like", "you know", "basically",
        "literally", "actually", "right", "so"
    ]


class DevelopmentConfig(Config):
    """Development environment — extra debug info enabled."""
    DEBUG = True
    SQLALCHEMY_ECHO = False  # Set True if you want to see raw SQL queries


class ProductionConfig(Config):
    """Production environment — strict and secure."""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    RATELIMIT_STORAGE_URL = os.getenv("REDIS_URL") or os.getenv("RATELIMIT_STORAGE_URL", "memory://")  # Switch to Redis URL when scaling


class TestingConfig(Config):
    """Testing environment — uses separate in-memory DB."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {}
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)


# Config selector — used by app factory
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
