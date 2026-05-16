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
        hours=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_HOURS", 168))
    )
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(
        days=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES_DAYS", 30))
    )

    # Google OAuth
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

    # CORS
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Rate Limiting
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per day, 50 per hour")
    RATELIMIT_STORAGE_URL = "memory://"

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
    RATELIMIT_STORAGE_URL = "memory://"  # Switch to Redis URL when scaling


class TestingConfig(Config):
    """Testing environment — uses separate in-memory DB."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)


# Config selector — used by app factory
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
