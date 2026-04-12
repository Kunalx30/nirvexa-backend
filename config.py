import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration — shared across all environments."""

    # App
    APP_NAME = os.getenv("APP_NAME", "NirVexa")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    SECRET_KEY = os.getenv("SECRET_KEY")

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,        # Auto-reconnect if connection drops
        "pool_recycle": 300,          # Recycle connections every 5 minutes
        "pool_size": 10,              # Max 10 connections in pool
        "max_overflow": 20,           # Allow 20 extra connections under load
    }

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
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Rate Limiting
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per day, 50 per hour")
    RATELIMIT_STORAGE_URL = "memory://"


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