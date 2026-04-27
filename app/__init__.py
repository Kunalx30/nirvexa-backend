import logging
import os
from flask import Flask, jsonify
from flask_cors import CORS
from colorlog import ColoredFormatter

from app.database.db import init_db
from app.middleware.rate_limiter import limiter
from config import config_map

from app.services.scheduler import init_scheduler


def create_app(env: str = None) -> Flask:
    app = Flask(__name__)

    # --- Load Config ---
    env = env or os.getenv("FLASK_ENV", "development")
    app.config.from_object(config_map[env])

    # --- Initialize Logging ---
    _setup_logging(app)

    # --- Initialize Database + Migrate ---
    init_db(app)

    # --- Import ALL Models inside app context ---
    with app.app_context():
        from app.models import User, ChatMessage, InterviewSession, InterviewResponse  # noqa
        from app.models.job import Job         
        from app.models.news_cache import NewsCache         # noqa
        from app.models.saved_job import SavedJob           # noqa
        from app.models.job_alert import JobAlert           # noqa
        from app.models.resume_analysis import ResumeAnalysis  # noqa

    # --- Initialize Rate Limiter ---
    limiter.init_app(app)

    # --- Initialize CORS ---
    # Using resources dict so Flask-CORS applies headers on EVERY route
    # including preflight OPTIONS requests for PUT / DELETE
    CORS(
        app,
        resources={r"/api/*": {
            "origins": [
                # Web - production
                "https://nirvexa-frontend.vercel.app",
                # Web - local dev (all common Vite ports)
                "http://localhost:5173",
                "http://localhost:5174",
                "http://localhost:3000",
                # Android emulator hits 10.0.2.2 for host machine localhost
                "http://10.0.2.2",
                "http://10.0.2.2:5000",
                # Allow null origin (Android WebView / file:// during dev)
                "null",
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "Accept",
            ],
            "expose_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True,
            "max_age": 86400,  # Cache preflight for 24h — reduces OPTIONS spam
        }},
    )

    # --- Register Blueprints ---
    _register_blueprints(app)

    # --- Register Error Handlers ---
    _register_error_handlers(app)

    # --- Initialize Scheduler ---
    init_scheduler(app)

    # --- Initialize FAISS Semantic Search Index ---
    from app.services.rag_pipeline import load_or_build_index
    load_or_build_index(app)
    app.logger.info("FAISS index load triggered at startup.")

    # --- Health Check ---
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "app": app.config.get("APP_NAME", "NirVexa"),
            "version": app.config.get("APP_VERSION", "1.0.0"),
            "environment": env,
        }), 200

    app.logger.info(f"NirVexa backend started in [{env}] mode")
    return app


def _register_blueprints(app: Flask):
    from app.routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/api/auth")

    from app.routes.chat import chat_bp
    app.register_blueprint(chat_bp, url_prefix="/api")

    from app.routes.interview import interview_bp
    app.register_blueprint(interview_bp, url_prefix="/api/interview")

    from app.routes.jobs import jobs_bp
    app.register_blueprint(jobs_bp, url_prefix="/api/jobs")

    from app.routes.resume import resume_bp
    app.register_blueprint(resume_bp)

    from app.routes.user import user_bp
    app.register_blueprint(user_bp)

    from app.routes.news import news_bp
    app.register_blueprint(news_bp)


def _register_error_handlers(app: Flask):
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad Request", "message": str(e)}), 400

    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({"error": "Unauthorized", "message": "Authentication required"}), 401

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({"error": "Forbidden", "message": "You do not have permission"}), 403

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "message": "The requested resource does not exist"}), 404

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return jsonify({"error": "Too Many Requests", "message": "Rate limit exceeded. Please slow down."}), 429

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal Server Error", "message": "Something went wrong on our end"}), 500


def _setup_logging(app: Flask):
    log_level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    ))

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/nirvexa.log")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    app.logger.setLevel(log_level)
    app.logger.addHandler(console_handler)
    app.logger.addHandler(file_handler)