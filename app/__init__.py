import logging
import os
import io
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from colorlog import ColoredFormatter

from app.database.db import init_db
from app.middleware.rate_limiter import limiter
from config import config_map
from app.routes.payment import payment_bp
from app.routes.usage import usage_bp

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
        from app.models import User, ChatSession, ChatMessage, InterviewSession, InterviewResponse  # noqa
        from app.models.job import Job
        from app.models.news_cache import NewsCache         # noqa
        from app.models.saved_job import SavedJob           # noqa
        from app.models.job_alert import JobAlert           # noqa
        from app.models.resume_analysis import ResumeAnalysis  # noqa
        from app.models.resume_template import ResumeTemplate  # noqa
        from app.models.user_resume import UserResume          # noqa
        from app.models.payment import Payment                 # noqa
        from app.models.support_ticket import SupportTicket     # noqa
        from app.models.admin_job import AdminJob               # noqa
        from app.extensions import db
        db.create_all()
        app.logger.info("Database tables verified (including admin_jobs).")

    # --- Initialize Rate Limiter ---
    limiter.init_app(app)

    # --- Initialize CORS ---
    CORS(
        app,
        resources={r"/api/*": {
            "origins": [
                "https://nyrvexa.in",
                "https://www.nyrvexa.in",
                "https://nyrvexa-frontend.vercel.app",
                "http://localhost:5173",
                "http://localhost:5174",
                "http://localhost:3000",
                "http://10.0.2.2",
                "http://10.0.2.2:5000",
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
            "max_age": 86400,
        }},
    )

    # --- Register Blueprints ---
    _register_blueprints(app)

    # --- Register Error Handlers ---
    _register_error_handlers(app)

    if not app.config.get("TESTING"):
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

    @app.route("/api/tts", methods=["POST"])
    def standalone_interviewer_tts():
        """Compatibility endpoint from the standalone Anya interviewer app."""
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        voice = (data.get("voice") or "ananya").strip().lower()

        if not text or len(text) > 3000:
            return jsonify({"error": "Invalid text length"}), 400

        try:
            from app.services.tts_service import text_to_speech
            audio_bytes = text_to_speech(text, voice=voice)
            return send_file(
                io.BytesIO(audio_bytes),
                mimetype="audio/mpeg",
                as_attachment=False,
                download_name="speech.mp3",
            )
        except Exception as e:
            app.logger.error("[TTS] POST /api/tts failed: %s", e)
            return jsonify({"error": "Failed to generate speech"}), 500

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

    from app.routes.career import career_bp
    app.register_blueprint(career_bp)

    from app.routes.roadmap_graph import roadmap_graph_bp
    app.register_blueprint(roadmap_graph_bp)

    app.register_blueprint(payment_bp)
    app.register_blueprint(usage_bp)

    from app.routes.admin import admin_bp
    app.register_blueprint(admin_bp)


def _register_error_handlers(app: Flask):
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad Request", "message": str(e)}), 400

    @app.errorhandler(401)
    def unauthorized(e):
        from flask import request
        response = jsonify({"error": "Unauthorized", "message": "Authentication required"})
        origin = request.headers.get("Origin")
        allowed = app.config.get("CORS_ORIGINS") or [
            "https://nyrvexa.in",
            "https://www.nyrvexa.in",
            "https://nyrvexa-frontend.vercel.app",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:3000",
        ]
        if origin in allowed:
            response.headers.add("Access-Control-Allow-Origin", origin)
            response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 401

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
