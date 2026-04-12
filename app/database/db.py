from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# These are initialized here and bound to the app in app/__init__.py
db = SQLAlchemy()
migrate = Migrate()


def init_db(app):
    """Bind database and migration engine to the Flask app."""
    db.init_app(app)
    migrate.init_app(app, db)