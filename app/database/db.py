from app.extensions import db, migrate

def init_db(app):
    """Bind database and migration engine to the Flask app."""
    db.init_app(app)
    migrate.init_app(app, db)