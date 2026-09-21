"""add vector embedding to ai_document_chunks

Revision ID: e7b9c1d2e3f4
Revises: fe7a4221a474
Create Date: 2026-09-21 15:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e7b9c1d2e3f4'
down_revision = 'fe7a4221a474'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        # 1. Enable pgvector extension if not already present
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Add nullable vector column
        op.execute("ALTER TABLE ai_document_chunks ADD COLUMN IF NOT EXISTS embedding vector(768);")

        # 3. Create HNSW index for high-recall cosine distance queries
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_ai_document_chunks_embedding "
            "ON ai_document_chunks USING hnsw (embedding vector_cosine_ops) "
            "WHERE embedding IS NOT NULL;"
        )
    else:
        # SQLite / generic fallback for testing migrations
        with op.batch_alter_table('ai_document_chunks', schema=None) as batch_op:
            batch_op.add_column(sa.Column('embedding', sa.Text(), nullable=True))


def downgrade():
    bind = op.get_bind()
    dialect_name = bind.dialect.name

    if dialect_name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_ai_document_chunks_embedding;")
        op.execute("ALTER TABLE ai_document_chunks DROP COLUMN IF EXISTS embedding;")
    else:
        with op.batch_alter_table('ai_document_chunks', schema=None) as batch_op:
            batch_op.drop_column('embedding')
