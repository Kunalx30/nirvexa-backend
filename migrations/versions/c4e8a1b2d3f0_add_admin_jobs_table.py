"""add admin_jobs table

Revision ID: c4e8a1b2d3f0
Revises: 9a7c1f2d4b6e
Create Date: 2026-05-24

"""
from alembic import op
import sqlalchemy as sa


revision = "c4e8a1b2d3f0"
down_revision = "9a7c1f2d4b6e"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "admin_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("company", sa.String(200), nullable=False),
        sa.Column("company_logo", sa.Text(), nullable=True),
        sa.Column("location", sa.String(200), nullable=True),
        sa.Column("job_type", sa.String(50), nullable=True),
        sa.Column("experience", sa.String(100), nullable=True),
        sa.Column("salary", sa.String(100), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("requirements", sa.Text(), nullable=True),
        sa.Column("skills", sa.ARRAY(sa.String()), nullable=True),
        sa.Column("apply_url", sa.Text(), nullable=True),
        sa.Column("apply_email", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_featured", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("posted_by", sa.String(100), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("admin_jobs")
