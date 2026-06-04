"""add public profile fields

Revision ID: 7d8e9f0a1b2c
Revises: fe7a4221a474
Create Date: 2026-06-04 02:30:00.000000

"""
import re

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "7d8e9f0a1b2c"
down_revision = "fe7a4221a474"
branch_labels = None
depends_on = None


def _slugify(value):
    base = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower())
    base = re.sub(r"-+", "-", base).strip("-")
    return (base or "user")[:72]


def _has_column(inspector, table, column):
    return any(col["name"] == column for col in inspector.get_columns(table))


def _has_index(inspector, table, index):
    return any(idx["name"] == index for idx in inspector.get_indexes(table))


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    columns = {
        "username": sa.Column("username", sa.String(length=80), nullable=True),
        "bio": sa.Column("bio", sa.Text(), nullable=True),
        "linkedin_url": sa.Column("linkedin_url", sa.String(length=255), nullable=True),
        "github_url": sa.Column("github_url", sa.String(length=255), nullable=True),
        "portfolio_url": sa.Column("portfolio_url", sa.String(length=255), nullable=True),
        "target_roles": sa.Column("target_roles", postgresql.ARRAY(sa.Text()), nullable=True),
        "location_label": sa.Column("location_label", sa.String(length=100), nullable=True),
        "theme_gradient": sa.Column("theme_gradient", sa.String(length=80), nullable=True),
        "public_profile_enabled": sa.Column("public_profile_enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        "show_interview_scores": sa.Column("show_interview_scores", sa.Boolean(), server_default=sa.true(), nullable=False),
        "show_skill_match": sa.Column("show_skill_match", sa.Boolean(), server_default=sa.true(), nullable=False),
        "show_resume_download": sa.Column("show_resume_download", sa.Boolean(), server_default=sa.false(), nullable=False),
    }

    with op.batch_alter_table("users") as batch_op:
        for name, column in columns.items():
            if not _has_column(inspector, "users", name):
                batch_op.add_column(column)

    users = bind.execute(sa.text("SELECT id, name, email, username FROM users ORDER BY created_at, id")).mappings()
    taken = set()
    for row in users:
        existing = (row.get("username") or "").strip().lower()
        if existing:
            taken.add(existing)
            continue

        base = _slugify(row.get("name") or (row.get("email") or "").split("@")[0])
        candidate = base
        suffix = 2
        while candidate in taken:
            suffix_text = f"-{suffix}"
            candidate = f"{base[:80 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        taken.add(candidate)
        bind.execute(
            sa.text("UPDATE users SET username = :username WHERE id = :id"),
            {"username": candidate, "id": row["id"]},
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "users", "ix_users_username"):
        op.create_index("ix_users_username", "users", ["username"])
    if not _has_index(inspector, "users", "uq_users_username"):
        op.create_index("uq_users_username", "users", ["username"], unique=True)


def downgrade():
    inspector = sa.inspect(op.get_bind())
    if _has_index(inspector, "users", "uq_users_username"):
        op.drop_index("uq_users_username", table_name="users")
    if _has_index(inspector, "users", "ix_users_username"):
        op.drop_index("ix_users_username", table_name="users")

    with op.batch_alter_table("users") as batch_op:
        for name in (
            "show_resume_download",
            "show_skill_match",
            "show_interview_scores",
            "public_profile_enabled",
            "theme_gradient",
            "location_label",
            "target_roles",
            "portfolio_url",
            "github_url",
            "linkedin_url",
            "bio",
            "username",
        ):
            if _has_column(inspector, "users", name):
                batch_op.drop_column(name)
