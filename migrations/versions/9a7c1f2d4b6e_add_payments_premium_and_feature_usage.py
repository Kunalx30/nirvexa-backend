"""add payments premium and feature usage

Revision ID: 9a7c1f2d4b6e
Revises: 131be334d3ed
Create Date: 2026-05-23 23:20:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "9a7c1f2d4b6e"
down_revision = "131be334d3ed"
branch_labels = None
depends_on = None


def _has_column(table_name, column_name):
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return column_name in [col["name"] for col in inspector.get_columns(table_name)]


def _has_table(table_name):
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade():
    if not _has_column("users", "is_premium"):
        op.add_column("users", sa.Column("is_premium", sa.Boolean(), nullable=False, server_default=sa.false()))
    if not _has_column("users", "premium_plan"):
        op.add_column("users", sa.Column("premium_plan", sa.String(length=50), nullable=True))
    if not _has_column("users", "premium_expiry"):
        op.add_column("users", sa.Column("premium_expiry", sa.DateTime(timezone=True), nullable=True))

    if not _has_table("payments"):
        op.create_table(
            "payments",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("razorpay_order_id", sa.String(length=100), nullable=False),
            sa.Column("razorpay_payment_id", sa.String(length=100), nullable=True),
            sa.Column("razorpay_signature", sa.String(length=256), nullable=True),
            sa.Column("amount", sa.Integer(), nullable=False),
            sa.Column("currency", sa.String(length=10), nullable=True),
            sa.Column("plan", sa.String(length=50), nullable=False),
            sa.Column("status", sa.String(length=30), nullable=True),
            sa.Column("payment_method", sa.String(length=50), nullable=True),
            sa.Column("upi_transaction_id", sa.String(length=100), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("razorpay_order_id"),
            sa.UniqueConstraint("razorpay_payment_id"),
        )

    if not _has_table("feature_usage"):
        op.create_table(
            "feature_usage",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("feature", sa.String(length=50), nullable=False),
            sa.Column("used_date", sa.Date(), nullable=False),
            sa.Column("count", sa.Integer(), nullable=True, server_default="0"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "feature", "used_date"),
        )


def downgrade():
    if _has_table("feature_usage"):
        op.drop_table("feature_usage")
    if _has_table("payments"):
        op.drop_table("payments")
    if _has_column("users", "premium_expiry"):
        op.drop_column("users", "premium_expiry")
    if _has_column("users", "premium_plan"):
        op.drop_column("users", "premium_plan")
    if _has_column("users", "is_premium"):
        op.drop_column("users", "is_premium")
