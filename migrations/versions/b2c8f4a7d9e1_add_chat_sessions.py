"""add chat sessions

Revision ID: b2c8f4a7d9e1
Revises: fe7a4221a474
Create Date: 2026-05-15 17:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2c8f4a7d9e1'
down_revision = 'fe7a4221a474'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('title', sa.String(length=120), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('chat_sessions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_chat_sessions_user_id'), ['user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_chat_sessions_updated_at'), ['updated_at'], unique=False)

    with op.batch_alter_table('chat_history', schema=None) as batch_op:
        batch_op.add_column(sa.Column('session_id', sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            'fk_chat_history_session_id_chat_sessions',
            'chat_sessions',
            ['session_id'],
            ['id'],
            ondelete='CASCADE',
        )
        batch_op.create_index(batch_op.f('ix_chat_history_session_id'), ['session_id'], unique=False)


def downgrade():
    with op.batch_alter_table('chat_history', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_chat_history_session_id'))
        batch_op.drop_constraint('fk_chat_history_session_id_chat_sessions', type_='foreignkey')
        batch_op.drop_column('session_id')

    with op.batch_alter_table('chat_sessions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_chat_sessions_updated_at'))
        batch_op.drop_index(batch_op.f('ix_chat_sessions_user_id'))

    op.drop_table('chat_sessions')
