"""
app/ai_engine/embeddings/backfill.py
Safe, restartable, bounded offline backfill script for generating vector embeddings
for AIDocumentChunk records where embedding is NULL.
Does NOT run at application startup.
"""
import time
import logging
import argparse
from typing import Optional, Dict, Any

from app import create_app
from app.extensions import db
from app.models.ai_document import AIDocumentChunk
from app.ai_engine.embeddings.factory import EmbeddingProviderFactory
from app.ai_engine.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


def backfill_chunk_embeddings(
    batch_size: int = 50,
    max_chunks: Optional[int] = None,
    provider: Optional[BaseEmbeddingProvider] = None,
) -> Dict[str, Any]:
    """
    Finds chunks where embedding is NULL, batch embeds their text,
    and commits in bounded transactions.
    """
    if provider is None:
        provider = EmbeddingProviderFactory.create()

    logger.info(
        "[Backfill] Starting embedding backfill with provider=%s model=%s batch_size=%d",
        provider.name, provider.model, batch_size,
    )

    total_processed = 0
    total_failed = 0
    start_time = time.monotonic()

    while True:
        # Bounded query for unembedded chunks ordered deterministically
        query = (
            AIDocumentChunk.query.filter(AIDocumentChunk.embedding.is_(None))
            .order_by(AIDocumentChunk.created_at.asc(), AIDocumentChunk.id.asc())
            .limit(batch_size)
        )
        chunks = query.all()

        if not chunks:
            logger.info("[Backfill] No more unembedded chunks found.")
            break

        texts = [ch.text for ch in chunks]
        try:
            embeddings = provider.embed_batch(texts)
            for ch, emb in zip(chunks, embeddings):
                if emb is not None:
                    ch.embedding = emb
                    total_processed += 1
                else:
                    total_failed += 1

            db.session.commit()
            logger.info(
                "[Backfill] Successfully committed batch of %d chunks (cumulative processed=%d)",
                len(chunks), total_processed,
            )
        except Exception as exc:
            db.session.rollback()
            total_failed += len(chunks)
            logger.error("[Backfill] Batch embedding generation or commit failed: %s", exc)
            # Break to avoid infinite error loops on persistent external failure
            break

        if max_chunks is not None and total_processed >= max_chunks:
            logger.info("[Backfill] Reached max_chunks limit (%d). Stopping.", max_chunks)
            break

        # Brief pause between batches to prevent database / API throttling
        time.sleep(0.2)

    elapsed = round(time.monotonic() - start_time, 2)
    summary = {
        "status": "completed" if total_failed == 0 else "completed_with_errors",
        "chunks_processed": total_processed,
        "chunks_failed": total_failed,
        "elapsed_seconds": elapsed,
        "provider": provider.name,
    }
    logger.info("[Backfill] Finished backfill summary: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill vector embeddings for un-embedded AIDocumentChunks.")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of chunks to embed per batch.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of chunks to process.")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        backfill_chunk_embeddings(batch_size=args.batch_size, max_chunks=args.max_chunks)
