"""
app/services/rag_pipeline.py
NirVexa — Phase 5.6: FAISS Semantic Search
Uses Google Gemini embedding API (free, zero memory cost on Render free tier)
No sentence-transformers, no torch — works within 512MB RAM limit.
Gemini model: models/gemini-embedding-001
Dimensions: 768
Rate limit fix: embeds ONE text at a time with retry-backoff on 429
"""

import logging
import os
import threading
import time
import numpy as np

import faiss
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL      = "models/gemini-embedding-001"
VECTOR_DIM        = 768
INDEX_PATH        = "nirvexa_jobs.index"
INTER_EMBED_SLEEP = 0.7    # ~85 requests/min — safely under the 100/min free limit
MAX_RETRIES       = 5
RETRY_BASE_DELAY  = 15.0   # seconds; doubles each retry on 429

# ── Module-level state ────────────────────────────────────────────────────────
_index:   faiss.Index | None = None
_id_map:  list               = []    # position → job UUID string
_lock     = threading.Lock()
_building = False                    # prevent concurrent rebuilds


# ── Gemini setup ──────────────────────────────────────────────────────────────
def _configure_gemini():
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_AI_KEY")
    )
    if not api_key:
        raise RuntimeError("No Gemini API key found in environment variables.")
    genai.configure(api_key=api_key)


# ── Embedding ─────────────────────────────────────────────────────────────────
def _embed_single(text: str, task_type: str = "retrieval_document"):
    """
    Embed ONE text string via Gemini with retry-backoff on 429.
    Returns the embedding list or None on permanent failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = genai.embed_content(
                model=GEMINI_MODEL,
                content=text,           # ← always a single string, never a list
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "[FAISS] 429 rate-limit (attempt %d/%d) — sleeping %.0fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                logger.error("[FAISS] embed_content error: %s", e)
                return None

    logger.error("[FAISS] Gave up embedding after %d retries", MAX_RETRIES)
    return None


def _embed_texts(texts):
    """
    Embed a list of strings one-by-one with INTER_EMBED_SLEEP between each.
    Returns (N, 768) float32 numpy array. Also returns list of indices that succeeded
    so id_map can be aligned correctly.
    """
    _configure_gemini()
    vectors        = []
    success_indices = []
    failed         = 0

    for i, text in enumerate(texts):
        vec = _embed_single(text)
        if vec is not None:
            vectors.append(vec)
            success_indices.append(i)
        else:
            failed += 1

        # Sleep between every embed to stay under 100 req/min
        if i < len(texts) - 1:
            time.sleep(INTER_EMBED_SLEEP)

        # Progress log every 50 jobs
        if (i + 1) % 50 == 0:
            logger.info(
                "[FAISS] Embedded %d/%d jobs (failed: %d)", i + 1, len(texts), failed
            )

    if failed:
        logger.warning(
            "[FAISS] %d/%d embeddings failed and were skipped", failed, len(texts)
        )

    return np.array(vectors, dtype="float32"), success_indices


def _embed_query(query: str):
    """Embed a single search query. Returns (1, 768) float32 array."""
    _configure_gemini()
    vec = _embed_single(query, task_type="retrieval_query")
    if vec is None:
        raise RuntimeError("Failed to embed search query")
    return np.array([vec], dtype="float32")


# ── Disk persistence ──────────────────────────────────────────────────────────
def _save_to_disk(index, id_map):
    faiss.write_index(index, INDEX_PATH)
    with open(INDEX_PATH + ".ids", "w") as f:
        f.write("\n".join(str(i) for i in id_map))
    logger.info("[FAISS] Saved to disk.")


def _load_from_disk():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(INDEX_PATH + ".ids"):
        return None
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(INDEX_PATH + ".ids") as f:
            id_map = [line.strip() for line in f if line.strip()]
        logger.info("[FAISS] Loaded from disk: %d vectors.", index.ntotal)
        return index, id_map
    except Exception as e:
        logger.warning("[FAISS] Disk load failed: %s", e)
        return None


# ── Build from DB ─────────────────────────────────────────────────────────────
def _build_index_from_db():
    """Query all active jobs and embed them. Must be called inside app context."""
    from app.models.job import Job

    jobs = Job.query.filter_by(is_active=True).all()
    if not jobs:
        logger.warning("[FAISS] No active jobs — building empty index.")
        index = faiss.IndexFlatL2(VECTOR_DIM)
        return index, []

    logger.info("[FAISS] Fetching %d active jobs from DB...", len(jobs))
    texts  = [f"{j.title} {j.company} {' '.join(j.skills or [])}" for j in jobs]
    id_map = [str(j.id) for j in jobs]

    logger.info(
        "[FAISS] Embedding %d jobs one-by-one (est. %.0f min) …",
        len(texts),
        len(texts) * INTER_EMBED_SLEEP / 60,
    )

    vectors, success_indices = _embed_texts(texts)

    if len(vectors) == 0:
        logger.error("[FAISS] All embeddings failed — index not built")
        index = faiss.IndexFlatL2(VECTOR_DIM)
        return index, []

    # Align id_map to only the jobs that were successfully embedded
    aligned_ids = [id_map[i] for i in success_indices]

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vectors)
    logger.info("[FAISS] Index built: %d vectors.", index.ntotal)

    _save_to_disk(index, aligned_ids)
    return index, aligned_ids


# ── Public API ────────────────────────────────────────────────────────────────
def load_or_build_index(app=None):
    """Called at app startup — non-blocking background thread."""
    def _init():
        global _index, _id_map
        with _lock:
            result = _load_from_disk()
            if result and result[0].ntotal > 0:
                _index, _id_map = result
            else:
                logger.info("[FAISS] No disk index — building from DB.")
                try:
                    if app:
                        with app.app_context():
                            _index, _id_map = _build_index_from_db()
                    else:
                        _index, _id_map = _build_index_from_db()
                except Exception as e:
                    logger.error("[FAISS] Startup build failed: %s", e)

    threading.Thread(target=_init, daemon=True, name="faiss-startup").start()


def rebuild_index(app=None):
    """
    Called by scheduler after scrape pipeline completes.
    Runs in its own background thread — never blocks the caller.
    Returns immediately with a status dict.
    """
    global _building

    if _building:
        logger.info("[FAISS] Rebuild already in progress — skipping.")
        return {"status": "already_running"}

    def _do_rebuild():
        global _index, _id_map, _building
        _building = True
        try:
            if app:
                with app.app_context():
                    new_index, new_map = _build_index_from_db()
            else:
                new_index, new_map = _build_index_from_db()

            with _lock:
                _index  = new_index
                _id_map = new_map

            logger.info("[FAISS] Rebuild complete: %d vectors.", new_index.ntotal)
        except Exception as e:
            logger.error("[FAISS] Rebuild failed: %s", e)
        finally:
            _building = False

    threading.Thread(target=_do_rebuild, daemon=True, name="faiss-rebuild").start()
    return {"status": "rebuild_started"}


def search_jobs(query: str, k: int = 20):
    """Semantic search — returns up to k job UUIDs."""
    if _index is None or _index.ntotal == 0:
        logger.warning("[FAISS] Index not ready — returning empty results.")
        return []
    try:
        q_vec = _embed_query(query)
        distances, indices = _index.search(q_vec, min(k, _index.ntotal))
        return [_id_map[i] for i in indices[0] if 0 <= i < len(_id_map)]
    except Exception as e:
        logger.error("[FAISS] Search failed: %s", e)
        return []


def match_jobs_by_skills(skills, k: int = 10):
    """Skill-match search — returns job UUIDs with match percentages."""
    if _index is None or _index.ntotal == 0:
        return []
    try:
        query  = " ".join(skills)
        q_vec  = _embed_query(query)
        k_real = min(k, _index.ntotal)
        distances, indices = _index.search(q_vec, k_real)

        max_dist = float(distances[0].max()) if distances[0].max() > 0 else 1.0
        results  = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(_id_map):
                pct = round((1 - dist / max_dist) * 100, 1)
                results.append({"job_id": _id_map[idx], "match_percentage": max(pct, 1.0)})
        return results
    except Exception as e:
        logger.error("[FAISS] Skill match failed: %s", e)
        return []


def get_index_status():
    return {
        "index_ready":         _index is not None and _index.ntotal > 0,
        "vectors_total":       _index.ntotal if _index else 0,
        "id_map_size":         len(_id_map),
        "vector_dim":          VECTOR_DIM,
        "embedding_model":     GEMINI_MODEL,
        "index_on_disk":       os.path.exists(INDEX_PATH),
        "rebuild_in_progress": _building,
    }