"""
app/services/rag_pipeline.py
NirVexa — Phase 5.6: FAISS Semantic Search
Kunal Chandelkar | April 2026

Flow:
  1. At app startup → load_or_build_index() is called
  2. Nightly (after scheduler pipeline) → rebuild_index() is called
  3. On user search with q= param → search_jobs(query, k=20) returns job IDs
  4. Jobs API combines FAISS results with SQL filters for final response
"""

import os
import logging
import threading
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_NAME   = "all-MiniLM-L6-v2"   # ~80MB, 384-dim embeddings, fast + accurate
INDEX_PATH   = "nirvexa_jobs.index"  # saved on disk, lives in project root
ID_MAP_PATH  = "nirvexa_jobs_ids.npy"  # maps FAISS integer positions → job UUIDs
VECTOR_DIM   = 384                   # must match the model's output dimension

# ─────────────────────────────────────────────
# Module-level singletons (loaded once at startup)
# ─────────────────────────────────────────────
_model: SentenceTransformer | None = None
_index: faiss.IndexFlatL2 | None   = None
_id_map: list[str]                 = []   # list of job UUID strings in FAISS order
_lock = threading.Lock()           # prevent concurrent rebuilds


# ═══════════════════════════════════════════════════════════════════
# 1. Model loader — called once, cached for the app lifetime
# ═══════════════════════════════════════════════════════════════════

def _get_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and cache it globally."""
    global _model
    if _model is None:
        logger.info("[FAISS] Loading SentenceTransformer model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("[FAISS] Model loaded successfully.")
    return _model


# ═══════════════════════════════════════════════════════════════════
# 2. Text builder — converts a job row into a searchable string
# ═══════════════════════════════════════════════════════════════════

def _job_to_text(job) -> str:
    """
    Combine the most semantically meaningful fields into a single string
    for embedding. Weights: title > skills > company > location.

    Accepts either a Job ORM object or a plain dict.
    """
    if isinstance(job, dict):
        title    = job.get("title", "") or ""
        company  = job.get("company", "") or ""
        location = job.get("location", "") or ""
        skills   = job.get("skills", []) or []
        job_type = job.get("job_type", "") or ""
    else:
        title    = getattr(job, "title", "") or ""
        company  = getattr(job, "company", "") or ""
        location = getattr(job, "location", "") or ""
        skills   = getattr(job, "skills", []) or []
        job_type = getattr(job, "job_type", "") or ""

    skills_str = " ".join(skills) if isinstance(skills, list) else str(skills)

    # Title is repeated to give it more weight in the embedding
    return f"{title} {title} {skills_str} {company} {location} {job_type}".strip()


# ═══════════════════════════════════════════════════════════════════
# 3. Index builder — builds FAISS index from all active jobs in DB
# ═══════════════════════════════════════════════════════════════════

def _build_index_from_db() -> tuple[faiss.IndexFlatL2, list[str]]:
    """
    Fetch all active jobs from PostgreSQL, embed them, build a fresh
    FAISS IndexFlatL2, save to disk, and return (index, id_map).
    """
    # Import here to avoid circular imports at module load time
    from app.models.job import Job

    logger.info("[FAISS] Fetching active jobs from database...")
    jobs = Job.query.filter_by(is_active=True).all()

    if not jobs:
        logger.warning("[FAISS] No active jobs found — building empty index.")
        index  = faiss.IndexFlatL2(VECTOR_DIM)
        id_map = []
        _save_to_disk(index, id_map)
        return index, id_map

    logger.info("[FAISS] Building embeddings for %d jobs...", len(jobs))
    model = _get_model()

    texts  = [_job_to_text(j) for j in jobs]
    id_map = [str(j.id) for j in jobs]   # UUID strings in same order as texts

    # Encode in batches of 256 to avoid OOM on Render free tier
    vectors = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    ).astype("float32")

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vectors)

    _save_to_disk(index, id_map)
    logger.info("[FAISS] Index built with %d vectors and saved to disk.", index.ntotal)
    return index, id_map


def _save_to_disk(index: faiss.IndexFlatL2, id_map: list[str]) -> None:
    """Persist FAISS index + UUID map to disk."""
    faiss.write_index(index, INDEX_PATH)
    np.save(ID_MAP_PATH, np.array(id_map, dtype=object))
    logger.info("[FAISS] Saved index → %s | id_map → %s", INDEX_PATH, ID_MAP_PATH)


def _load_from_disk() -> tuple[faiss.IndexFlatL2, list[str]] | None:
    """Load FAISS index + UUID map from disk. Returns None if files don't exist."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        return None
    try:
        index  = faiss.read_index(INDEX_PATH)
        id_map = np.load(ID_MAP_PATH, allow_pickle=True).tolist()
        logger.info("[FAISS] Loaded index from disk — %d vectors.", index.ntotal)
        return index, id_map
    except Exception as e:
        logger.error("[FAISS] Failed to load from disk: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════
# 4. Public API — called by app/__init__.py and scheduler.py
# ═══════════════════════════════════════════════════════════════════

def load_or_build_index() -> None:
    """
    Called once at app startup inside create_app().
    Tries to load saved index from disk; if missing or empty, builds fresh.
    Non-blocking: runs in a background thread so app startup isn't delayed.
    """
    def _init():
        global _index, _id_map
        with _lock:
            result = _load_from_disk()
            if result and result[0].ntotal > 0:
                _index, _id_map = result
                logger.info("[FAISS] Startup: loaded existing index (%d vectors).", _index.ntotal)
            else:
                logger.info("[FAISS] Startup: no valid index on disk — building from DB.")
                try:
                    _index, _id_map = _build_index_from_db()
                except Exception as e:
                    logger.error("[FAISS] Startup build failed: %s", e)

    thread = threading.Thread(target=_init, daemon=True, name="faiss-startup")
    thread.start()


def rebuild_index() -> dict:
    """
    Called by APScheduler after the nightly scraper pipeline completes.
    Rebuilds the FAISS index with all newly inserted jobs.
    Returns a stats dict for logging.
    """
    global _index, _id_map
    started_at = datetime.utcnow()

    with _lock:
        logger.info("[FAISS] Nightly rebuild started at %s UTC", started_at.isoformat())
        try:
            new_index, new_id_map = _build_index_from_db()
            _index  = new_index
            _id_map = new_id_map
            elapsed = (datetime.utcnow() - started_at).total_seconds()
            stats = {
                "status":        "success",
                "vectors_total": new_index.ntotal,
                "elapsed_sec":   round(elapsed, 2),
                "rebuilt_at":    datetime.utcnow().isoformat(),
            }
            logger.info("[FAISS] Rebuild complete: %s", stats)
            return stats
        except Exception as e:
            logger.error("[FAISS] Rebuild failed: %s", e)
            return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# 5. Search function — called by Jobs API when q= is present
# ═══════════════════════════════════════════════════════════════════

def search_jobs(query: str, k: int = 20) -> list[str]:
    """
    Embed the user's query and find the k nearest jobs in the FAISS index.

    Returns a list of job UUID strings (may be fewer than k if index is small).
    Returns [] if the index isn't ready yet (building in background).

    Usage in jobs.py:
        from app.services.rag_pipeline import search_jobs
        job_ids = search_jobs(query=q, k=20)
        # Then filter those IDs with SQL for location/type/salary
    """
    global _index, _id_map

    if _index is None or _index.ntotal == 0:
        logger.warning("[FAISS] search_jobs called but index not ready — returning []")
        return []

    if not query or not query.strip():
        return []

    try:
        model       = _get_model()
        query_vec   = model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Clamp k to the actual index size
        k_actual    = min(k, _index.ntotal)
        distances, indices = _index.search(query_vec, k_actual)

        # indices[0] is the list of integer positions in our id_map
        # Filter out -1 (FAISS returns -1 when fewer results than k exist)
        matched_ids = [
            _id_map[i]
            for i in indices[0]
            if i != -1 and i < len(_id_map)
        ]

        logger.debug(
            "[FAISS] query='%s' → %d results (k=%d)", query, len(matched_ids), k
        )
        return matched_ids

    except Exception as e:
        logger.error("[FAISS] search_jobs error for query '%s': %s", query, e)
        return []


# ═══════════════════════════════════════════════════════════════════
# 6. Skill-match function — used by POST /api/jobs/match endpoint
# ═══════════════════════════════════════════════════════════════════

def match_jobs_by_skills(skills: list[str], k: int = 10) -> list[dict]:
    """
    Given a list of user skills, find the top k matching jobs and compute
    a match_percentage for each.

    Returns list of dicts: [{"job_id": str, "match_percentage": float}]

    Usage in jobs.py:
        from app.services.rag_pipeline import match_jobs_by_skills
        results = match_jobs_by_skills(skills=["Python", "SQL", "ML"], k=10)
    """
    global _index, _id_map

    if _index is None or _index.ntotal == 0:
        return []

    if not skills:
        return []

    try:
        from app.models.job import Job

        model       = _get_model()
        query_text  = " ".join(skills)
        query_vec   = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        k_actual = min(k, _index.ntotal)
        distances, indices = _index.search(query_vec, k_actual)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(_id_map):
                continue

            job_id = _id_map[idx]

            # Convert L2 distance → similarity percentage
            # With normalized vectors: similarity = 1 - (dist / 2)
            # dist=0 means identical → 100%, dist=2 means opposite → 0%
            similarity      = max(0.0, 1.0 - (float(dist) / 2.0))
            match_pct       = round(similarity * 100, 1)

            # Boost score if job skills explicitly contain user's skills
            job = Job.query.get(job_id)
            if job and job.skills:
                job_skills_lower  = [s.lower() for s in job.skills]
                user_skills_lower = [s.lower() for s in skills]
                explicit_matches  = sum(1 for s in user_skills_lower if s in job_skills_lower)
                boost             = min(explicit_matches * 3, 15)  # max +15% boost
                match_pct         = min(match_pct + boost, 100.0)

            results.append({
                "job_id":          job_id,
                "match_percentage": match_pct,
            })

        # Sort descending by match score
        results.sort(key=lambda x: x["match_percentage"], reverse=True)
        return results

    except Exception as e:
        logger.error("[FAISS] match_jobs_by_skills error: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════
# 7. Status helper — used by /api/jobs/admin/faiss-status endpoint
# ═══════════════════════════════════════════════════════════════════

def get_index_status() -> dict:
    """Returns a diagnostic snapshot of the current FAISS index state."""
    return {
        "index_ready":   _index is not None,
        "vectors_total": _index.ntotal if _index else 0,
        "id_map_size":   len(_id_map),
        "index_path":    INDEX_PATH,
        "index_on_disk": os.path.exists(INDEX_PATH),
        "model_loaded":  _model is not None,
        "model_name":    MODEL_NAME,
    }