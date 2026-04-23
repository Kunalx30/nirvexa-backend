"""
app/services/rag_pipeline.py
NirVexa — Phase 5.6: FAISS Semantic Search
Uses Google Gemini embedding API (free, zero memory cost on Render free tier)
No sentence-transformers, no torch — works within 512MB RAM limit.

Gemini model: models/text-embedding-004
Dimensions:   768
Cost:         Free (1500 requests/min on free tier)
"""

import os
import logging
import threading
from datetime import datetime

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
INDEX_PATH   = "nirvexa_jobs.index"
ID_MAP_PATH  = "nirvexa_jobs_ids.npy"
VECTOR_DIM   = 768          # Gemini text-embedding-004 output dimension
GEMINI_MODEL = "models/text-embedding-004"
BATCH_SIZE   = 100          # Gemini free tier: 1500 req/min, batch safely

# ─────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────
_index: faiss.IndexFlatL2 | None = None
_id_map: list[str]               = []
_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
# 1. Gemini Embedder
# ═══════════════════════════════════════════════════════════════════

def _get_gemini():
    import google.generativeai as genai
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_AI_KEY")
    )
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Set GOOGLE_API_KEY in .env and Render environment."
        )
    genai.configure(api_key=api_key)
    return genai


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns float32 array of shape (n, 768)."""
    genai = _get_gemini()
    all_vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        result = genai.embed_content(
            model=GEMINI_MODEL,
            content=batch,
            task_type="retrieval_document",
        )
        all_vectors.extend(result["embedding"])

    return np.array(all_vectors, dtype="float32")


def _embed_query(query: str) -> np.ndarray:
    """Embed a single search query. Returns float32 array of shape (1, 768)."""
    genai = _get_gemini()
    result = genai.embed_content(
        model=GEMINI_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return np.array([result["embedding"]], dtype="float32")


# ═══════════════════════════════════════════════════════════════════
# 2. Text builder — converts Job ORM → searchable string
# ═══════════════════════════════════════════════════════════════════

def _job_to_text(job) -> str:
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
    return f"{title} {title} {skills_str} {company} {location} {job_type}".strip()


# ═══════════════════════════════════════════════════════════════════
# 3. Index builder
# ═══════════════════════════════════════════════════════════════════

def _build_index_from_db() -> tuple:
    from app.models.job import Job

    logger.info("[FAISS] Fetching active jobs from DB...")
    jobs = Job.query.filter_by(is_active=True).all()

    if not jobs:
        logger.warning("[FAISS] No active jobs — building empty index.")
        index = faiss.IndexFlatL2(VECTOR_DIM)
        _save_to_disk(index, [])
        return index, []

    logger.info("[FAISS] Embedding %d jobs via Gemini...", len(jobs))
    texts  = [_job_to_text(j) for j in jobs]
    id_map = [str(j.id) for j in jobs]

    vectors = _embed_texts(texts)

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vectors)

    _save_to_disk(index, id_map)
    logger.info("[FAISS] Index built: %d vectors.", index.ntotal)
    return index, id_map


def _save_to_disk(index, id_map):
    faiss.write_index(index, INDEX_PATH)
    np.save(ID_MAP_PATH, np.array(id_map, dtype=object))
    logger.info("[FAISS] Saved to disk.")


def _load_from_disk():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        return None
    try:
        index  = faiss.read_index(INDEX_PATH)
        id_map = np.load(ID_MAP_PATH, allow_pickle=True).tolist()
        logger.info("[FAISS] Loaded from disk — %d vectors.", index.ntotal)
        return index, id_map
    except Exception as e:
        logger.error("[FAISS] Disk load failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════
# 4. Public API
# ═══════════════════════════════════════════════════════════════════

def load_or_build_index() -> None:
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
                    _index, _id_map = _build_index_from_db()
                except Exception as e:
                    logger.error("[FAISS] Startup build failed: %s", e)

    threading.Thread(target=_init, daemon=True, name="faiss-startup").start()


def rebuild_index() -> dict:
    """Called by scheduler after nightly scrape."""
    global _index, _id_map
    started_at = datetime.utcnow()

    with _lock:
        try:
            _index, _id_map = _build_index_from_db()
            elapsed = (datetime.utcnow() - started_at).total_seconds()
            return {
                "status":        "success",
                "vectors_total": _index.ntotal,
                "elapsed_sec":   round(elapsed, 2),
            }
        except Exception as e:
            logger.error("[FAISS] Rebuild failed: %s", e)
            return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# 5. Search — called by GET /api/jobs?q=
# ═══════════════════════════════════════════════════════════════════

def search_jobs(query: str, k: int = 20) -> list:
    """
    Returns list of job UUID strings matching the query.
    Returns [] if index not ready yet.
    """
    global _index, _id_map

    if _index is None or _index.ntotal == 0:
        logger.warning("[FAISS] Index not ready — returning []")
        return []

    if not query or not query.strip():
        return []

    try:
        query_vec = _embed_query(query.strip())
        k_actual  = min(k, _index.ntotal)
        distances, indices = _index.search(query_vec, k_actual)

        matched_ids = [
            _id_map[i]
            for i in indices[0]
            if i != -1 and i < len(_id_map)
        ]
        logger.debug("[FAISS] query='%s' → %d results", query, len(matched_ids))
        return matched_ids

    except Exception as e:
        logger.error("[FAISS] search_jobs error: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════
# 6. Skill match — called by POST /api/jobs/match
# ═══════════════════════════════════════════════════════════════════

def match_jobs_by_skills(skills: list, k: int = 10) -> list:
    """
    Returns list of {"job_id": str, "match_percentage": float}
    """
    global _index, _id_map

    if _index is None or _index.ntotal == 0 or not skills:
        return []

    try:
        from app.models.job import Job

        query_vec = _embed_query(" ".join(skills))
        k_actual  = min(k, _index.ntotal)
        distances, indices = _index.search(query_vec, k_actual)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(_id_map):
                continue

            job_id     = _id_map[idx]
            similarity = max(0.0, 1.0 - (float(dist) / 2.0))
            match_pct  = round(similarity * 100, 1)

            # Boost for explicit skill matches
            job = Job.query.get(job_id)
            if job and job.skills:
                job_skills_lower  = [s.lower() for s in job.skills]
                user_skills_lower = [s.lower() for s in skills]
                explicit_matches  = sum(1 for s in user_skills_lower if s in job_skills_lower)
                boost             = min(explicit_matches * 3, 15)
                match_pct         = min(match_pct + boost, 100.0)

            results.append({"job_id": job_id, "match_percentage": match_pct})

        results.sort(key=lambda x: x["match_percentage"], reverse=True)
        return results

    except Exception as e:
        logger.error("[FAISS] match_jobs_by_skills error: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════
# 7. Status — called by GET /api/jobs/admin/faiss-status
# ═══════════════════════════════════════════════════════════════════

def get_index_status() -> dict:
    return {
        "index_ready":   _index is not None,
        "vectors_total": _index.ntotal if _index else 0,
        "id_map_size":   len(_id_map),
        "index_on_disk": os.path.exists(INDEX_PATH),
        "embedding_model": GEMINI_MODEL,
        "vector_dim":    VECTOR_DIM,
    }