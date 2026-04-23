"""
app/services/rag_pipeline.py
NirVexa — Phase 5.6: FAISS Semantic Search
Uses Google Gemini embedding API (free, zero memory cost on Render free tier)
No sentence-transformers, no torch — works within 512MB RAM limit.

Gemini model: models/gemini-embedding-001
Dimensions: 768
Cost: Free tier (1500 requests/minute)
"""

import logging
import os
import threading
import time
import pickle
import numpy as np

import faiss
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL  = "models/gemini-embedding-001"
VECTOR_DIM    = 768
INDEX_PATH    = "nirvexa_jobs.index"
IDMAP_PATH    = "nirvexa_jobs_idmap.pkl"
BATCH_SIZE    = 50          # embed 50 jobs at a time (Gemini rate-limit friendly)
BATCH_DELAY   = 1.0         # seconds between batches

# ─── Globals ───────────────────────────────────────────────────────────────────
_index:   faiss.Index | None = None
_id_map:  list[str]          = []   # position → job UUID
_lock     = threading.Lock()

# ─── Gemini setup ──────────────────────────────────────────────────────────────
def _configure_gemini():
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_AI_KEY")
    )
    if not api_key:
        raise RuntimeError("No Gemini API key found. Set GOOGLE_API_KEY in .env")
    genai.configure(api_key=api_key)


# ─── Embed helpers ─────────────────────────────────────────────────────────────
def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings via Gemini. Returns float32 ndarray (N, 768)."""
    _configure_gemini()
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            result = genai.embed_content(
                model=GEMINI_MODEL,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
            for emb in result["embedding"]:
                vectors.append(emb)
            logger.info("[FAISS] Embedded batch %d-%d", i, i + len(batch))
            if i + BATCH_SIZE < len(texts):
                time.sleep(BATCH_DELAY)
        except Exception as e:
            logger.error("[FAISS] Embed batch %d failed: %s", i, e)
            # Fill with zeros so index positions stay aligned
            for _ in batch:
                vectors.append([0.0] * VECTOR_DIM)
    return np.array(vectors, dtype="float32")


def _embed_query(query: str) -> np.ndarray:
    """Embed a single search query string."""
    _configure_gemini()
    result = genai.embed_content(
        model=GEMINI_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    return np.array([result["embedding"]], dtype="float32")


# ─── Disk persistence ──────────────────────────────────────────────────────────
def _save_to_disk(index: faiss.Index, id_map: list[str]):
    faiss.write_index(index, INDEX_PATH)
    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(id_map, f)
    logger.info("[FAISS] Saved to disk.")


def _load_from_disk():
    if os.path.exists(INDEX_PATH) and os.path.exists(IDMAP_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(IDMAP_PATH, "rb") as f:
                id_map = pickle.load(f)
            logger.info("[FAISS] Loaded from disk: %d vectors.", index.ntotal)
            return index, id_map
        except Exception as e:
            logger.warning("[FAISS] Disk load failed: %s", e)
    return None


# ─── Build index from DB ───────────────────────────────────────────────────────
def _build_index_from_db():
    """Query DB and build a fresh FAISS index. Must be called inside app context."""
    from app.models.job import Job

    logger.info("[FAISS] Fetching active jobs from DB...")
    jobs = Job.query.filter_by(is_active=True).all()

    if not jobs:
        logger.warning("[FAISS] No active jobs — building empty index.")
        empty = faiss.IndexFlatL2(VECTOR_DIM)
        return empty, []

    logger.info("[FAISS] Embedding %d jobs via Gemini...", len(jobs))

    texts  = [f"{j.title} {j.company} {' '.join(j.skills or [])}" for j in jobs]
    uuids  = [str(j.id) for j in jobs]
    vecs   = _embed_texts(texts)

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vecs)
    logger.info("[FAISS] Index built: %d vectors.", index.ntotal)

    _save_to_disk(index, uuids)
    return index, uuids


# ─── Public API ────────────────────────────────────────────────────────────────
def load_or_build_index(app=None) -> None:
    """
    Called at app startup — non-blocking background thread.
    Pass the Flask app object so the thread can push its own context.
    """
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


def rebuild_index(app=None) -> dict:
    """
    Rebuild the FAISS index from scratch — called from scheduler after scrape.
    Runs in a BACKGROUND THREAD so it never blocks gunicorn workers.
    Returns immediately with a status message.
    """
    def _rebuild():
        global _index, _id_map
        logger.info("[FAISS] Background rebuild started...")
        try:
            if app:
                with app.app_context():
                    new_index, new_id_map = _build_index_from_db()
            else:
                new_index, new_id_map = _build_index_from_db()

            with _lock:
                _index  = new_index
                _id_map = new_id_map
            logger.info("[FAISS] Background rebuild complete: %d vectors.", new_index.ntotal)
        except Exception as e:
            logger.error("[FAISS] Background rebuild failed: %s", e)

    threading.Thread(target=_rebuild, daemon=True, name="faiss-rebuild").start()
    return {"status": "rebuild started in background"}


def search_jobs(query: str, k: int = 20) -> list[str]:
    """
    Semantic search — returns list of job UUIDs ranked by similarity.
    Falls back to empty list if index not ready yet.
    """
    global _index, _id_map
    if _index is None or _index.ntotal == 0:
        logger.warning("[FAISS] Index not ready — returning empty results.")
        return []
    try:
        vec = _embed_query(query)
        k   = min(k, _index.ntotal)
        _, indices = _index.search(vec, k)
        return [_id_map[i] for i in indices[0] if 0 <= i < len(_id_map)]
    except Exception as e:
        logger.error("[FAISS] Search failed: %s", e)
        return []


def match_jobs_by_skills(skills: list[str], k: int = 10) -> list[dict]:
    """
    Skill-based matching — returns list of {job_id, match_score} dicts.
    """
    if not skills:
        return []
    query = " ".join(skills)
    uuids = search_jobs(query, k=k)
    if not uuids:
        return []
    # Assign rough match scores based on rank
    results = []
    for rank, uid in enumerate(uuids):
        score = max(100 - rank * 8, 20)
        results.append({"job_id": uid, "match_score": score})
    return results


def get_index_status() -> dict:
    """Admin diagnostic — returns current index state."""
    global _index, _id_map
    return {
        "index_ready":   _index is not None and _index.ntotal > 0,
        "vectors_total": _index.ntotal if _index else 0,
        "id_map_size":   len(_id_map),
        "vector_dim":    VECTOR_DIM,
        "embedding_model": GEMINI_MODEL,
        "index_on_disk": os.path.exists(INDEX_PATH),
    }