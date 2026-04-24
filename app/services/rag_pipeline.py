"""
app/services/rag_pipeline.py
NirVexa — Phase 5.6: FAISS Semantic Search
Uses HuggingFace Inference API for embeddings — no daily quota limit.

Model: sentence-transformers/all-MiniLM-L6-v2
Dimensions: 384
Cost: Free (HuggingFace free tier)
No sentence-transformers library needed — pure HTTP call.
"""

import logging
import os
import threading
import time
import pickle
import numpy as np
import requests

import faiss

logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
HF_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
VECTOR_DIM     = 384
INDEX_PATH     = "nirvexa_jobs.index"
IDMAP_PATH     = "nirvexa_jobs_idmap.pkl"
BATCH_SIZE     = 64     # HF handles up to 64 texts per call fine
BATCH_DELAY    = 0.5    # small delay between batches
MAX_RETRIES    = 4

# ─── Globals ───────────────────────────────────────────────────────────────────
_index:  faiss.Index | None = None
_id_map: list[str]          = []
_lock    = threading.Lock()


# ─── HuggingFace embed ─────────────────────────────────────────────────────────
def _get_hf_headers():
    api_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        raise RuntimeError("No HuggingFace API key found. Set HF_API_KEY in .env")
    return {"Authorization": f"Bearer {api_key}"}


def _embed_batch_with_retry(batch: list[str], batch_start: int) -> list:
    """Embed one batch via HuggingFace API with retry on 503 (model loading)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                HF_API_URL,
                headers=_get_hf_headers(),
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=60,
            )
            if response.status_code == 200:
                embeddings = response.json()
                # HF returns (N, 384) list — mean pool if nested
                result = []
                for emb in embeddings:
                    if isinstance(emb[0], list):
                        # token-level embeddings — mean pool
                        arr = np.array(emb, dtype="float32")
                        result.append(arr.mean(axis=0).tolist())
                    else:
                        result.append(emb)
                logger.info("[FAISS] Embedded batch %d-%d", batch_start, batch_start + len(batch))
                return result

            elif response.status_code == 503:
                # Model still loading — wait and retry
                wait = 20 * attempt
                logger.warning("[FAISS] HF model loading (attempt %d/%d) — sleeping %ds", attempt, MAX_RETRIES, wait)
                time.sleep(wait)

            elif response.status_code == 429:
                logger.warning("[FAISS] HF rate limit (attempt %d/%d) — sleeping 30s", attempt, MAX_RETRIES)
                time.sleep(30)

            else:
                logger.error("[FAISS] HF API error %d: %s", response.status_code, response.text[:200])
                break

        except Exception as e:
            logger.error("[FAISS] Batch %d request failed: %s", batch_start, e)
            if attempt < MAX_RETRIES:
                time.sleep(10)

    logger.error("[FAISS] Batch %d: all retries failed, using zero vectors", batch_start)
    return [[0.0] * VECTOR_DIM for _ in batch]


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns float32 ndarray (N, 384)."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        embeddings = _embed_batch_with_retry(batch, i)
        vectors.extend(embeddings)
        if i + BATCH_SIZE < len(texts):
            time.sleep(BATCH_DELAY)
    return np.array(vectors, dtype="float32")


def _embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    result = _embed_batch_with_retry([query], 0)
    return np.array([result[0]], dtype="float32")


# ─── Disk persistence ──────────────────────────────────────────────────────────
def _save_to_disk(index: faiss.Index, id_map: list[str]):
    faiss.write_index(index, INDEX_PATH)
    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(id_map, f)
    logger.info("[FAISS] Saved to disk (%d vectors).", index.ntotal)


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


# ─── Incremental index update ──────────────────────────────────────────────────
def _update_index_incremental(existing_index, existing_id_map: list[str]):
    """Only embeds jobs NOT already in existing_id_map. Appends to index."""
    from app.models.job import Job

    all_jobs     = Job.query.filter_by(is_active=True).all()
    existing_set = set(existing_id_map)
    new_jobs     = [j for j in all_jobs if str(j.id) not in existing_set]

    if not new_jobs:
        logger.info("[FAISS] No new jobs to embed — index up to date (%d vectors).", existing_index.ntotal)
        return existing_index, existing_id_map

    logger.info("[FAISS] Embedding %d new jobs (skipping %d already indexed).", len(new_jobs), len(existing_id_map))

    texts = [f"{j.title} {j.company} {' '.join(j.skills or [])}" for j in new_jobs]
    uuids = [str(j.id) for j in new_jobs]
    vecs  = _embed_texts(texts)

    if vecs.shape != (len(new_jobs), VECTOR_DIM):
        raise ValueError(f"[FAISS] Shape mismatch: got {vecs.shape}, expected ({len(new_jobs)}, {VECTOR_DIM})")

    existing_index.add(vecs)
    updated_id_map = existing_id_map + uuids

    logger.info("[FAISS] Index updated: %d total vectors.", existing_index.ntotal)
    _save_to_disk(existing_index, updated_id_map)
    return existing_index, updated_id_map


def _build_fresh_index():
    """Full build from scratch — used when no disk index exists."""
    from app.models.job import Job

    logger.info("[FAISS] Fresh build — fetching all active jobs...")
    jobs = Job.query.filter_by(is_active=True).all()

    if not jobs:
        logger.warning("[FAISS] No active jobs — empty index.")
        return faiss.IndexFlatL2(VECTOR_DIM), []

    logger.info("[FAISS] Embedding all %d jobs via HuggingFace...", len(jobs))
    texts = [f"{j.title} {j.company} {' '.join(j.skills or [])}" for j in jobs]
    uuids = [str(j.id) for j in jobs]
    vecs  = _embed_texts(texts)

    if vecs.shape != (len(jobs), VECTOR_DIM):
        raise ValueError(f"[FAISS] Shape mismatch: got {vecs.shape}, expected ({len(jobs)}, {VECTOR_DIM})")

    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vecs)
    logger.info("[FAISS] Fresh index built: %d vectors.", index.ntotal)

    _save_to_disk(index, uuids)
    return index, uuids


# ─── Public API ────────────────────────────────────────────────────────────────
def load_or_build_index(app=None) -> None:
    """Startup — load from disk only. No build to avoid deploy race."""
    def _init():
        global _index, _id_map
        with _lock:
            result = _load_from_disk()
            if result and result[0].ntotal > 0:
                _index, _id_map = result
                logger.info("[FAISS] Loaded at startup: %d vectors.", _index.ntotal)
            else:
                logger.warning("[FAISS] No disk index. Trigger pipeline to build.")

    threading.Thread(target=_init, daemon=True, name="faiss-startup").start()


def rebuild_index(app=None) -> dict:
    """
    Incremental update — only embeds new jobs.
    Called from scheduler after daily scrape.
    Returns immediately, runs in background thread.
    """
    def _rebuild():
        global _index, _id_map
        logger.info("[FAISS] Incremental update started...")

        def _do_update():
            result = _load_from_disk()
            if result and result[0].ntotal > 0:
                new_index, new_id_map = _update_index_incremental(result[0], result[1])
            else:
                new_index, new_id_map = _build_fresh_index()

            with _lock:
                _index  = new_index
                _id_map = new_id_map
            logger.info("[FAISS] Update complete: %d vectors.", new_index.ntotal)

        try:
            if app:
                with app.app_context():
                    _do_update()
            else:
                _do_update()
        except Exception as e:
            logger.error("[FAISS] Update failed: %s", e)

    threading.Thread(target=_rebuild, daemon=True, name="faiss-rebuild").start()
    return {"status": "incremental update started in background"}


def search_jobs(query: str, k: int = 20) -> list[str]:
    """Semantic search — returns list of job UUIDs ranked by similarity."""
    global _index, _id_map
    if _index is None or _index.ntotal == 0:
        logger.warning("[FAISS] Index not ready.")
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
    """Skill-based matching — returns {job_id, match_percentage} dicts."""
    if not skills:
        return []
    query = " ".join(skills)
    uuids = search_jobs(query, k=k)
    return [
        {"job_id": uid, "match_percentage": max(100 - rank * 8, 20)}
        for rank, uid in enumerate(uuids)
    ]


def get_index_status() -> dict:
    global _index, _id_map
    return {
        "index_ready":     _index is not None and _index.ntotal > 0,
        "vectors_total":   _index.ntotal if _index else 0,
        "id_map_size":     len(_id_map),
        "vector_dim":      VECTOR_DIM,
        "embedding_model": HF_MODEL,
        "index_on_disk":   os.path.exists(INDEX_PATH),
    }


def load_index_from_disk() -> dict:
    """Load existing disk index into memory without rebuilding."""
    global _index, _id_map
    with _lock:
        result = _load_from_disk()
        if result and result[0].ntotal > 0:
            _index  = result[0]
            _id_map = result[1]
            logger.info("[FAISS] Loaded from disk: %d vectors.", _index.ntotal)
            return {"status": "loaded", "vectors": _index.ntotal}
        else:
            return {"status": "no disk index found"}