"""
app/services/roadmap_graph_service.py
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

_cache: list | None = None


def _get_data_path() -> str:
    """Resolve path robustly — works both locally and on server."""
    # Try relative to this file first
    here = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(here, "roadmap_graph_data.json")
    if os.path.exists(p1):
        return p1

    # Try relative to cwd (Flask root)
    p2 = os.path.join(os.getcwd(), "app", "services", "roadmap_graph_data.json")
    if os.path.exists(p2):
        return p2

    # Try one level up from here
    p3 = os.path.join(here, "..", "services", "roadmap_graph_data.json")
    if os.path.exists(p3):
        return os.path.normpath(p3)

    return p1  # fallback — will show clear error


def _load() -> list:
    global _cache
    if _cache is not None:
        return _cache
    path = _get_data_path()
    if not os.path.exists(path):
        logger.error("[RoadmapGraph] Data file not found at: %s", path)
        logger.error("[RoadmapGraph] Run scrape_roadmaps.py first.")
        return []
    with open(path, encoding="utf-8") as f:
        _cache = json.load(f)
    logger.info("[RoadmapGraph] Loaded %d roadmaps from %s", len(_cache), path)
    return _cache


def list_roadmaps() -> list[dict]:
    return [{"id": r["id"], "title": r["title"]} for r in _load()]


def get_roadmap(roadmap_id: str) -> dict | None:
    for r in _load():
        if r["id"] == roadmap_id:
            return r
    return None


def search_roadmaps(query: str) -> list[dict]:
    q = query.lower().strip()
    results = []
    for r in _load():
        if q in r["title"].lower() or q in r["id"].lower():
            results.append({"id": r["id"], "title": r["title"]})
    return results


def get_flat_topics(roadmap_id: str) -> list[dict]:
    roadmap = get_roadmap(roadmap_id)
    if not roadmap:
        return []
    flat = []

    def flatten(nodes, parent=None, level=0):
        for node in nodes:
            flat.append({
                "level":     level,
                "label":     node["label"],
                "parent":    parent,
                "resources": node.get("resources", []),
            })
            if node.get("children"):
                flatten(node["children"], node["label"], level + 1)

    flatten(roadmap.get("topics", []))
    return flat