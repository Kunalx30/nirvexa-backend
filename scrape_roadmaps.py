"""
scrape_roadmaps.py
==================
Reads the locally cloned roadmap.sh repo and produces a clean JSON file.

Usage:
    cd nirvexa-backend
    python scrape_roadmaps.py

Output:
    app/services/roadmap_graph_data.json
"""

import json
import os
import re
import glob

# ── Config ────────────────────────────────────────────────────────────────────
REPO_DIR    = os.path.join(os.path.dirname(__file__), "roadmap_repo", "src", "data", "roadmaps")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "app", "services", "roadmap_graph_data.json")

# Node types to keep (skip decorative section boxes)
KEEP_TYPES = {"topic", "subtopic", "todo", "checkbox"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_md_resources(md_text: str) -> list[dict]:
    """Extract links from a content markdown file."""
    resources = []
    if not md_text:
        return resources

    # Format 1: <BadgeLink href="URL" ...>Title</BadgeLink>
    for m in re.finditer(r"<BadgeLink[^>]*href=['\"]([^'\"]+)['\"][^>]*>([^<]+)</BadgeLink>", md_text, re.I):
        url, title = m.group(1).strip(), m.group(2).strip()
        if url.startswith("http") and title:
            rtype = "paid" if any(x in url for x in ["coursera", "udemy", "pluralsight", "linkedin"]) else "free"
            resources.append({"title": title[:60], "url": url, "type": rtype})

    # Format 2: plain markdown links [Title](URL)
    if not resources:
        for title, url in re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', md_text):
            resources.append({"title": title.strip()[:60], "url": url.strip(), "type": "free"})

    return resources[:5]


def build_content_map(content_dir: str) -> dict[str, list]:
    """
    Map node_id → list of resources by reading content markdown files.
    Filename format: topicname@NODE_ID.md
    """
    result = {}
    if not os.path.isdir(content_dir):
        return result

    for fpath in glob.glob(os.path.join(content_dir, "*.md")):
        fname = os.path.basename(fpath)
        # Extract node id after @
        match = re.search(r'@([^.]+)\.md$', fname)
        if not match:
            continue
        node_id = match.group(1)
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            resources = parse_md_resources(text)
            if resources:
                result[node_id] = resources
        except Exception:
            pass

    return result


def parse_roadmap_json(json_path: str, content_dir: str, title: str, roadmap_id: str) -> dict | None:
    """Parse a roadmap JSON + content dir into a clean tree structure."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  JSON parse error: {e}")
        return None

    nodes_raw = data.get("nodes", [])
    edges_raw = data.get("edges", [])

    # Build node map — keep only topic/subtopic types with labels
    node_map = {}
    for n in nodes_raw:
        ntype = n.get("type", "")
        if ntype not in KEEP_TYPES:
            continue
        label = (n.get("data") or {}).get("label", "").strip()
        if not label:
            continue
        nid = n.get("id", "")
        pos = n.get("position", {})
        node_map[nid] = {
            "id":        nid,
            "label":     label,
            "type":      ntype,
            "y":         pos.get("y", 9999),
            "x":         pos.get("x", 0),
            "children":  [],
            "resources": [],
        }

    if not node_map:
        return None

    # Build adjacency from edges
    children_of = {nid: [] for nid in node_map}
    has_parent  = set()
    for edge in edges_raw:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src in node_map and tgt in node_map:
            children_of[src].append(tgt)
            has_parent.add(tgt)

    # Root nodes = topic nodes with no parent, sorted by y then x
    roots = [
        node_map[nid] for nid in node_map
        if nid not in has_parent and node_map[nid]["type"] == "topic"
    ]
    roots.sort(key=lambda n: (round(n["y"] / 80) * 80, n["x"]))

    # Load content resources
    content_map = build_content_map(content_dir)
    for nid, res in content_map.items():
        if nid in node_map:
            node_map[nid]["resources"] = res

    # Recursively resolve children
    visited = set()

    def resolve(nid: str, depth: int = 0) -> dict | None:
        if nid in visited or depth > 6:
            return None
        visited.add(nid)
        n = node_map[nid]
        kids_raw = sorted(
            [node_map[c] for c in children_of.get(nid, []) if c in node_map],
            key=lambda k: (round(k["y"] / 60) * 60, k["x"])
        )
        kids = [r for r in (resolve(k["id"], depth + 1) for k in kids_raw) if r]
        return {
            "id":        n["id"],
            "label":     n["label"],
            "type":      n["type"],
            "resources": n["resources"],
            "children":  kids,
        }

    topics = [r for r in (resolve(root["id"]) for root in roots) if r]

    if not topics:
        return None

    return {
        "id":     roadmap_id,
        "title":  title,
        "topics": topics,
    }


def get_title_from_md(md_path: str) -> str | None:
    """Read briefTitle or title from the roadmap markdown frontmatter."""
    if not os.path.exists(md_path):
        return None
    try:
        with open(md_path, encoding="utf-8", errors="ignore") as f:
            text = f.read(2000)
        m = re.search(r"briefTitle:\s*['\"]?(.+?)['\"]?\s*$", text, re.M)
        if m:
            return m.group(1).strip().strip("'\"")
        m = re.search(r"title:\s*['\"]?(.+?)['\"]?\s*$", text, re.M)
        if m:
            return m.group(1).strip().strip("'\"")
    except Exception:
        pass
    return None


def main():
    print("roadmap.sh local scraper — starting")
    print(f"Repo:   {REPO_DIR}")
    print(f"Output: {OUTPUT_PATH}")

    if not os.path.isdir(REPO_DIR):
        print(f"\nERROR: Repo not found at {REPO_DIR}")
        print("Run: git clone --depth=1 https://github.com/kamranahmedse/developer-roadmap.git roadmap_repo")
        return

    # Discover all roadmap folders
    roadmap_dirs = sorted([
        d for d in os.listdir(REPO_DIR)
        if os.path.isdir(os.path.join(REPO_DIR, d))
    ])
    print(f"Found {len(roadmap_dirs)} roadmap folders\n")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load existing to allow resuming
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, encoding="utf-8") as f:
                for item in json.load(f):
                    existing[item["id"]] = item
            print(f"Resuming — {len(existing)} already done\n")
        except Exception:
            pass

    results = dict(existing)
    ok, failed = 0, 0

    for roadmap_id in roadmap_dirs:
        if roadmap_id in results:
            print(f"  SKIP  {roadmap_id}")
            continue

        folder      = os.path.join(REPO_DIR, roadmap_id)
        json_path   = os.path.join(folder, f"{roadmap_id}.json")
        md_path     = os.path.join(folder, f"{roadmap_id}.md")
        content_dir = os.path.join(folder, "content")

        if not os.path.exists(json_path):
            print(f"  SKIP  {roadmap_id} — no JSON file")
            continue

        title = get_title_from_md(md_path) or roadmap_id.replace("-", " ").title()
        print(f"  PARSE {roadmap_id} ({title})", end="", flush=True)

        roadmap = parse_roadmap_json(json_path, content_dir, title, roadmap_id)
        if roadmap:
            results[roadmap_id] = roadmap
            topic_count = len(roadmap["topics"])
            print(f" — {topic_count} root topics  OK")
            ok += 1
        else:
            print(f" — FAILED (no topics extracted)")
            failed += 1

        # Save after every roadmap (resumable)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, indent=2, ensure_ascii=False)

    print(f"\nDone!  {ok} OK  |  {failed} failed  |  {len(results)} total")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()