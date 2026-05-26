"""NDJSON debug logs for agent session d6f639."""
import json
import time
from pathlib import Path

_LOG_PATHS = [
    Path(r"c:\Users\kunal\nirvexa-frontend\debug-d6f639.log"),
    Path(__file__).resolve().parents[2] / "debug-d6f639.log",
]


def agent_log(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    payload = {
        "sessionId": "d6f639",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload) + "\n"
    for path in _LOG_PATHS:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return
        except OSError:
            continue
