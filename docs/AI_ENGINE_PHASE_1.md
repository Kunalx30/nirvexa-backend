# Nirvexa AI Engine — Phase 1: Web Research Engine

## 1. Overview & Architecture

Phase 1 establishes the foundational **Web Research Engine** for Nirvexa. It is built as an independent, modular subsystem inside `app/ai_engine/` to enable retrieval-augmented research while strictly preserving existing production AI capabilities.

### Architecture Diagram

```
[ POST /api/ai/research/search ]
              │
              ▼
   [ WebResearchEngine ] ── (Orchestrator)
     │               │
     ▼               ▼
[ SearchService ]  [ WebFetcher ] ── (SSRF Validator & Multi-Hop Redirect Guard)
     │               │
     ▼               ▼
[ SearchProvider ]  [ ContentExtractor ]
 (DuckDuckGo / Mock) (html.parser Clean Text & Metadata)
```

---

## 2. Directory Structure & Files Created

All Phase 1 code is strictly isolated inside `app/ai_engine/`, its Blueprint route `app/routes/ai_engine.py`, test suite `tests/test_ai_engine_phase1.py`, and this documentation:

```
nirvexa-backend/
├── app/
│   ├── ai_engine/
│   │   ├── __init__.py
│   │   ├── coordinator.py             # WebResearchEngine orchestrator
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── research.py            # Pydantic schemas (SearchRequest, ResearchResponse, etc.)
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Abstract SearchProvider interface
│   │   │   ├── models.py              # RawSearchResult dataclass
│   │   │   ├── provider.py            # DuckDuckGoSearchProvider & MockSearchProvider
│   │   │   └── service.py             # Provider resolver factory
│   │   ├── fetch/
│   │   │   ├── __init__.py
│   │   │   ├── url_validator.py       # SSRF protection, normalizer & deduplicator
│   │   │   └── web_fetcher.py         # Targeted streaming fetcher (<=2MB, redirect SSRF checks)
│   │   └── extraction/
│   │       ├── __init__.py
│   │       └── content_extractor.py   # Standard html.parser extractor (title, meta, full body)
│   └── routes/
│       └── ai_engine.py               # Flask Blueprint (POST /api/ai/research/search)
├── tests/
│   └── test_ai_engine_phase1.py       # 14 zero-network automated unit tests
└── docs/
    └── AI_ENGINE_PHASE_1.md           # This specification
```

---

## 3. Configuration & Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `AI_ENGINE_ENABLED` | bool | `False` | Feature flag to enable/disable the AI Engine subsystem. |
| `AI_ENGINE_SEARCH_PROVIDER` | str | `duckduckgo` | Active search provider (`duckduckgo` or `mock`). |
| `AI_ENGINE_FETCH_TIMEOUT_SECONDS` | int | `10` | Timeout per HTTP fetch request in seconds. |
| `AI_ENGINE_MAX_FETCH_WORKERS` | int | `5` | Thread pool concurrency limit for parallel fetching. |

---

## 4. Security & SSRF Protection Specifications

The web research fetcher is a **strictly targeted document fetcher** (NOT a crawler). It enforces rigorous SSRF (Server-Side Request Forgery) protection:

1. **Scheme Validation:** Only `http` and `https` protocols are permitted. `file:`, `ftp:`, `gopher:`, `javascript:`, etc. are rejected immediately.
2. **Cloud Metadata IP Blocking:** Proactively blocks AWS/GCP/Azure link-local metadata address `169.254.169.254` and its subnets.
3. **Private & Loopback Address Blocking:** Resolves DNS hostname to IP address and validates with Python's `ipaddress` standard library module:
   - `127.0.0.0/8` and `::1` (Loopback)
   - `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` (Private IPv4)
   - `fc00::/7` (Unique Local IPv6)
   - `169.254.0.0/16`, `fe80::/10` (Link-Local)
   - `224.0.0.0/4`, `ff00::/8` (Multicast)
4. **Internal Domain Suffixes:** Prohibits domains ending in `.local`, `.internal`, `.localhost`, `.lan`.
5. **Multi-Hop Redirect Validation:** Requests are issued with `allow_redirects=False`. Every 3xx redirect location header is captured, re-validated through the SSRF filter before following, and capped at a maximum of 5 hops.
6. **Streaming Size Limit:** Responses are streamed in 8KB chunks and aborted if `Content-Length` or total downloaded bytes exceed 2 MB (2,097,152 bytes) to defend against compression bombs or infinite streams.

---

## 5. API Endpoint Reference

### `POST /api/ai/research/search`

Researches a query by fetching web search results, retrieving page contents, and extracting clean structured data.

#### Request Body
```json
{
  "query": "top frontend developer skills 2026",
  "max_results": 3,
  "fetch_content": true
}
```

#### Response Body (`200 OK`)
```json
{
  "query": "top frontend developer skills 2026",
  "total_results": 3,
  "results": [
    {
      "url": "https://example.com/skills",
      "title": "Top Frontend Skills in 2026",
      "snippet": "Essential skills include React, TypeScript, performance optimization...",
      "clean_text": "Top Frontend Skills in 2026. TypeScript and modern frontend frameworks dominate...",
      "metadata": {
        "title": "Top Frontend Skills in 2026",
        "description": "Comprehensive guide to frontend developer skills",
        "author": "Tech Insights",
        "published_date": "2026-01-15",
        "word_count": 450
      },
      "fetched_successfully": true,
      "error": null
    }
  ],
  "execution_time_ms": 420.5
}
```

---

## 6. Zero New Dependencies

No external third-party libraries were added to `requirements.txt`. Phase 1 fully utilizes:
- `requests` & `urllib3` (already installed in Nirvexa backend)
- `pydantic` (already installed in Nirvexa backend)
- Python Standard Library (`html.parser`, `ipaddress`, `socket`, `urllib.parse`, `concurrent.futures`, `dataclasses`)

---

## 7. Automated Test Verification

A dedicated unit test suite with zero external network access is located in [tests/test_ai_engine_phase1.py](file:///c:/Users/kunal/nirvexa-backend/tests/test_ai_engine_phase1.py).

Run the tests using:
```bash
.\venv_new\Scripts\python.exe -m pytest tests/test_ai_engine_phase1.py -v
```

### Test Coverage (14/14 Passed)
- `test_normalize_url`: Strips query parameters, tracking tokens (`utm_*`), and URL fragments.
- `test_search_request_validation`: Rejects empty strings, whitespace, and invalid result counts.
- `test_mock_search_provider`: Deterministic in-memory search provider generation.
- `test_ssrf_validator_blocks_private_and_loopback`: Blocks `localhost`, `127.0.0.1`, `10.0.0.1`, `192.168.1.1`, and cloud metadata `169.254.169.254`.
- `test_ssrf_validator_blocks_invalid_schemes`: Rejects `ftp://`, `file:///etc/passwd`, `gopher://`.
- `test_ssrf_validator_allows_public_ip`: Permits standard public URLs (`https://example.com`).
- `test_content_extractor_extracts_clean_text`: Strips `<script>`, `<style>`, extracts `<title>`, meta description, author, published date, and computes word count.
- `test_web_fetcher_blocks_direct_ssrf`: Prevents fetcher from dispatching requests to blocked endpoints.
- `test_web_fetcher_blocks_redirect_ssrf`: Catches attempts to redirect a public URL to loopback or private networks.
- `test_web_fetcher_enforces_size_cap`: Truncates/aborts responses exceeding 2MB.
- `test_web_fetcher_handles_timeout_and_404`: Gracefully captures HTTP error statuses without crashing.
- `test_web_research_engine_end_to_end_mock`: Tests end-to-end orchestration with mock provider and mock HTTP responses.
- `test_api_endpoint_feature_flag_disabled`: Returns HTTP 503 when `AI_ENGINE_ENABLED=False`.
- `test_api_endpoint_success`: Validates HTTP 200 payload against Pydantic response schema when enabled.
