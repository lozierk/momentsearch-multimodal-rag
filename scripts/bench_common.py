"""Shared plumbing for the 3.3 benchmark harness.

Used by bench_accept.py, bench_backfill.py and run_eval.py. Nothing in here
talks to the app's internals — every measurement goes over HTTP through the
same endpoints a browser uses, so the numbers are end-to-end, not in-process.

Auth follows scripts/upload_doc.py exactly: ADMIN_TOKEN is read from the repo's
.env and is never printed, logged, or passed on argv.
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent      # .../momentsearch
BENCH_DIR = REPO.parent / "docs" / "bench"         # ../docs/bench (results)
EVAL_DIR = REPO.parent / "eval"                    # ../eval (golden set)
CORPUS_DIR = REPO.parent / "corpus"                # ../corpus (papers/, decks/)

DEFAULT_BASE = "http://127.0.0.1:8000"
# A row in one of these is done moving. `skipped` = dedup found the same bytes
# already indexed for this user — an outcome, not a failure (src/ingest/doc_pipeline.py).
TERMINAL_STATUSES = ("indexed", "skipped", "failed")

START_HINT = (
    "Start the stack from momentsearch/ (see ../CLAUDE.md):\n"
    "  .venv/bin/uvicorn src.clip_service:app --port 8001\n"
    "  CLIP_SERVICE_URL=http://127.0.0.1:8001 .venv/bin/uvicorn src.app:app --port 8000"
)


def die(msg: str) -> None:
    print(f"\n[fatal] {msg}\n", file=sys.stderr)
    raise SystemExit(2)


def admin_token() -> str | None:
    """ADMIN_TOKEN from the repo .env — same source the app itself reads."""
    load_dotenv(REPO / ".env")
    return os.environ.get("ADMIN_TOKEN") or None


def call(base: str, method: str, path: str, token: str | None,
         body: bytes | None = None, ctype: str = "application/json",
         timeout: float = 180) -> dict:
    """One JSON request. `path` may be absolute (presigned bucket URLs)."""
    url = path if path.startswith("http") else base + path
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", ctype)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def post_json(base: str, path: str, token: str | None, payload: dict,
              timeout: float = 180) -> dict:
    return call(base, "POST", path, token, json.dumps(payload).encode(),
                timeout=timeout)


def preflight(base: str, token: str | None) -> list[dict]:
    """Fail fast, loudly, before any benchmark spends a minute of wall clock."""
    try:
        return call(base, "GET", "/api/videos", token, timeout=10).get("videos", [])
    except urllib.error.HTTPError as e:
        extra = (f"\nCheck ADMIN_TOKEN in {REPO / '.env'} — it must match the "
                 f"token the API was started with." if e.code == 401 else "")
        die(f"API at {base} rejected GET /api/videos: HTTP {e.code} {e.reason}.{extra}")
    except urllib.error.URLError as e:
        die(f"Cannot reach the API at {base} ({e.reason}).\n{START_HINT}")
    except Exception as e:  # malformed JSON, wrong service on the port, ...
        die(f"GET {base}/api/videos did not look like MomentSearch: "
            f"{type(e).__name__}: {e}\n{START_HINT}")
    return []


# ── Upload path (presign -> PUT -> register), the same three calls as upload_doc.py ──

def presign(base: str, token: str | None, filename: str, size: int,
            ctype: str = "application/pdf") -> dict:
    return post_json(base, "/api/videos/presign", token,
                     {"filename": filename, "content_type": ctype, "size": size})


def put_object(base: str, token: str | None, ps: dict, data: bytes,
               ctype: str = "application/pdf") -> None:
    """Upload the bytes wherever presign pointed us: the bucket (presigned, no
    auth header) or the API itself (STORAGE_PROVIDER=local dev fallback)."""
    if ps["mode"] == "direct":
        call(base, "PUT", ps["url"], token, data, ctype)
        return
    req = urllib.request.Request(ps["url"], data=data, method="PUT")
    for k, v in ps.get("headers", {}).items():
        req.add_header(k, v)
    urllib.request.urlopen(req, timeout=180).read()


def register(base: str, token: str | None, video_id: str, key: str,
             kind: str, title: str) -> dict:
    return post_json(base, "/api/videos", token,
                     {"video_id": video_id, "key": key, "kind": kind, "title": title})


def delete_video(base: str, token: str | None, video_id: str) -> bool:
    try:
        call(base, "DELETE", f"/api/videos/{video_id}", token, timeout=60)
        return True
    except Exception as e:
        print(f"  [warn] delete {video_id} failed: {type(e).__name__}: {e}")
        return False


def list_videos(base: str, token: str | None) -> dict[str, dict]:
    rows = call(base, "GET", "/api/videos", token, timeout=30).get("videos", [])
    return {r["id"]: r for r in rows}


# ── Stats ────────────────────────────────────────────────────────────────────

def pctl(samples: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation) — with n=100 the p95 is a
    sample we actually observed, which is what we want to report."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = max(0, math.ceil(p / 100.0 * len(s)) - 1)
    return s[k]


def summarize(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    return {
        "n": len(samples),
        "min_ms": round(min(samples), 1),
        "p50_ms": round(pctl(samples, 50), 1),
        "p95_ms": round(pctl(samples, 95), 1),
        "p99_ms": round(pctl(samples, 99), 1),
        "max_ms": round(max(samples), 1),
        "mean_ms": round(sum(samples) / len(samples), 1),
    }


def fmt_summary(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"  {label:<28} (no samples)"
    return (f"  {label:<28} n={s['n']:<4} p50={s['p50_ms']:>8.1f}  "
            f"p95={s['p95_ms']:>8.1f}  p99={s['p99_ms']:>8.1f}  "
            f"max={s['max_ms']:>8.1f}  (ms)")


def write_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
