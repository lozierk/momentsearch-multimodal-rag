"""Upload a PDF through the real API path: presign -> PUT -> register -> poll.

Usage:
    .venv/bin/python scripts/upload_doc.py <file.pdf> paper|deck [--base http://127.0.0.1:8000] [--no-wait]

Reads ADMIN_TOKEN (and nothing else) from .env in the repo root, the same way
the app does, so credentials stay out of shell history and terminals. Exits 0
on `indexed`/`skipped`, 1 on `failed` or timeout. Doubles as the per-file
worker for the 3.3 backfill harness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
POLL_S = 3
TIMEOUT_S = 600


def _call(base: str, method: str, path: str, token: str | None,
          body: bytes | None = None, ctype: str = "application/json") -> dict:
    req = urllib.request.Request(base + path, data=body, method=method)
    req.add_header("Content-Type", ctype)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read() or b"{}")


def admin_token() -> str | None:
    """ADMIN_TOKEN from the repo .env — never from argv, so it stays out of
    shell history, `ps` output and any printed command line."""
    load_dotenv(REPO / ".env")
    return os.environ.get("ADMIN_TOKEN") or None


def register_pdf(base: str, token: str | None, pdf: Path, kind: str) -> tuple[str, float]:
    """presign -> PUT -> register. Returns (video_id, accept_ms).

    Shared with the 3.3 durability harness (scripts/kill_worker_test.py) so both
    go through exactly the same API path a browser upload takes.
    """
    data = pdf.read_bytes()
    t0 = time.monotonic()
    ps = _call(base, "POST", "/api/videos/presign", token, json.dumps({
        "filename": pdf.name, "content_type": "application/pdf",
        "size": len(data)}).encode())
    vid = ps["video_id"]

    if ps["mode"] == "direct":
        _call(base, "PUT", ps["url"], token, data, "application/pdf")
    else:  # presigned: PUT straight to the bucket, no auth header
        req = urllib.request.Request(ps["url"], data=data, method="PUT")
        for k, v in ps.get("headers", {}).items():
            req.add_header(k, v)
        urllib.request.urlopen(req).read()

    _call(base, "POST", "/api/videos", token, json.dumps({
        "video_id": vid, "key": ps["key"], "kind": kind,
        "title": pdf.stem}).encode())
    return vid, (time.monotonic() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("kind", choices=["paper", "deck"])
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-wait", action="store_true",
                    help="return right after the 202 (backfill mode)")
    args = ap.parse_args()

    token = admin_token()
    t0 = time.monotonic()
    vid, accept_ms = register_pdf(args.base, token, args.pdf, args.kind)
    print(f"[upload] {args.pdf.name} -> {vid} accepted in {accept_ms:.0f}ms")
    if args.no_wait:
        return 0

    last = None
    while time.monotonic() - t0 < TIMEOUT_S:
        rows = _call(args.base, "GET", "/api/videos", token)
        row = next((r for r in rows.get("videos", rows if isinstance(rows, list) else [])
                    if r.get("id") == vid), None)
        status = (row or {}).get("status")
        if status != last:
            print(f"[status] {vid}: {status}  (+{time.monotonic() - t0:.0f}s)")
            last = status
        if status in ("indexed", "skipped"):
            return 0
        if status == "failed":
            print(f"[error] {row.get('error')}")
            return 1
        time.sleep(POLL_S)
    print("[error] timed out")
    return 1


if __name__ == "__main__":
    sys.exit(main())
