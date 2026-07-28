"""3.3 — Accept latency: how fast does the write path say 202 and get out of the way?

The claim under test is decoupling: `POST /api/videos` writes one Postgres row
and returns; parsing, rendering, embedding and indexing all happen later in a
worker. So the number that matters is the round-trip of the register call, and
we report it two ways:

  * register_only  — the POST /api/videos round-trip alone. This is the SLO.
  * presign_to_202 — presign + PUT + register, i.e. what an uploader feels.
                     Includes moving the bytes, so it is bounded by the object
                     store, not by our accept path. Reported for honesty.

Each iteration generates its own tiny one-page PDF in memory with a random UUID
printed on the page, so every upload has a different sha256 and the ingest
dedup (`source_hash`) never short-circuits a registration into `skipped`.

    .venv/bin/python scripts/bench_accept.py --n 100

Nothing here waits for ingestion — that is the point of the benchmark. The
registered rows ARE real work: with the worker running they will be picked up
and ingested. **Run this with the worker STOPPED** and the queue stays clean;
`--cleanup` (default ON) then deletes every row it created.

Target (docs/Solution_Design_20260728.md): accept p95 < 250 ms (Glimpse: 171 ms).
"""
from __future__ import annotations

import argparse
import time
import uuid
from datetime import datetime, timezone

import fitz  # PyMuPDF — already a dependency (src/ingest/documents.py)

from bench_common import (BENCH_DIR, DEFAULT_BASE, admin_token, delete_video,
                          fmt_summary, preflight, presign, put_object,
                          register, summarize, write_json)

TARGET_P95_MS = 250.0  # Solution_Design_20260728.md, §3.3


def tiny_pdf(nonce: str) -> bytes:
    """A one-page PDF whose text (and therefore whose sha256) is unique."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 40), f"momentsearch bench {nonce}", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def one_iteration(base: str, token: str | None, kind: str) -> tuple[str, float, float]:
    """Returns (video_id, register_only_ms, presign_to_202_ms)."""
    nonce = uuid.uuid4().hex
    data = tiny_pdf(nonce)
    name = f"bench_{nonce}.pdf"

    t0 = time.perf_counter()
    ps = presign(base, token, name, len(data))
    put_object(base, token, ps, data)
    t_reg = time.perf_counter()
    register(base, token, ps["video_id"], ps["key"], kind, f"bench {nonce[:8]}")
    t_end = time.perf_counter()
    return ps["video_id"], (t_end - t_reg) * 1000, (t_end - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure MomentSearch accept latency (presign -> PUT -> register).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=3,
                    help="untimed iterations first (JIT/pool/connection warm-up)")
    ap.add_argument("--kind", choices=["paper", "deck"], default="paper")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True,
                    help="DELETE every registered id when the run finishes")
    ap.add_argument("--strict", action="store_true",
                    help=f"exit non-zero if p95 >= {TARGET_P95_MS:.0f} ms")
    args = ap.parse_args()

    token = admin_token()
    preflight(args.base, token)
    print(f"[bench-accept] {args.base} · n={args.n} (+{args.warmup} warm-up) · kind={args.kind}")
    print("[note] Ingestion is NOT awaited. Run with the worker STOPPED to keep the "
          "queue clean — recommended.")

    ids: list[str] = []
    reg_ms: list[float] = []
    total_ms: list[float] = []
    started = datetime.now(timezone.utc)
    t_wall = time.perf_counter()

    for i in range(args.warmup + args.n):
        try:
            vid, r, t = one_iteration(args.base, token, args.kind)
        except Exception as e:
            print(f"\n[fatal] iteration {i} failed: {type(e).__name__}: {e}")
            if ids and args.cleanup:
                print(f"[cleanup] removing {len(ids)} rows registered so far…")
                for v in ids:
                    delete_video(args.base, token, v)
            return 1
        ids.append(vid)
        if i >= args.warmup:
            reg_ms.append(r)
            total_ms.append(t)
        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{args.warmup + args.n}  last register={r:.0f}ms")

    wall_s = time.perf_counter() - t_wall
    reg, tot = summarize(reg_ms), summarize(total_ms)
    verdict = "PASS" if reg["p95_ms"] < TARGET_P95_MS else "FAIL"

    print("\n── accept latency ─────────────────────────────────────────────")
    print(fmt_summary("register only (SLO)", reg))
    print(fmt_summary("presign + PUT + register", tot))
    print(f"\n  target: register p95 < {TARGET_P95_MS:.0f} ms  →  "
          f"measured {reg['p95_ms']:.1f} ms  →  {verdict}"
          f"   (Glimpse reference: 171 ms)")
    print(f"  wall clock: {wall_s:.1f}s for {args.warmup + args.n} registrations "
          f"(sequential, one client)")

    cleaned = 0
    if args.cleanup:
        print(f"\n[cleanup] deleting {len(ids)} benchmark rows…")
        cleaned = sum(delete_video(args.base, token, v) for v in ids)
        print(f"[cleanup] {cleaned}/{len(ids)} deleted.")
    else:
        print(f"\n[note] {len(ids)} benchmark rows left in the manifest "
              f"(--cleanup to remove them).")

    out = write_json(BENCH_DIR / f"accept_{args.n}.json", {
        "benchmark": "accept_latency",
        "started_utc": started.isoformat(),
        "base_url": args.base,
        "n": args.n, "warmup": args.warmup, "kind": args.kind,
        "wall_clock_s": round(wall_s, 2),
        "target_p95_ms": TARGET_P95_MS,
        "verdict": verdict,
        "register_only_ms": reg,
        "presign_put_register_ms": tot,
        "cleanup": {"requested": args.cleanup, "registered": len(ids), "deleted": cleaned},
        "method": (
            "Per iteration: a one-page PDF is generated in memory with a random UUID as "
            "page text (unique sha256 -> ingest dedup never marks it skipped), then "
            "presign -> PUT -> POST /api/videos. register_only is the POST round-trip; "
            "presign_put_register is all three. Ingestion is not awaited. Sequential, "
            "single client, no concurrency."),
        "samples_ms": {"register_only": [round(x, 2) for x in reg_ms],
                       "presign_put_register": [round(x, 2) for x in total_ms]},
    })
    print(f"[out] {out}")
    if args.strict and verdict == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
