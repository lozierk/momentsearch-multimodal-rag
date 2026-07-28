"""3.3 durability harness — SIGKILL a worker mid-ingest and prove nothing is lost.

    RECONCILE_STUCK_S=60 .venv/bin/python scripts/kill_worker_test.py

What it does: registers every PDF under ../corpus through the real API, runs its
OWN worker subprocess, `kill -9`s that worker's whole process group once a
document is genuinely mid-flight, waits, starts a fresh worker, and then polls
until every registered id reaches a terminal status.

The point of the test is that `kill -9` raises NO exception — no `except`, no
`finally`, no Prefect retry. Recovery is state-driven: rows left in an inflight
status go stale, the reconciler (src/reconciler.py) puts them back to `pending`,
and the WFQ dispatcher re-admits them. Ingestion is idempotent (uuid5 point
ids), so a re-run cannot duplicate points.

IMPORTANT — recovery latency is bounded by RECONCILE_STUCK_S (+ one
RECONCILE_INTERVAL_S tick). The 600 s default is a production number; for a test
run that finishes this decade, export RECONCILE_STUCK_S=60 (and optionally
RECONCILE_INTERVAL_S=10) before running. The effective values are printed at
start — they are inherited by the worker subprocess this script spawns.

Also: stop any worker you already have running first. This script only kills the
worker it spawned; a second worker would quietly pick the work back up and the
test would prove nothing.

Verdict: PASS = every registered id reached a terminal status, none `failed`,
none lost. Results land in ../docs/bench/kill_worker_<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_doc import _call, admin_token, register_pdf  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO.parent / "corpus"
BENCH = REPO.parent / "docs" / "bench"

TERMINAL = ("indexed", "skipped", "failed")
RUNNING = ("fetching", "sampling", "embedding")   # actually executing, not just queued
POLL_S = 3


def find_pdfs(limit: int | None) -> list[tuple[Path, str]]:
    """(path, kind) for every corpus PDF — folder name picks paper vs deck."""
    out = []
    for p in sorted(CORPUS.rglob("*.pdf")):
        kind = "deck" if "decks" in p.parts else "paper"
        out.append((p, kind))
    return out[:limit] if limit else out


def spawn_worker(tag: str) -> subprocess.Popen:
    """Start `python -m src.worker` in its own process group.

    The group matters: Prefect runs each flow in a CHILD process, so killing
    only the parent would leave the actual ingest running and the test would
    prove nothing. `start_new_session=True` + killpg takes the whole tree down,
    which is what a machine dying really looks like.
    """
    env = dict(os.environ)
    env.setdefault("CLIP_SERVICE_URL", "http://127.0.0.1:8001")
    p = subprocess.Popen([sys.executable, "-m", "src.worker"],
                         cwd=REPO, env=env, start_new_session=True)
    print(f"[worker:{tag}] pid {p.pid} (own process group)")
    return p


def sigkill(p: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    p.wait(timeout=30)


def stop(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=20)
    except Exception:
        sigkill(p)


def statuses(base: str, token: str | None, ids: list[str]) -> dict[str, dict]:
    rows = _call(base, "GET", "/api/videos", token).get("videos", [])
    by_id = {r["id"]: r for r in rows}
    return {i: by_id.get(i, {}) for i in ids}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kill -9 a worker mid-ingest and verify zero loss.",
        epilog="Recovery waits on RECONCILE_STUCK_S (default 600s). Run with "
               "RECONCILE_STUCK_S=60 for a fast test. Stop any other worker first.")
    ap.add_argument("--files", type=int, default=None,
                    help="how many corpus PDFs to register (default: all)")
    ap.add_argument("--down-s", type=float, default=15.0,
                    help="seconds to stay dead before restarting the worker")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="overall seconds to wait for every row to go terminal")
    ap.add_argument("--arm-s", type=float, default=180.0,
                    help="max seconds to wait for a row to be mid-flight before killing anyway")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    stuck_s = float(os.environ.get("RECONCILE_STUCK_S", "600"))
    tick_s = float(os.environ.get("RECONCILE_INTERVAL_S", "60"))
    print("=" * 72)
    print("DURABILITY TEST — stop any worker you already have running first.")
    print("This script spawns and kills only its OWN worker; a second worker")
    print("would silently do the recovery and invalidate the result.")
    print(f"RECONCILE_STUCK_S={stuck_s:.0f}s  RECONCILE_INTERVAL_S={tick_s:.0f}s "
          f"-> expect recovery ~{stuck_s + tick_s:.0f}s after the kill.")
    if stuck_s + tick_s > args.timeout:
        print(f"WARNING: recovery window exceeds --timeout {args.timeout:.0f}s — "
              "the test will time out before the reconciler acts.")
    print("=" * 72)

    pdfs = find_pdfs(args.files)
    if not pdfs:
        print(f"[error] no PDFs under {CORPUS}")
        return 1
    token = admin_token()
    worker: subprocess.Popen | None = None
    t0 = time.monotonic()

    try:
        worker = spawn_worker("first")

        registered: dict[str, dict] = {}
        for path, kind in pdfs:
            vid, accept_ms = register_pdf(args.base, token, path, kind)
            registered[vid] = {"file": path.name, "kind": kind,
                               "accept_ms": round(accept_ms)}
            print(f"[register] {path.name} -> {vid} ({accept_ms:.0f}ms)")
        ids = list(registered)

        # 1. Wait until something is genuinely executing, not merely queued.
        killed_state: dict[str, str] = {}
        armed = False
        while time.monotonic() - t0 < args.arm_s:
            snap = statuses(args.base, token, ids)
            if any(r.get("status") in RUNNING for r in snap.values()):
                killed_state = {i: r.get("status") for i, r in snap.items()}
                armed = True
                break
            time.sleep(POLL_S)
        if not armed:
            snap = statuses(args.base, token, ids)
            killed_state = {i: r.get("status") for i, r in snap.items()}
            print(f"[kill] nothing reached {RUNNING} within {args.arm_s:.0f}s — "
                  "killing anyway (the reconciler still has to recover the queued rows)")

        inflight_at_kill = [i for i, s in killed_state.items()
                            if s not in ("pending",) + TERMINAL]
        print(f"[kill] SIGKILL worker pid {worker.pid} — in-flight at kill: "
              f"{inflight_at_kill or 'none'}")
        sigkill(worker)
        worker = None

        # 2. Stay dead. Nothing is retrying: no process exists to retry.
        print(f"[down] no worker for {args.down_s:.0f}s")
        time.sleep(args.down_s)

        # 3. Fresh worker — it brings up its own dispatcher + reconciler.
        worker = spawn_worker("restarted")

        # 4. Poll to terminal.
        last: dict[str, str] = {}
        while time.monotonic() - t0 < args.timeout:
            snap = statuses(args.base, token, ids)
            now = {i: r.get("status") for i, r in snap.items()}
            for i, s in now.items():
                if last.get(i) != s:
                    print(f"[status] {i}: {last.get(i)} -> {s} "
                          f"(+{time.monotonic() - t0:.0f}s)")
            last = now
            if all(s in TERMINAL for s in now.values()):
                break
            time.sleep(POLL_S)

        # 5. Report.
        final = statuses(args.base, token, ids)
        rows = []
        for i in ids:
            r = final.get(i) or {}
            rows.append({"id": i, **registered[i], "status": r.get("status"),
                         "attempts": r.get("attempts"), "error": r.get("error"),
                         "status_at_kill": killed_state.get(i)})
        lost = [r["id"] for r in rows if not r["status"]]
        failed = [r["id"] for r in rows if r["status"] == "failed"]
        pending = [r["id"] for r in rows if r["status"] not in TERMINAL and r["status"]]
        passed = not lost and not failed and not pending

        print("\n" + "-" * 72)
        print(f"{'id':<18}{'status':<10}{'attempts':<10}{'at kill':<12}file")
        for r in rows:
            print(f"{r['id']:<18}{str(r['status']):<10}{str(r['attempts']):<10}"
                  f"{str(r['status_at_kill']):<12}{r['file']}")
        print("-" * 72)
        print(f"VERDICT: {'PASS' if passed else 'FAIL'} — "
              f"{len(rows) - len(failed) - len(lost) - len(pending)}/{len(rows)} recovered, "
              f"{len(failed)} failed, {len(lost)} lost, {len(pending)} still non-terminal "
              f"after {time.monotonic() - t0:.0f}s")

        BENCH.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = BENCH / f"kill_worker_{stamp}.json"
        out.write_text(json.dumps({
            "test": "3.3 durability — SIGKILL worker mid-ingest",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"reconcile_stuck_s": stuck_s, "reconcile_interval_s": tick_s,
                       "down_s": args.down_s, "timeout_s": args.timeout,
                       "files": len(rows), "base": args.base},
            "inflight_at_kill": inflight_at_kill,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "passed": passed, "failed": failed, "lost": lost,
            "non_terminal": pending, "documents": rows,
        }, indent=2))
        print(f"[bench] wrote {out}")
        return 0 if passed else 1
    finally:
        stop(worker)


if __name__ == "__main__":
    sys.exit(main())
