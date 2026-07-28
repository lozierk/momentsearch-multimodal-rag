"""3.3 — Search/ingest decoupling: does the read path hold up during a backfill?

Phase A  measure query latency on an idle system (baseline).
Phase B  register every PDF in ../corpus/ without waiting for any of it, then
         re-run the same measurement loop, round after round, until every
         backfilled row reaches a terminal status. Compare.

WHAT IS ACTUALLY BEING TIMED — read this before quoting a number
----------------------------------------------------------------
`POST /api/ask` has no `skip_llm` flag (checked: src/api/search.py AskRequest
takes question/video_id/video_ids/top_k only; src/rag/search.py::ask always
runs the LLM once the confidence gate passes). Adding one is an API change and
this harness does not touch src/, so we measure two series every round:

  ask_full    a real golden-set question. Retrieval + confidence gate + the
              multimodal LLM call. Seconds, and dominated by someone else's
              network. This is the series the design doc's SLO names, but the
              LLM's variance can mask a retrieval regression in either direction.

  ask_retr    the SAME question, scoped to a source id that does not exist
              (`video_ids: ["<sentinel>"]`). Same HTTP entry, same request
              validation, same CLIP text embedding and same bge query embedding
              — then both filtered Qdrant searches return nothing, so ask()
              takes its zero-citation early return and never calls the LLM
              (`abstained: true`, `llm_used: false`). Zero tokens, measured
              ~350 ms locally and stable.

  WHY NOT the abstain gate, as originally planned: it will not fire. Gate 1
  abstains only when the best CLIP score < 0.2 AND the best bge score < 0.35,
  and CLIP text->image cosines against a few hundred real frames clear 0.2 even
  for deliberate gibberish — four separate nonsense probes ("tomato bisque
  recipe", "golden retriever swimming", "Fiat 500 torque spec", random letters)
  all answered with a real LLM call. The empty-scope request reaches the same
  zero-LLM early return by a mechanism that is deterministic instead of hopeful.

  ask_retr is a PROXY, not "retrieval latency". It includes both query
  embeddings and the API round-trip (the parts that contend with an ingest
  backfill for CPU and for the CLIP service) but NOT the HNSW traversal over
  real candidates, the RRF fusion, or the citation/thumbnail assembly that a
  hit performs. It therefore UNDERSTATES absolute retrieval latency; its value
  is the ratio, baseline vs during backfill. Every sample is verified to have
  come back `abstained` with zero citations — if one does not, the run says so.

Throughput is reported as PAGES/SECOND, not chunks/second: /api/videos exposes
`frame_count` (pages rendered + embedded for a document) but not the number of
text chunks, so chunks/s is not measurable from outside the app. Stated, not
fudged.

Needs the WORKER RUNNING (the opposite of bench_accept.py).

    .venv/bin/python scripts/bench_backfill.py --run-id 20260728_2130

Targets (docs/Solution_Design_20260728.md §3.3):
  search p95 during backfill <= 1.25x baseline (Glimpse: 1.00x)
  ingest throughput: report honestly; Glimpse's actual was 4.0 chunks/s.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_common import (BENCH_DIR, CORPUS_DIR, DEFAULT_BASE, EVAL_DIR,
                          TERMINAL_STATUSES, admin_token, delete_video, die,
                          fmt_summary, list_videos, post_json, preflight,
                          presign, put_object, register, summarize, write_json)

TARGET_RATIO = 1.25   # Solution_Design_20260728.md, §3.3
INFLIGHT = ("pending", "queued", "fetching", "sampling", "embedding")

# A source id no row will ever have. Scoping a question to it makes both Qdrant
# branches return nothing, which is the app's zero-citation / zero-LLM path.
EMPTY_SCOPE_ID = "yt_bench_no_src"


def corpus_files(limit: int | None) -> list[tuple[Path, str]]:
    """(path, kind) for every PDF under ../corpus/ — folder decides the kind."""
    out: list[tuple[Path, str]] = []
    for folder, kind in (("papers", "paper"), ("decks", "deck")):
        out += [(p, kind) for p in sorted((CORPUS_DIR / folder).glob("*.pdf"))]
    if not out:
        die(f"no PDFs under {CORPUS_DIR}/papers or {CORPUS_DIR}/decks")
    return out[:limit] if limit else out


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        die(f"golden set not found: {path} (needed for the query loop)")
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data["queries"] if isinstance(data, dict) else data
    return [q["question"] for q in qs]


def validate_proxy(base: str, token: str | None, question: str) -> None:
    """The retrieval proxy is only valid if the empty scope really does return
    zero citations without an LLM call. Check once, up front, loudly."""
    r = post_json(base, "/api/ask", token,
                  {"question": question, "video_ids": [EMPTY_SCOPE_ID]})
    if r.get("citations") or r.get("llm_used"):
        die(f"the retrieval proxy is invalid: scoping to {EMPTY_SCOPE_ID!r} returned "
            f"{len(r.get('citations') or [])} citations / llm_used="
            f"{r.get('llm_used')}. A row with that id must exist — change "
            f"EMPTY_SCOPE_ID in this script.")


def measure_round(base: str, token: str | None, questions: list[str],
                  top_k: int | None) -> tuple[list[float], list[float], int, int]:
    """One interleaved pass over the query set.

    Returns (full_ms, retr_ms, n_valid_retr, n_llm_used_full) — the last two are
    what let the report label each series honestly."""
    full, retr, valid, llm_used = [], [], 0, 0
    for q in questions:
        body: dict = {"question": q}
        if top_k:
            body["top_k"] = top_k
        t0 = time.perf_counter()
        r = post_json(base, "/api/ask", token, body)
        full.append((time.perf_counter() - t0) * 1000)
        llm_used += int(bool(r.get("llm_used")))

        # Same question (so the embedding work is identical), empty scope, so
        # the response comes back before any LLM call. Interleaved so both
        # series see the same system state.
        t0 = time.perf_counter()
        r = post_json(base, "/api/ask", token,
                      {**body, "video_ids": [EMPTY_SCOPE_ID]})
        retr.append((time.perf_counter() - t0) * 1000)
        valid += int(not r.get("citations") and not r.get("llm_used"))
    return full, retr, valid, llm_used


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prove search stays fast while a document backfill runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--m", type=int, default=3, help="baseline rounds over the query set")
    ap.add_argument("--golden", type=Path, default=EVAL_DIR / "golden_set.json")
    ap.add_argument("--top-k", type=int, default=None,
                    help="citations per query (default: the API's TOP_K)")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--limit", type=int, default=None,
                    help="only backfill the first N corpus PDFs (smoke runs)")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="seconds to wait for the backfill to reach terminal states")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Phase A only — no uploads, nothing registered")
    ap.add_argument("--delete-skipped", action=argparse.BooleanOptionalAction, default=True,
                    help="delete rows dedup marked `skipped` (junk manifest rows, no vectors)")
    args = ap.parse_args()

    token = admin_token()
    preflight(args.base, token)
    questions = load_questions(args.golden)
    files = corpus_files(args.limit)
    print(f"[bench-backfill] {args.base} · {len(questions)} queries × {args.m} baseline "
          f"rounds · {len(files)} corpus PDFs · run-id={args.run_id}")
    print("[note] The WORKER must be running for Phase B to make progress.")
    print(f"[note] ask_full = real question (LLM runs). ask_retr = same question "
          f"scoped to a non-existent source → zero-LLM retrieval proxy.")
    started = datetime.now(timezone.utc)
    validate_proxy(args.base, token, questions[0])

    # ── Phase A — baseline ───────────────────────────────────────────────────
    print("\n── Phase A: baseline (idle) ───────────────────────────────────")
    base_full, base_retr, valid, llm = [], [], 0, 0
    for r in range(args.m):
        f, p, g, u = measure_round(args.base, token, questions, args.top_k)
        base_full += f
        base_retr += p
        valid += g
        llm += u
        print(f"  round {r + 1}/{args.m}: ask_full p50={summarize(f)['p50_ms']:.0f}ms  "
              f"ask_retr p50={summarize(p)['p50_ms']:.0f}ms")
    bf, bp = summarize(base_full), summarize(base_retr)
    print(fmt_summary("baseline ask_full", bf))
    print(fmt_summary("baseline ask_retr", bp))
    proxy_clean = valid == len(base_retr)
    print(f"  zero-LLM path confirmed on {valid}/{len(base_retr)} ask_retr queries"
          f"{'' if proxy_clean else '  ← PROXY CONTAMINATED, see caveats'}")
    print(f"  LLM synthesis ran on {llm}/{len(base_full)} ask_full queries"
          f"{' (none — ask_full is retrieval + fallback only)' if not llm else ''}")

    if args.baseline_only:
        write_json(BENCH_DIR / f"backfill_{args.run_id}.json", {
            "benchmark": "search_under_backfill", "run_id": args.run_id,
            "phase": "baseline_only", "baseline": {"ask_full": bf, "ask_retr": bp},
            "proxy_valid_samples": valid, "proxy_clean": proxy_clean})
        print("[done] --baseline-only: nothing was registered.")
        return 0

    # ── Phase B — register everything, then keep measuring ───────────────────
    print(f"\n── Phase B: registering {len(files)} PDFs (no waiting) ────────")
    ids: dict[str, dict] = {}
    t_backfill0 = time.perf_counter()
    accept_ms: list[float] = []
    for path, kind in files:
        data = path.read_bytes()
        t0 = time.perf_counter()
        ps = presign(args.base, token, path.name, len(data))
        put_object(args.base, token, ps, data)
        register(args.base, token, ps["video_id"], ps["key"], kind, path.stem)
        ms = (time.perf_counter() - t0) * 1000
        accept_ms.append(ms)
        ids[ps["video_id"]] = {"file": path.name, "kind": kind, "accept_ms": round(ms, 1)}
        print(f"  {ps['video_id']}  {kind:<5} {ms:6.0f}ms  {path.name}")
    print(f"  all {len(ids)} registered in {(time.perf_counter() - t_backfill0):.1f}s")

    print("\n── Phase B: measuring while the queue drains ──────────────────")
    rounds, dur_full, dur_retr = [], [], []
    dvalid = dllm = 0
    t_terminal = None
    warned_stalled = False
    while time.perf_counter() - t_backfill0 < args.timeout:
        r_t0 = time.perf_counter()
        f, p, g, u = measure_round(args.base, token, questions, args.top_k)
        dur_full += f
        dur_retr += p
        dvalid += g
        dllm += u
        rows = list_videos(args.base, token)
        states = {vid: (rows.get(vid) or {}).get("status", "missing") for vid in ids}
        remaining = [v for v, s in states.items() if s not in TERMINAL_STATUSES]
        rounds.append({
            "round": len(rounds) + 1,
            "t_offset_s": round(r_t0 - t_backfill0, 1),
            "ask_full": summarize(f), "ask_retr": summarize(p),
            "remaining": len(remaining),
            "status_counts": {s: sum(1 for x in states.values() if x == s)
                              for s in sorted(set(states.values()))},
        })
        print(f"  round {len(rounds):>2} (+{r_t0 - t_backfill0:6.0f}s): "
              f"ask_full p50={summarize(f)['p50_ms']:>7.0f}ms  "
              f"ask_retr p50={summarize(p)['p50_ms']:>7.0f}ms  "
              f"remaining={len(remaining):>2}  {rounds[-1]['status_counts']}")
        if not remaining:
            t_terminal = time.perf_counter()
            break
        if (not warned_stalled and time.perf_counter() - t_backfill0 > 180
                and all(states[v] == "pending" for v in remaining)):
            warned_stalled = True
            print("  [warn] every row is still `pending` after 3 minutes — is the "
                  "worker running? (`python -m src.worker`)")
    if t_terminal is None:
        print(f"  [warn] timed out after {args.timeout}s with rows still in flight; "
              f"throughput below is a LOWER BOUND.")
        t_terminal = time.perf_counter()

    wall_s = t_terminal - t_backfill0
    rows = list_videos(args.base, token)
    indexed = [v for v in ids if (rows.get(v) or {}).get("status") == "indexed"]
    skipped = [v for v in ids if (rows.get(v) or {}).get("status") == "skipped"]
    failed = [v for v in ids if (rows.get(v) or {}).get("status") == "failed"]
    stuck = [v for v in ids
             if (rows.get(v) or {}).get("status") in INFLIGHT]
    pages = sum(int((rows.get(v) or {}).get("frame_count") or 0) for v in indexed)

    df, dp = summarize(dur_full), summarize(dur_retr)
    ratio_full = df["p95_ms"] / bf["p95_ms"] if bf.get("p95_ms") else float("nan")
    ratio_retr = dp["p95_ms"] / bp["p95_ms"] if bp.get("p95_ms") else float("nan")
    verdict = "PASS" if ratio_full <= TARGET_RATIO else "FAIL"

    print("\n── results ────────────────────────────────────────────────────")
    print(fmt_summary("baseline ask_full", bf))
    print(fmt_summary("during   ask_full", df))
    print(fmt_summary("baseline ask_retr", bp))
    print(fmt_summary("during   ask_retr", dp))
    print(f"\n  p95 ratio (ask_full, SLO series):      {ratio_full:.2f}x   "
          f"target <= {TARGET_RATIO}x  →  {verdict}   (Glimpse: 1.00x)")
    print(f"  p95 ratio (ask_retr, retrieval proxy): {ratio_retr:.2f}x   "
          f"← cleaner read: no LLM in this path (understates absolute latency)")
    if not proxy_clean or dvalid != len(dur_retr):
        print("  [caveat] some ask_retr samples came back with citations or an LLM call "
              f"(clean: baseline {valid}/{len(base_retr)}, during {dvalid}/{len(dur_retr)}) "
              "— the retrieval proxy is contaminated for those samples.")
    if not (llm or dllm):
        print("  [caveat] no LLM ran on ANY ask_full query (no model configured, or "
              "everything abstained) — ask_full is then retrieval + fallback, i.e. "
              "nearly the same path as ask_retr, and the SLO ratio is not testing "
              "what it claims to.")
    elif bf.get("p50_ms", 0) > 3000:
        print(f"  [caveat] ask_full p50 is {bf['p50_ms'] / 1000:.1f}s — the remote LLM "
              "call dominates, so its ratio is INSENSITIVE to retrieval slowdown "
              "(seconds of model latency swamp tens of ms of contention). Read the "
              "ask_retr ratio for the decoupling claim; quote ask_full for what a "
              "user waits.")
    print(f"\n  backfill wall clock:  {wall_s:.1f}s for {len(ids)} documents")
    print(f"  indexed {len(indexed)} · skipped {len(skipped)} (dedup — counts as success) "
          f"· failed {len(failed)} · still in flight {len(stuck)}")
    print(f"  ingest throughput: {pages} pages / {wall_s:.1f}s = "
          f"{pages / wall_s:.2f} pages/s")
    print("  (pages = sum of frame_count over rows that reached `indexed`; text-chunk "
          "counts are not exposed by /api/videos, so chunks/s is not measurable here.)")
    if failed:
        for v in failed:
            print(f"  [failed] {v} {ids[v]['file']}: {(rows.get(v) or {}).get('error')}")

    deleted = []
    if args.delete_skipped and skipped:
        print(f"\n[cleanup] deleting {len(skipped)} `skipped` duplicate rows "
              f"(no vectors, no thumbnails — the original stays indexed)…")
        deleted = [v for v in skipped if delete_video(args.base, token, v)]
        print(f"[cleanup] {len(deleted)}/{len(skipped)} deleted.")

    out = write_json(BENCH_DIR / f"backfill_{args.run_id}.json", {
        "benchmark": "search_under_backfill",
        "run_id": args.run_id,
        "started_utc": started.isoformat(),
        "base_url": args.base,
        "queries": len(questions), "baseline_rounds": args.m,
        "during_rounds": len(rounds), "top_k": args.top_k,
        "target_ratio": TARGET_RATIO, "verdict": verdict,
        "baseline": {"ask_full": bf, "ask_retr": bp,
                     "retr_zero_llm_samples": valid, "llm_used": llm},
        "during": {"ask_full": df, "ask_retr": dp,
                   "retr_zero_llm_samples": dvalid, "llm_used": dllm},
        "p95_ratio": {"ask_full": round(ratio_full, 3),
                      "ask_retr": round(ratio_retr, 3)},
        "backfill": {
            "documents": len(ids), "indexed": len(indexed), "skipped": len(skipped),
            "failed": failed, "still_in_flight": stuck,
            "wall_clock_s": round(wall_s, 1),
            "pages_indexed": pages,
            "pages_per_s": round(pages / wall_s, 3) if wall_s else None,
            "accept_ms": summarize(accept_ms),
            "timed_out": bool(stuck),
            "deleted_skipped": deleted,
            "rows": {v: {**meta, "status": (rows.get(v) or {}).get("status"),
                         "frame_count": (rows.get(v) or {}).get("frame_count")}
                     for v, meta in ids.items()},
        },
        "rounds": rounds,
        "method": {
            "ask_full": "POST /api/ask with a golden-set question — retrieval + gate + "
                        "multimodal LLM. The SLO series named in the design doc.",
            "ask_retr": f"POST /api/ask with the SAME question scoped to video_ids="
                        f"['{EMPTY_SCOPE_ID}'], a source id that does not exist. Both "
                        "query embeddings still run; both Qdrant searches return nothing, "
                        "so ask() takes its zero-citation early return and never calls "
                        "the LLM. Retrieval-only PROXY: includes HTTP + CLIP text embed + "
                        "bge query embed, excludes HNSW traversal over real candidates, "
                        "RRF fusion and citation assembly. Understates absolute retrieval "
                        "latency; use the ratio, not the value.",
            "skip_llm_flag": "Not supported by the API (src/api/search.py AskRequest has "
                             "no such field) and adding one would mean changing src/, "
                             "which this harness does not do.",
            "why_not_abstain_gate": "The planned proxy was a nonsense question tripping "
                                    "Gate 1, but the gate abstains only when best CLIP "
                                    "< 0.2 AND best bge < 0.35, and CLIP text->image "
                                    "cosines over real frames clear 0.2 even for "
                                    "gibberish — four nonsense probes all triggered a "
                                    "real LLM call. Empty scope is deterministic.",
            "throughput": "pages/s = sum(frame_count of rows reaching `indexed`) / wall "
                          "clock from first register to last terminal status. Text-chunk "
                          "counts are not exposed by the API, so chunks/s is not measured.",
            "skipped": "`skipped` = content dedup matched an already-indexed document "
                       "(same source_hash). Counted as a successful backfill outcome; it "
                       "contributes no pages, which drags pages/s down slightly.",
        },
    })
    print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
