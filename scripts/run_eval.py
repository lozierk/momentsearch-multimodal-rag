"""3.3 — Retrieval quality: recall@10 over the golden set, plus the citation-check sheet.

Reads ../eval/golden_set.json, asks every question through POST /api/ask, and
scores whether the citations contain the sources we expected.

  recall@10 = queries hit / queries asked, with top_k forced to 10.
              (The app's default TOP_K is 6 — asking for 10 is what makes the
              metric's name true. Override with --top-k.)

  hit rule   video / doc  : >=1 expected source appears in the citations.
             mixed        : at least one VIDEO expectation AND at least one
                            DOCUMENT expectation both appear. STRICTER on
                            purpose — the whole claim of a mixed corpus is that
                            one query fuses both, and a mixed query answered
                            from a single corpus has not demonstrated that.

Recall is a retrieval metric: it says the right source came back, not that the
answer is faithful to it. Faithfulness is the separate human pass — this script
writes `citation_check_<run-id>.md` with every answer and every locator laid out
for a yes/no judgement (target: >= 0.9 supported).

    .venv/bin/python scripts/run_eval.py --run-id 20260728_2130
    .venv/bin/python scripts/run_eval.py --self-test   # scoring logic only, no API
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_common import (BENCH_DIR, DEFAULT_BASE, EVAL_DIR, admin_token, die,
                          list_videos, post_json, preflight, pctl, write_json)

TARGET_RECALL = 0.85          # Solution_Design_20260728.md, §3.3
TARGET_CITATION_ACC = 0.90


# ── Scoring (pure functions — exercised by --self-test) ──────────────────────

def is_video_pattern(pattern: str) -> bool:
    """Videos are yt_<id>; everything else is a document filename stem."""
    return pattern.lower().startswith("yt_")


def citation_matches(pattern: str, citation: dict) -> bool:
    p = pattern.lower().strip()
    hay = f"{citation.get('video_id') or ''}\n{citation.get('title') or ''}".lower()
    return bool(p) and p in hay


def score_query(query: dict, citations: list[dict]) -> dict:
    """Which expectations were met, and does that count as a hit for this kind?"""
    matched = [e["source_id_pattern"] for e in query["expect"]
               if any(citation_matches(e["source_id_pattern"], c) for c in citations)]
    kind = query.get("kind", "doc")
    if kind == "mixed":
        hit = (any(is_video_pattern(p) for p in matched)
               and any(not is_video_pattern(p) for p in matched))
        why = "needs one video AND one document expectation"
    else:
        hit = bool(matched)
        why = "needs >=1 expected source"
    return {
        "hit": hit,
        "rule": why,
        "matched_patterns": matched,
        "missed_patterns": [e["source_id_pattern"] for e in query["expect"]
                            if e["source_id_pattern"] not in matched],
        "found_sources": sorted({f"{c.get('video_id')} ({c.get('timestamp')})"
                                 for c in citations}),
    }


def aggregate(results: list[dict]) -> dict:
    by_kind: dict[str, dict] = {}
    for r in results:
        b = by_kind.setdefault(r["kind"], {"asked": 0, "hit": 0})
        b["asked"] += 1
        b["hit"] += int(r["hit"])
    for b in by_kind.values():
        b["recall"] = round(b["hit"] / b["asked"], 3) if b["asked"] else 0.0
    asked = len(results)
    hits = sum(int(r["hit"]) for r in results)
    return {"asked": asked, "hit": hits,
            "recall": round(hits / asked, 3) if asked else 0.0,
            "by_kind": by_kind}


# ── Self-test: prove the scoring logic without touching the API ──────────────

def self_test() -> int:
    cases = [
        ("video hit",
         {"kind": "video", "expect": [{"source_id_pattern": "yt_wjZofJX0v4M"}]},
         [{"video_id": "yt_wjZofJX0v4M", "title": "Transformers", "timestamp": "04:10"}],
         True),
        ("video miss",
         {"kind": "video", "expect": [{"source_id_pattern": "yt_wjZofJX0v4M"}]},
         [{"video_id": "yt_zjkBMFhNj_g", "title": "Karpathy", "timestamp": "10:00"}],
         False),
        ("doc hit by title substring",
         {"kind": "doc", "expect": [{"source_id_pattern": "Clip_Paper_2103.00020"}]},
         [{"video_id": "doc_abc123", "title": "Clip_Paper_2103.00020v1", "timestamp": "p. 2"}],
         True),
        ("doc hit is case-insensitive",
         {"kind": "doc", "expect": [{"source_id_pattern": "somml health_rmg"}]},
         [{"video_id": "doc_def456", "title": "Somml Health_RMG - 20210907.pptx",
           "timestamp": "slide 2"}],
         True),
        ("mixed hit needs both corpora",
         {"kind": "mixed", "expect": [{"source_id_pattern": "yt_eMlx5fFNoYc"},
                                      {"source_id_pattern": "Attention_is_All_You_Need"}]},
         [{"video_id": "yt_eMlx5fFNoYc", "title": "Attention", "timestamp": "08:00"},
          {"video_id": "doc_1c8e196fec", "title": "Attention_is_All_You_Need_1706.03762v7",
           "timestamp": "p. 4"}],
         True),
        ("mixed miss — video only",
         {"kind": "mixed", "expect": [{"source_id_pattern": "yt_eMlx5fFNoYc"},
                                      {"source_id_pattern": "Attention_is_All_You_Need"}]},
         [{"video_id": "yt_eMlx5fFNoYc", "title": "Attention", "timestamp": "08:00"}],
         False),
        ("mixed miss — doc only",
         {"kind": "mixed", "expect": [{"source_id_pattern": "yt_eMlx5fFNoYc"},
                                      {"source_id_pattern": "Attention_is_All_You_Need"}]},
         [{"video_id": "doc_1c8e196fec", "title": "Attention_is_All_You_Need_1706.03762v7",
           "timestamp": "p. 4"}],
         False),
        ("no citations at all (abstained)",
         {"kind": "doc", "expect": [{"source_id_pattern": "RAG_Survey"}]}, [], False),
    ]
    bad = 0
    for name, q, cits, want in cases:
        got = score_query(q, cits)["hit"]
        ok = got == want
        bad += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: hit={got} expected={want}")

    agg = aggregate([{"kind": "video", "hit": True}, {"kind": "video", "hit": False},
                     {"kind": "doc", "hit": True}, {"kind": "mixed", "hit": True}])
    ok = agg["recall"] == 0.75 and agg["by_kind"]["video"]["recall"] == 0.5
    bad += not ok
    print(f"  [{'ok ' if ok else 'FAIL'}] aggregate: overall={agg['recall']} "
          f"video={agg['by_kind']['video']['recall']}")

    # The real golden set must load and be well-formed.
    try:
        g = load_golden(EVAL_DIR / "golden_set.json")
        kinds = {}
        for q in g:
            kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
            assert q["expect"] and all(e.get("source_id_pattern") for e in q["expect"])
            if q["kind"] == "mixed":
                pats = [e["source_id_pattern"] for e in q["expect"]]
                assert any(is_video_pattern(p) for p in pats), f"{q['id']}: no video expectation"
                assert any(not is_video_pattern(p) for p in pats), f"{q['id']}: no doc expectation"
        print(f"  [ok ] golden_set.json loads: {len(g)} queries {kinds}")
    except Exception as e:
        bad += 1
        print(f"  [FAIL] golden_set.json: {type(e).__name__}: {e}")

    print("self-test: " + ("PASS" if not bad else f"{bad} FAILURE(S)"))
    return 1 if bad else 0


# ── Runner ───────────────────────────────────────────────────────────────────

def load_golden(path: Path) -> list[dict]:
    if not path.exists():
        die(f"golden set not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data["queries"] if isinstance(data, dict) else data
    for q in queries:
        if not {"id", "question", "expect", "kind"} <= set(q):
            die(f"golden set entry missing a required field: {q}")
    return queries


def check_corpus(base: str, token: str | None, queries: list[dict]) -> list[str]:
    """Warn about expectations whose source is not in the manifest yet — a miss
    caused by 'never ingested' is not a retrieval failure and must not read as one."""
    rows = list_videos(base, token)
    hay = [f"{r['id']}\n{r.get('title') or ''}".lower() for r in rows.values()]
    absent = sorted({e["source_id_pattern"] for q in queries for e in q["expect"]
                     if not any(e["source_id_pattern"].lower() in h for h in hay)})
    if absent:
        print("[warn] these expected sources are NOT indexed — run the backfill first:")
        for p in absent:
            print(f"         · {p}")
        print("       (scripts/bench_backfill.py, or scripts/upload_doc.py <file> paper|deck)")
    return absent


def citation_check_md(run_id: str, results: list[dict]) -> str:
    n_cites = sum(len(r["citations"]) for r in results)
    out = [
        f"# Citation accuracy check — run `{run_id}`",
        "",
        f"Generated by `scripts/run_eval.py`. {len(results)} answers, {n_cites} citations.",
        "",
        "**How to score:** open each citation's locator (timestamp / page / slide) and mark",
        "`y` if the cited moment actually supports the sentence it is attached to, `n` if it",
        "does not. A citation that is merely *topically related* but does not contain the",
        "claim is an `n`.",
        "",
        f"`citation_accuracy = supported / total` — target **>= {TARGET_CITATION_ACC:.2f}** "
        "(Solution_Design_20260728.md §3.3; a metric Glimpse did not have).",
        "",
        "Fill in the tally at the bottom when done.",
        "",
        "---",
        "",
    ]
    for r in results:
        out += [f"## {r['id']} · {r['kind']} · {'HIT' if r['hit'] else 'MISS'}",
                "", f"**Q:** {r['question']}", ""]
        if r.get("abstained"):
            out += ["_Abstained — no citations to check._", ""]
        out += [f"**Answer:** {r.get('answer', '') or '(none)'}", "",
                "| # | source | locator | modalities | supports? (y/n) | note |",
                "|---|---|---|---|---|---|"]
        for c in r["citations"]:
            title = (c.get("title") or c.get("video_id") or "?").replace("|", "/")
            out.append(f"| {c.get('n')} | {title} | {c.get('timestamp')} | "
                       f"{'+'.join(c.get('modalities') or [])} |  |  |")
        out += ["", ""]
    out += ["---", "", "## Tally", "", f"- citations checked: ____ / {n_cites}",
            "- supported (`y`): ____", "- unsupported (`n`): ____",
            "- **citation_accuracy: ____**", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score retrieval quality (recall@10) over ../eval/golden_set.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--golden", type=Path, default=EVAL_DIR / "golden_set.json")
    ap.add_argument("--top-k", type=int, default=10,
                    help="citations requested per query (10 makes 'recall@10' literal)")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--only", help="run a single query id (debugging)")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise the scoring logic on stubbed responses and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    queries = load_golden(args.golden)
    if args.only:
        queries = [q for q in queries if q["id"] == args.only] or die(
            f"no query with id {args.only!r}")

    token = admin_token()
    preflight(args.base, token)
    print(f"[run-eval] {args.base} · {len(queries)} queries · top_k={args.top_k} "
          f"· run-id={args.run_id}")
    absent = check_corpus(args.base, token, queries)
    print()

    results, latencies = [], []
    for q in queries:
        t0 = time.perf_counter()
        try:
            resp = post_json(args.base, "/api/ask", token,
                             {"question": q["question"], "top_k": args.top_k})
        except Exception as e:
            die(f"POST /api/ask failed on {q['id']}: {type(e).__name__}: {e}")
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        cits = resp.get("citations", [])
        sc = score_query(q, cits)
        row = {"id": q["id"], "kind": q["kind"], "question": q["question"],
               "expect": q["expect"], "latency_ms": round(ms, 1),
               "abstained": bool(resp.get("abstained")),
               "llm_used": bool(resp.get("llm_used")),
               "answer": resp.get("answer", ""), "citations": cits, **sc}
        results.append(row)

        mark = "HIT " if sc["hit"] else "MISS"
        print(f"[{mark}] {q['id']:<3} {q['kind']:<5} {ms:7.0f}ms  {q['question'][:66]}")
        found = ", ".join(f"{c.get('title') or c.get('video_id')} @ {c.get('timestamp')}"
                          for c in cits[:5]) or "(no citations)"
        print(f"        found: {found}")
        if not sc["hit"]:
            print(f"        missed: {', '.join(sc['missed_patterns'])}"
                  f"   [{sc['rule']}]")
        if row["abstained"]:
            print("        note: the abstain gate fired — retrieval scored below "
                  "CONFIDENCE_THRESHOLD")

    agg = aggregate(results)
    print("\n── recall@%d ──────────────────────────────────────────────────" % args.top_k)
    for kind in ("video", "doc", "mixed"):
        b = agg["by_kind"].get(kind)
        if b:
            extra = "  (strict: video AND doc must both appear)" if kind == "mixed" else ""
            print(f"  {kind:<6} {b['hit']}/{b['asked']}  recall={b['recall']:.2f}{extra}")
    verdict = "PASS" if agg["recall"] >= TARGET_RECALL else "FAIL"
    print(f"  {'OVERALL':<6} {agg['hit']}/{agg['asked']}  recall={agg['recall']:.2f}"
          f"   target >= {TARGET_RECALL:.2f}  →  {verdict}")
    print(f"  ask latency: p50={pctl(latencies, 50):.0f}ms  p95={pctl(latencies, 95):.0f}ms "
          f"(with LLM synthesis where configured)")
    if absent:
        print(f"  [caveat] {len(absent)} expected source(s) were not indexed at run time — "
              f"those misses are corpus gaps, not retrieval failures.")

    out = write_json(BENCH_DIR / f"eval_{args.run_id}.json", {
        "benchmark": "recall_at_k",
        "run_id": args.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base,
        "top_k": args.top_k,
        "golden_set": str(args.golden),
        "target_recall": TARGET_RECALL,
        "verdict": verdict,
        "aggregate": agg,
        "ask_latency_ms": {"p50": round(pctl(latencies, 50), 1),
                           "p95": round(pctl(latencies, 95), 1)},
        "sources_not_indexed": absent,
        "scoring_rules": {
            "video_doc": ">=1 expected source in citations",
            "mixed": "at least one video AND one document expectation (stricter)",
            "match": "case-insensitive substring of citation.video_id or citation.title",
        },
        "results": results,
    })
    md = BENCH_DIR / f"citation_check_{args.run_id}.md"
    md.write_text(citation_check_md(args.run_id, results), encoding="utf-8")
    print(f"[out] {out}\n[out] {md}  ← fill this in by hand for citation accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
