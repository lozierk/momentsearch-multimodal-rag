# 3.3 Benchmark harness — how to run it, and where the numbers go

Everything here measures the running app over HTTP, the same endpoints the browser
uses. Nothing imports the app's internals, so a number in this folder is end-to-end.

All scripts live in `momentsearch/scripts/`, run with `momentsearch/.venv/bin/python`,
read `ADMIN_TOKEN` from `momentsearch/.env` themselves (never on argv, never printed),
default to `http://127.0.0.1:8000`, and fail fast with a start-the-stack hint if the API
is not reachable. Results land in this folder as JSON.

## Run order

| # | Script | Worker must be | Writes |
|---|---|---|---|
| 1 | `bench_accept.py` | **STOPPED** (keeps the queue clean) | `accept_<n>.json` |
| 2 | `bench_backfill.py` | **RUNNING** (it has to drain the queue) | `backfill_<run-id>.json` |
| 3 | `run_eval.py` | either (backfill must have finished) | `eval_<run-id>.json`, `citation_check_<run-id>.md` |

Run the backfill before the eval: ten of the fourteen golden queries expect documents
that only the backfill puts in the index. `run_eval.py` prints a warning listing any
expected source that is not indexed yet, so a corpus gap never gets mistaken for a
retrieval failure.

```bash
cd momentsearch

# 1 — accept latency (worker stopped). --n 200 matches the design doc's method.
.venv/bin/python scripts/bench_accept.py --n 100
.venv/bin/python scripts/bench_accept.py --n 200 --strict     # exit 1 if p95 >= 250ms

# 2 — search under backfill (worker running). Registers all 14 corpus PDFs.
.venv/bin/python scripts/bench_backfill.py --run-id 20260728_2130
.venv/bin/python scripts/bench_backfill.py --baseline-only    # Phase A only, uploads nothing
.venv/bin/python scripts/bench_backfill.py --limit 3 --run-id smoke   # small rehearsal

# 3 — retrieval quality + the citation-accuracy sheet
.venv/bin/python scripts/run_eval.py --run-id 20260728_2130
.venv/bin/python scripts/run_eval.py --self-test              # scoring logic only, no API
```

Useful flags: `bench_accept --no-cleanup` (keep the junk rows), `bench_backfill --m 5`
(more baseline rounds), `bench_backfill --no-delete-skipped`, `run_eval --only m1`.

## SLO targets — fill in the right-hand columns after each run

Targets are ours, from `docs/Solution_Design_20260728.md` §3.3. Glimpse is the reference
solution we are benchmarked against, not a target.

| Metric | Target | Glimpse | Measured | Verdict | Run / date |
|---|---|---|---|---|---|
| Accept latency p95 (`POST /api/videos`) | < 250 ms | 171 ms | | | |
| Accept p95, presign+PUT+register | (reported, no target) | — | | | |
| Search p95 during backfill — `ask_full` | ≤ 1.25× baseline | 1.00× (519/518 ms) | | | |
| Search p95 during backfill — `ask_retr` proxy | ≤ 1.25× baseline | — | | | |
| Ingest throughput | report honestly; stretch > 4.0/s | 4.0 chunks/s | pages/s | | |
| Backfill wall clock (14 PDFs) | (reported) | — | | | |
| Recall@10 (14 queries, 4 mixed) | ≥ 0.85 | 1.00 on 3 queries | | | |
| — recall, video-only (5) | | — | | | |
| — recall, doc-only (5) | | — | | | |
| — recall, mixed (4, strict) | | never tested | | | |
| Citation accuracy (human pass) | ≥ 0.90 | no such metric | | | |
| Durability: `kill -9` worker ×4 | 4/4, no loss, no dupes | 4/4 | | | |
| Cost per document | report | no such metric | | | |

Durability and cost-per-document are not covered by these three scripts
(`scripts/kill_worker_test.sh` and the cost accounting are separate work).

## What each number actually means — read before quoting one

**Accept latency.** Two series. `register_only` is the `POST /api/videos` round-trip
alone — that is the SLO, because that call writes one Postgres row and returns 202 while
the worker does everything else. `presign_put_register` adds the presign call and the
byte upload, so it is bounded by the object store rather than by our accept path; it is
reported because it is what an uploader actually feels. Each iteration generates its own
one-page PDF in memory with a random UUID as page text, so every upload has a unique
`source_hash` and content dedup never turns a registration into a `skipped` no-op.
Sequential, single client, no concurrency — this is a latency benchmark, not a load test.

**Search under backfill — the retrieval-latency proxy.** `POST /api/ask` has no
`skip_llm` flag, and this harness does not change `src/`. So two series are measured
on every round:

- `ask_full` — a real golden-set question: retrieval + confidence gate + the multimodal
  LLM call. This is the series the design doc names, and it is what a user waits. It also
  runs in **seconds** (~10–15 s locally), which makes its baseline-vs-during ratio nearly
  insensitive: seconds of model latency swamp tens of milliseconds of contention. The
  script prints that caveat when it detects it.
- `ask_retr` — the *same* question scoped to `video_ids: ["yt_bench_no_src"]`, a source
  id that does not exist. Both query embeddings still run (CLIP text + bge), both Qdrant
  searches come back empty, and `ask()` takes its zero-citation early return without ever
  calling the LLM. Zero tokens, ~350 ms and stable locally. **Read this ratio for the
  decoupling claim.**

  It is a *proxy*: it includes the HTTP round-trip and both query embeddings — the parts
  that contend with an ingest backfill — but excludes HNSW traversal over real candidates,
  RRF fusion, and citation/thumbnail assembly. It understates absolute retrieval latency.
  Use the ratio, not the value.

  The original plan was a nonsense question that trips the abstain gate. It does not
  work: Gate 1 abstains only when the best CLIP score < 0.2 **and** the best bge score
  < 0.35, and CLIP text→image cosines against real frames clear 0.2 even for deliberate
  gibberish. Four probes ("tomato bisque recipe", "golden retriever swimming", "Fiat 500
  torque spec", random letters) each triggered a full LLM answer. Empty scope reaches the
  same zero-LLM path deterministically. Every proxy sample is verified to come back with
  zero citations and `llm_used: false`; contaminated samples are reported.

**Ingest throughput is pages/second, not chunks/second.** `/api/videos` exposes
`frame_count` (pages rendered and embedded per document) but not the number of text
chunks, so chunks/s is not measurable from outside the app. Glimpse's 4.0 chunks/s is
therefore *not* directly comparable to our pages/s — say so when reporting it. Wall clock
runs from the first register to the moment the last row reaches a terminal status.

**`skipped` counts as success.** The Attention paper and the *business_deck_08*
deck are already indexed; re-registering them produces a new row that content dedup marks
`skipped`. That is a correct outcome, counted as a successful backfill, and it contributes
zero pages — which drags pages/s down slightly. `bench_backfill.py` deletes those junk
rows afterwards by default (`--no-delete-skipped` to keep them); the originals stay indexed.

**Recall@10 is a retrieval metric.** It says the expected source came back, not that the
answer is faithful to it. `--top-k` defaults to 10 (the app's own `TOP_K` is 6) so the
metric's name is literal. Mixed queries are scored **strictly**: a mixed query counts as a
hit only if at least one video expectation *and* at least one document expectation appear
in the citations, because a mixed query answered from a single corpus has not demonstrated
fusion. Faithfulness is the separate human pass in `citation_check_<run-id>.md`.

## Golden set

`../../eval/golden_set.json` — 14 queries: 5 video-only, 5 doc-only (3 papers, 2 decks),
4 mixed-corpus. Sources are matched by case-insensitive substring against a citation's
`video_id` or `title` (documents are registered with the filename stem as the title).
`locator_hint` is informational — it tells the human where the answer really lives and is
never scored.
