# 3.4 — Think Like a Forward Deployed Engineer

Assignment 3, Moment Search. Date: 2026-07-28.
Every number below comes from a file in `docs/bench/`. Nothing is estimated.

---

## 1. What we built

We extended a video-moment search app so PDF papers and slide decks are searchable in the same
index, through the same queue, with the same citations. Documents enter through the existing
`POST /api/videos` path, get parsed by PyMuPDF into page-aware text chunks and per-page JPEGs, and
land in the two collections video already used: chunks in `moments_text`, rendered pages
CLIP-embedded into `moments`. That second move is the differentiator — a slide is a picture, so a
deck becomes *visually* searchable, and a page matching both visually and textually earns the same
1.5x cross-modal boost a video moment does. Retrieval groups video hits by time window and
document hits by `(doc_id, page)`, because for a document the page *is* the moment. A second
Prefect deployment (`ms-ingest-doc`) reuses the video status lifecycle verbatim, so the UI chips
and the fair dispatcher needed zero changes. Then the benchmark changed the code.

| Metric | Target (ours) | Measured | Verdict | Glimpse |
|---|---|---|---|---|
| Accept p95, `POST /api/videos` | < 250 ms | **180 ms** (n=200) | PASS | 171 ms |
| Accept p95, presign+PUT+register | reported | 187 ms | — | not measured |
| Search p95 during backfill — `ask_full` | <= 1.25x | **1.05x** (15,178 -> 15,882 ms) | PASS | 1.00x (519/518 ms) |
| Search p95 during backfill — `ask_retr` proxy | <= 1.25x | 2.43x (882 -> 2,141 ms) | see §2 | not measured |
| Ingest throughput | report; stretch > 4.0/s | **1.33 pages/s** (237 pages / 178.5 s) | reported | 4.0 chunks/s (not comparable) |
| Recall@10, 14 queries (4 mixed, strict) | >= 0.85 | **12/14 (0.86)** | PASS | 1.00 on 3 queries |
| — video-only (5) / doc-only (5) / mixed (4) | — | 1.00 / 1.00 / 0.50 | — | mixed never tested |
| Durability, SIGKILL mid-ingest | no loss, no dupes | **14/14 recovered, 0 lost** | PASS | 4/4 recovered |
| Citation locator coverage | 100% | **140/140 citations carry a locator** | PASS | asserted, not counted |
| Citation accuracy | >= 0.90 | **40/43 (0.93)** claim-bearing ✓; 3 ✗; 2 visual claims excluded as transcript-unverifiable; AI-assisted first pass, human spot-check pending | PASS | no such metric |
| Cost per document | report | **not measured** — see §7 | OPEN | no such metric |

Sources: `accept_200.json`, `backfill_20260728A.json`, `eval_20260728B.json`,
`kill_worker_20260728_220849.json`, `citation_check_20260728B.md`. Values are rounded for
display — count-based scores carry their true precision as fractions; exact figures live in
`docs/bench/`.

One attribution the table needs: answer generation ran on **Claude Opus 5** (`.env`-configured,
swappable without a code change). That matters for reading three rows — the `ask_full` 1.05x
ratio holds because ~15 s of model latency swamps retrieval contention, and citation accuracy is
a property of model + prompt + guardrails, not of the pipeline alone. Recall, accept latency,
throughput, and the kill test involve no LLM and are model-independent.

---

## 2. Managed vs self-hosted queues

The architecture deck picked Prefect, and the reasons matter. Redis was rejected as "not built
for multi-step workflows... higher cost," a custom queue as "high engineering overhead," Temporal
as "costly and complex to operate... locked by cost" (p.11). Every rejection is a cost or
ops-burden argument; none says Prefect retries better or scales further. Prefect won because a
team with no platform engineer can run it — Glimpse's reason too: "because the client has no
platform team, not because it is impressive."

We kept that choice and paid for it twice.

**Poll latency.** Between "the row is `pending`" and "a worker is executing" sits our dispatch
tick (`DISPATCH_INTERVAL_S = 3.0`) plus Prefect's own long-poll. It never touches accept latency —
accept p95 held at 196.6 ms even *during* the 14-document backfill — but it is dead air inside the
178.5 s wall clock that produced 1.33 pages/s. Glimpse measured the same thing: "10s of dead air
per ingest from Prefect's default poll frequency."

**No DLQ topic.** Prefect's dead-letter equivalent is a failed-state badge in a dashboard, not a
queue you can drain — and the failure we care about most never reaches that badge: a `kill -9`
raises no exception, so Prefect marks the run `Crashed` and nothing reschedules it.

The compensation is cheap and deliberate: **Postgres is the source of truth, not the queue.**
Every row carries `status`, `attempts`, `updated_at`, and every stage transition bumps
`updated_at`. A reconciler thread inside the worker sweeps on `RECONCILE_INTERVAL_S`, finds rows
in an in-flight status untouched longer than `RECONCILE_STUCK_S`, and either returns them to
`pending` (under `RECONCILE_MAX_ATTEMPTS`) or marks them `failed` with
`error = 'reconciler: N attempts exhausted'`. That branch is a poor-man's DLQ and we label it as
one. Both are single atomic `UPDATE ... RETURNING` statements over disjoint `attempts` predicates,
so racing reconcilers cannot recover one row twice.

**One caveat on the latency number.** The `ask_full` series that passed at 1.05x is dominated by
~15 s of model latency, so its ratio is nearly insensitive to contention. The `ask_retr` proxy —
same questions, scoped to a nonexistent source id, so both query embeddings and both Qdrant
searches still run before the zero-citation early return — moved 882 ms -> 2,141 ms: 2.43x
over one during-backfill round of 14 samples. We report both. The read path is decoupled in
*architecture*, but on one laptop a query's embedding contends with a backfill's. That is a
capacity finding, and §7 fixes it.

---

## 3. Grounded citations

The rule is **locator or nothing**. Nothing renders unless it resolves to an exact place: `mm:ss`
for video, `p. N` for a paper, `slide N` for a deck. `_page_label` writes that label into the same
`timestamp` field video already used, so the LLM prompt, the UI card, and the fallback answer kept
working without knowing which corpus they were reading. In run `20260728B`, all **140 citations
across 14 answers carried a locator: 83 timestamps, 42 `p. N`, 15 `slide N`.** No bare sources.

Three layers enforce it:

1. **The abstain gate.** Before any LLM call, `ask()` compares each branch's *raw* cosine — not
   the RRF score, which is tiny by construction — against its own threshold: CLIP visual >= 0.2,
   bge text >= 0.35. If neither clears, the answer is a refusal and `llm_used` is false. Zero
   tokens on a question the corpus cannot answer.
2. **Regex-stripped out-of-range references.** Shown six numbered moments, a model still writes
   `[9]`. `_validate_citations` matches `\[(\d+(?:\s*,\s*\d+)*)\]`, keeps only integers in
   `1..n_frames`, and drops the bracket if nothing survives. The fix edits the answer text rather
   than asking the model to behave, so an invented reference cannot reach the page.
3. **The citation-accuracy checklist as a metric.** `run_eval.py` emits
   `citation_check_<run>.md`: every answer, every citation, its locator, and the rule that a
   merely topical citation scores against you. Target >= 0.90 — a metric Glimpse lacked; they
   asserted faithfulness architecturally and never scored it. **Scored: 40/43 (0.93)** (3 ✗ among
   the claim-bearing citations; of the 45 attributed, 2 were visual claims a transcript
   cannot confirm and are excluded from the denominator; the 95 citations carrying no `[n]`
   attribution are context, not claims, and sit outside it entirely). AI-assisted first pass,
   pending a human spot-check.

4. **The audit paid for itself.** Both wrong-locator findings traced to one root cause: a text
   chunk that straddles a page boundary was cited to the page it *starts* on, so a quote living
   on the chunk's last page pointed one page short. The chunk always knew its span; the label now
   says `pp. 9–10` (commit `96445a5`). Second full audit loop of the evening — the eval found the
   flooding bug, the citation audit found the locator bug, and both fixes shipped with tests
   before submission.

---

## 4. Resilience: SIGKILL is not an exception

A crashed worker does not raise, does not run a `finally`, never sets a status. Recovery built on
exception handling recovers nothing — and idempotency alone does not save you: Glimpse's first
resilience run scored **0 of 4** with correct idempotent writes, because nothing re-ran the work.
Recovery has to be **state-driven**:

- **Idempotent writes.** Point ids are `uuid5(f"{doc_id}:{n}")` for pages and
  `uuid5(f"{doc_id}:text:{n}")` for chunks, so re-ingesting overwrites the same points. Duplicates
  are impossible by construction, not by a dedup pass.
- **Liveness, not a heartbeat service.** `set_status`, `set_progress`, `bump_attempts`, and
  `wfq_claim` all touch `updated_at`. That column is the lease: "in-flight and untouched longer
  than the stuck window" is a complete definition of orphaned.
- **Something that re-triggers.** The reconciler is the only component that turns a dead row back
  into runnable work.

The test: register 14 PDFs, `kill -9` the worker mid-ingest, hold it down 20 s, restart, wait
(`reconcile_stuck_s = 60`, `reconcile_interval_s = 10`, `max_attempts = 3`). From
`kill_worker_20260728_220849.json`: **14/14 `indexed`, 0 lost, 0 failed, 0 non-terminal, 218.9 s
elapsed.** Two rows were in flight at the kill — `doc_eb02cb16e7` and `doc_6acf75038f`, both at
`status_at_kill: "embedding"` — and both finished with **`attempts: 2`**. The other twelve show
`attempts: 1`: still `pending`, never claimed. Exactly the two mid-flight rows re-ran, once each.

---

## 5. The eval-driven fix

The first full eval, `eval_20260728A.json`, **failed**: recall **11/14 (0.79)** — video-only 5/5,
doc-only 5/5, **mixed-corpus 1/4**. The system was excellent at every question answerable from one
corpus and bad at the one thing it exists for.

`found_sources` named the cause. On `m3`, nine of ten citations came from four papers, five from
one alone (`doc_5f91a5ea20`, the original RAG paper, at pp. 1, 8, 9, 18, 19). On `m4`, ten of ten
were video. A prolific source — a long video with many strong windows, a big paper with many
matching pages — flooded the ranking and squeezed the other corpus out. Fusion worked; the top-K
did not.

The fix is nine lines in `_fuse`: **`MAX_PER_SOURCE = 3`**, a diversity cap applied after sorting.
Each source keeps its best three moments in rank order; the rest are **demoted, not dropped**
(`return head + tail`), so a single-source corpus still fills the list instead of returning four
results. Re-run as `eval_20260728B.json`: recall **12/14 (0.86)**, mixed 1/4 -> 2/4, video and
doc unchanged at 1.00. `m2` flipped to a hit — the *Attention* paper's multi-head section now
ranks alongside both expected videos.

The two remaining misses, analysed rather than argued away:

- **`m3`** now retrieves *both* expected documents but not the expected video
  (`yt_zjkBMFhNj_g`, Karpathy's tool-use segment). A video citation did return — `yt_wjZofJX0v4M`
  at 17:41 — but the strict rule demands an *expected* video, so it scores 0. Four document
  sources at three moments each fill nine of ten slots. The cap bounds one source; it does not
  balance corpora. A per-*corpus* quota would fix this query and would also force video into
  answers that are genuinely paper-shaped. Not built.
- **`m4`** answers almost entirely from video and misses *Attention is All You Need* p. 5 §3.4 —
  two sentences, one short chunk, low lexical overlap with a question phrased in the videos'
  vocabulary, while the same content is a vivid animated frame CLIP scores highly. Small true
  chunks lose to large vivid ones. Chunk-length normalisation or a sparse branch is the real fix;
  neither is a one-line change.

Recall went 0.79 -> 0.86 because an eval existed. That is the argument for building one.

---

## 6. Scoping as an FDE skill: PDF-only, no `.pptx`

We took `.pptx` off the table on purpose, and we declare it rather than hope nobody notices.
Rendering native PowerPoint faithfully means LibreOffice headless in the worker image. The bill:

- **Image size.** Roughly 500 MB+ on an image that today needs only the PyMuPDF wheel — one
  dependency that already does both jobs we need, text extraction and page rendering.
- **Cold-start conversion latency.** LibreOffice's first invocation in a fresh container costs
  seconds per file before a page is rendered. That falls inside the window `bench_backfill.py`
  measures, so **1.33 pages/s** would have described LibreOffice's startup, not our pipeline. A
  benchmark you cannot interpret is worse than one you never ran.
- **Fidelity risk.** Missing fonts, SmartArt, and embedded charts degrade silently — not a crash,
  a slide image that stops matching the query it should answer, with nothing in the status field
  to say so. Glimpse hit the same wall from the other side: "PPTX image slides aren't captioned.
  `python-pptx` can't rasterize; PDF decks are fine."
- **New failure surface on small workers.** A 2 GB worker VM already holds CLIP and bge. A
  subprocess with unbounded memory behaviour on adversarial input adds a new OOM class inside the
  component whose durability we are trying to prove.

Against all that: **every deck tool exports PDF in one click.** The user cost is one menu item;
the engineering cost is an image, a benchmark, a fidelity risk, and an OOM mode. Nine of our 14
corpus documents are decks, seven exported from `.pptx` — the filenames still say so — and all
seven ingested with no code beyond the PDF path.

The fast-follow is **sketched, not built**: LibreOffice as a sidecar with its own image and memory
limit, exposing `convert(pptx) -> pdf`, called by `t_fetch` before parsing. It keeps the worker
image small, isolates the OOM, and makes conversion latency separately measurable. Naming it is
scoping. Building it today would have cost us the benchmark.

---

## 7. Demo -> production, through cost per document

| Now | For a real backfill | Why |
|---|---|---|
| One worker, `DISPATCH_MAX_INFLIGHT = 2` | N workers, autoscaled on queue depth | 1.33 pages/s is one machine's number. Throughput scales with workers only if a model lock does not serialise them — Glimpse's "one lock guarded two models" is the first bug to test for. |
| CLIP on CPU, sharing a box with the API | GPU CLIP service, one warm model | Page embedding dominates the 178.5 s wall clock and runs on the same CPU the query path embeds on — which is why `ask_retr` moved 2.43x under load. Moving CLIP off the query box fixes contention and throughput at once. |
| Single Qdrant collection, int8 + HNSW on disk | Shard and replicate; keep the `user_id` filter | The multi-tenant filter design holds. Only capacity changes. |
| Reconciler at 60 s / 600 s / 3 attempts | Tune the stuck window to the longest healthy gap between `updated_at` bumps | Too short and it requeues work that is merely slow — the bug that cost Glimpse throughput. |
| Cost per document: designed, not measured | Instrument it | Below. |

**Deploy day proved the table's point — twice.** Shipping to Fly surfaced two findings the local
benchmarks could not. First, tenancy: `X-User-Id` is a convenience header, not authentication,
and the read path has no bearer check — fine on a laptop, unacceptable on a public URL sharing
one Qdrant/Postgres with a private corpus, where anyone who guessed the tenant name could read
the whole private index (verified live: 18 rows exposed before the fix). A `LOCK_TENANT` flag
now pins every public request to the demo tenant and ignores the header; the private corpus
reads back zero rows. Second, sizing: the 2 GB clip machine was OOM-killed mid-backfill — it was
sized for one CLIP model before documents made the service bimodal (the bge text embedder lives
there too), and a 128-page image batch had no headroom. 4 GB plus a capped `CLIP_BATCH=32`
indexed all five papers, the 48-page CLIP paper included. Neither bug was visible until the
system left the laptop — which is this section's whole argument.

**The honest gap.** Cost per document sits in our SLO table with no number against it. We designed
the metric before learning our architecture makes half of it zero: document ingest is
`fetch -> parse -> embed_index` with **no LLM captioning step**, and both embedders (CLIP
ViT-B/32, bge-small via fastembed) run locally on CPU. Ingest API spend is $0; the real ingest
cost is machine-seconds — 178.5 s of one worker for 237 pages. Token spend lives entirely on the
read path, where one `ask` sends up to `TOP_K = 6` moments with images to a multimodal LLM and
takes 11–18 s (p50 11,232 ms, p95 18,505 ms in run B).

Measuring it needs three things we did not build: per-run token accounting written to the manifest
row, a worker-seconds counter per document (derivable from `updated_at`, not yet recorded), and a
price table mapping both to dollars. With those, every row above collapses to two numbers —
dollars per document ingested, dollars per question answered — instead of a latency argument. That
is what we would build next: it is the number the client actually signs.

The same instrumentation buys a second experiment for free. Because the harness isolates the
model-dependent metrics (citation accuracy, answer latency, the `ask_full` ratio) from the
model-independent ones, an A/B across answer models — Kimi K2 or GLM via their
Anthropic-compatible endpoints — is two env vars and an eval re-run against the same frozen
index. Same golden set, same index, different model: the comparison the cost table above would
price.
