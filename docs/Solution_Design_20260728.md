# Assignment 3 — Solution Design (DRAFT for Kurt's review)

Date: 2026-07-28 · Status: **APPROVED by Kurt 2026-07-28** — all 4 decisions signed off; PDF-only confirmed emphatically (**no .pptx**)
Sources: `Repo_Recon_20260728.md`, `Brief_Class_Deck.md`, `Brief_Architecture_Considerations.md`, `Brief_Glimpse_Solution.md`

---

## Design philosophy

Reuse the seams the repo already gives us (Glimpse validated this works), and spend our
differentiation budget where Glimpse is weak: **mixed-corpus retrieval quality, a real
eval set, and cost/throughput honesty.** The decks give no SLO numbers — every target
below is ours, chosen to be defensible, and we say so in the write-up (FDE virtue:
define your own SLOs when the customer hasn't).

## 3.1 — Multi-source ingestion

**Identity & routing**
- New id prefix `doc_<10-hex>`; `source` column gets values `paper` | `deck` (free-text
  column, no migration needed). User picks paper vs deck in the UI at upload.
- `POST /api/videos` gains a third branch keyed on content-type `application/pdf`;
  presign gate `ALLOWED_UPLOAD_TYPES` extended with `application/pdf`.

**Parsing (new module `src/ingest/documents.py`)**
- **PyMuPDF (fitz)** for both text extraction and page→JPEG rendering. Chose PyMuPDF
  because one dependency covers both jobs, it's fast pure-C, and needs no system
  packages beyond the wheel.
- Papers: page-aware text chunks (~400 tokens, 15% overlap), locator `{page_start, page_end}`.
- Decks: one chunk per slide (title + body text), locator `{slide}`.
- Both: render every page/slide to JPEG at THUMB_WIDTH, reusing the existing
  `frames/{user}/{id}/NNNNNN.jpg` key layout so thumbnails and citation images work unchanged.

**Indexing — one searchable space, two existing collections, zero new collections**
- Doc text chunks → existing `moments_text` collection, payload `modality: "doc_text"`,
  plus `kind: paper|deck`, `page`/`slide` locator. (Glimpse's validated move.)
- Page/slide JPEGs → CLIP-embedded into the existing `moments` collection,
  `modality: "page"`. This is **our differentiator #1**: decks and figure-heavy papers
  become *visually* searchable ("the slide with the architecture diagram"), which
  Glimpse never demonstrated. Same 512-D CLIP space, so query embedding is unchanged.
- Point ids stay `uuid5(f"{id}:{n}")` / `uuid5(f"{id}:text:{n}")` → idempotent re-ingest for free.
- We trust the repo (512-D clip-ViT-B-32) over the decks' "800D" label and footnote the
  discrepancy in the write-up.

**Retrieval & fusion changes (`_fuse`)**
- Video hits: time-window grouping, unchanged.
- Doc hits: group by `(doc_id, page/slide)` instead of time — a page IS the moment.
- Same RRF scoring and cross-modal boost apply: a page that matches both visually and
  textually gets the 1.5× boost, symmetrical with video.
- Citations: `page X` / `slide Y` label instead of `mm:ss`; deeplink = presigned page
  thumbnail + doc download anchor. Rule stays **locator-or-nothing**.

## 3.2 — Async queue

- Second flow `ms-ingest-doc` (fetch → parse → embed_index), second deployment; Prefect 3's
  module-level `serve(d1, d2, limit=N)` runs both deployments in the one worker process.
- **Same status lifecycle strings** (`pending → queued → fetching → sampling → embedding
  → indexed`) — "sampling" = parsing for docs. Parity is an explicit 3.2 requirement, and
  reuse means the UI status chips and WFQ dispatcher work with zero changes.
- Retries mirror video's per-task retry pattern (fetch ×2 backoff, embed ×2).
- WFQ fair dispatcher is source-agnostic (claims `pending` rows) — enqueue just picks the
  deployment by id prefix.
- **New: a reconciler** (differentiator #2, and Glimpse's hardest-won lesson): a periodic
  task that resets rows stuck in an inflight status beyond a timeout back to `pending`,
  with an attempts cap → `failed` (poor-man's DLQ, honestly labeled as such — the
  architecture deck admits Prefect has no real DLQ topic).

## 3.3 — Benchmark plan (all targets ours to define; Glimpse comparison in parens)

| Metric | Method | Target |
|---|---|---|
| Accept latency p95 | 200× `POST /api/videos` (doc registration), measure 202 round-trip | < 250 ms (Glimpse: 171 ms) |
| Search latency under backfill | `/api/ask` p95 before vs during a ~50-doc backfill | ≤ 1.25× baseline (Glimpse: 1.00×) |
| Ingest throughput | chunks indexed / sec during backfill, measured not promised | report honestly; stretch > 4.0 chunks/s (Glimpse's actual) |
| Recall@10 | golden set of **12–15 queries incl. ≥4 mixed-corpus** (video+doc answers) | ≥ 0.85 (Glimpse: 1.00 on only 3 queries — we beat them on rigor, maybe not on the number) |
| **Citation accuracy** | human-check every citation locator in eval answers | ≥ 0.9 — metric Glimpse didn't have |
| Durability | `kill -9` worker mid-ingest ×4; reconciler recovers; assert final indexed count + no duplicate Qdrant points | 4/4, zero loss, zero dupes |
| **Cost per document** | LLM + embed API spend / docs ingested | report — metric Glimpse didn't have |

Harness: `scripts/bench_accept.py`, `scripts/bench_backfill.py`, `scripts/kill_worker_test.sh`,
`eval/golden_set.json` — all new, repo has zero tests today.

## 3.4 — Write-up spine

1. Managed vs self-hosted: Prefect won on **cost + ops burden, not capability** (deck's own
   argument); we felt the flip side (poll latency, no DLQ) and mitigated with the reconciler.
2. Grounded citations: locator-or-nothing, regex-validated `[n]`, citation-accuracy metric.
3. Resilience: idempotent uuid5 upserts + Postgres manifest as source of truth +
   reconciler; SIGKILL is not an exception — recovery must be state-driven.
4. Demo→production: what we'd change for a real backfill (autoscaling workers, GPU CLIP,
   Qdrant sharding) with cost-per-document as the deciding lens.
5. **Scoping as an FDE skill (Kurt's explicit requirement):** declare the PDF-only decision
   and the .pptx baggage we refused — native slide rendering requires LibreOffice headless:
   ~500MB+ added to the image, **cold-start conversion latency of seconds per file** (which
   would pollute the 3.3 throughput benchmark), rendering-fidelity risk (fonts/SmartArt),
   and a new failure surface on 2GB worker VMs — versus a one-click PDF export from every
   deck tool. Documented fast-follow: LibreOffice sidecar design, sketched not built.

## Decisions — APPROVED by Kurt, 2026-07-28

1. ✅ **CLIP-embed doc pages into the visual index** (differentiator vs Glimpse).
2. ✅ **PDF-only uploads, NO .pptx** — and the 3.4 write-up MUST declare the LibreOffice
   baggage (image size, cold-start latency, fidelity risk) as the rationale (see spine §5).
3. ✅ **SLO targets table** as written.
4. ✅ **Eval set 12–15 queries**, ≥4 mixed-corpus.

## Build order (once .env is live)

1. Chunk 1: stock app up locally, seeded video search verified.
2. Chunk 3a: `documents.py` parser + unit-testable chunker (no services needed — can build BEFORE .env).
3. Chunk 3b: routing, flow, indexing, fusion, citations, UI tab.
4. Chunk 4: second deployment, reconciler, retries.
5. Chunk 5: harness + eval set + benchmark runs.
6. Chunk 6: Fly deploy + write-up. 7. Chunk 7: diagrams/slides, push to lozierk.
