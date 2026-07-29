# Brief — "Glimpse" reference solution (Assignment 3)

Source material: `Glimpse_Solution/Glimpse-Review-Deck_1.pdf` (13 slides) and
`Glimpse_Solution/glimpse-architecture.png`. Author: Imran Tauqir. Live at
`https://glimpse-lens.fly.dev`.

Purpose of this brief: understand Glimpse's choices as a benchmark. We are **not** copying it.
Slide numbers cited as `[s3]` etc.

---

## 1. Thesis and framing

> "Text RAG answers the question. Glimpse search shows the evidence. Neither alone earns trust."
> — `[s3]`

Every answer must return a grounded answer **plus** the clip, page, or slide that supports it,
each tied to an exact locator, and never shown without one. Their tagline: *"A citation is a
promise you can check."* `[s3]` One query, three source types (timestamp · page · slide).

The framing device for the whole deck is honesty about failure: the subtitle of the agenda is
"Four tasks, one measured claim, and the bugs I found on the way" `[s2]`, and the closing line
is "The most useful thing I built was the benchmark that told me I was wrong" `[s13]`.

---

## 2. Architecture

### Two paths that never share a request `[s4]`

**Write path (async, decoupled):**
`Sources (video · PDF papers · slide decks) → FastAPI (accept, register manifest, enqueue in ms)
→ Prefect workers (per-source parse → dedup → chunk → enrich → caption) → Embed (CLIP for images
+ text embedder) → Qdrant (unified index, scalar quantization int8/f32, HNSW, multi-tenant)
→ Knowledge Graph (entities + relations, GraphRAG)` — from the PNG.

Slide 4's compressed version: `API (row + schedule) → 202 in ~171ms (no parsing here) →
Prefect (paper · deck · video) → Worker (parse→enrich→chunk→embed)`.

**Read path (stays fast during backfill):**
`Query (natural language) → API (rate limit, fair queue) → Semantic cache (hit → instant answer)
→ Retrieve (hybrid dense + sparse, rescore f32) → Rank (cross-encoder rerank · RRF · cross-modal
fusion) → Reason (agentic plan / reflect / graph hops) → Multimodal LLM (frames + text)
→ Answer (grounded · citations · clips)` — from the PNG.

Slide 4's version: `Query (rate limit · tenant) → Two branches (CLIP frames · bge text)
→ Fuse + gate (RRF · abstain free) → Cited answer (locator or nothing)`.

### Stores and infra `[s4]` + PNG

| Concern | Choice |
| --- | --- |
| Vectors | Qdrant — int8 in RAM, f32 on disk, HNSW, multi-tenant |
| Manifest / status / audit | Postgres (Neon) |
| Queue | Prefect Cloud |
| Media / object storage | Tigris (PNG also lists GCS / S3) |
| Cache + limits + counters | Upstash / Redis (semantic cache) |
| Knowledge graph | Neo4j (links) |
| Compute | Autoscaling workers on Fly.io |

Everything stateful is a managed service, so every machine is disposable. **One Docker image,
three process groups: `api`, `worker`, `clip`.** `[s4]`

Cross-cutting concerns called out in the PNG: observability (metrics · tracing · SLOs),
guardrails (validation · prompt-injection · grounding · tenant isolation), eval + feedback
(golden set · thumbs · clicks → ranking), token & cost (per-tenant budgets · caps).

Note: the PNG is more ambitious than the deck. Agentic reasoning, GraphRAG, semantic cache, and
cross-encoder rerank appear in the diagram but are never demonstrated or benchmarked in the 13
slides. Treat the PNG as target-state, the deck as what was actually measured.

---

## 3. Task 3.1 — Multi-source ingestion and index design `[s5]`, `[s6]`

Headline: **"Papers and decks ride the text branch transcripts already used — reuse, not a
parallel pipeline."** `[s5]`

- **Only the head differs.** PDF → pages, deck → slides; the `enrich → embed → upsert` tail is
  shared and unchanged.
- **Locator rides into the payload.** `page` · `slide` · `start_ms` — the field the UI
  deep-links to.
- **One text collection, `kind` in the payload.** A single query fuses video transcripts, paper
  pages, and slides. Explicitly rejected: a second collection for documents `[s11]`.
- **`video_id = source_id`** so the existing fusion logic and Postgres joins work untouched. A
  deliberate "don't rename the primary key" call to avoid touching the read path.
- **Documents bucket by locator, not by time.** All doc chunks share `t=0`, so applying the
  video time-windowing logic to a paper would collapse the entire paper into one citation.

### The enrich bug `[s6]`

Before: `if text:` — pages with no extractable text were silently dropped, so figures, charts,
and scans never entered the index. Nothing in the status field said so. Decks captioned their
slides; papers captioned nothing.

After: **parse renders those pages** (local, cheap), and a **separate `enrich` task captions
them** (network, flaky, one LLM call per caption, concurrent and capped). One implementation
covering both source types.

The design payoff they highlight: a rate-limited caption retries *just the captions* — it does
not re-download and re-parse the PDF. "That split is the entire argument for per-stage tasks."

---

## 4. Task 3.2 — Async work queue design `[s7]`

Headline: **"The API writes a row and schedules a run. Everything else is the worker's problem."**

- `POST /admin/documents` returns `202 {id, status, kind}` before anything is parsed.
- Two flows beside video: `ms-ingest-paper` and `ms-ingest-deck`; **one worker serves all three.**
- **Per-stage tasks, per-stage retries** — a failed stage retries without redoing finished ones.
- Lifecycle mirrors video: `pending → queued → parsing → enriching → chunking → embedding →
  indexed`.
- **Outbound HTTPS only.** The worker long-polls Prefect Cloud; no inbound ports anywhere. This
  is a security-posture argument as much as an architecture one.

---

## 5. Task 3.3 — Benchmark: methodology and numbers `[s8]`–`[s10]`

Methodology: a `bench.py` harness that **exits non-zero if any threshold misses** — "the gate is
the grader, not my judgement." `[s8]` Five SLAs, each with an explicit target.

| Metric | Target | Measured | Verdict |
| --- | --- | --- | --- |
| Accept p95 | ≤ 300 ms | **171 ms** | PASS |
| Retrieval p95 during backfill | ≤ 1.3× idle | **1.00× — 519 / 518 ms** | PASS |
| Recall@10 | ≥ 0.70 | **1.00 (3/3)** | PASS |
| No-loss under worker crash | 100% | **4/4 recovered** | PASS |
| Ingest throughput | ≥ 8 chunks/s | **4.0 chunks/s** | RE-MEASURING |

`[s8]`

### The misdiagnosis `[s9]`

He first wrote "infra-bound, needs a GPU." Evidence: scaling 1 → 3 workers moved throughput only
4.2 → 4.6 chunks/s. Correct observation, wrong inference. Real causes:

1. **One lock guarded two models.** CLIP is not thread-safe; fastembed is. Sharing a single lock
   queued every paper batch behind every frame batch.
2. **The reconciler re-did queued work.** It re-enqueued rows that were merely *waiting* for a
   worker — under backfill, most of them.
3. **10s of dead air per ingest** from Prefect's default poll frequency.
4. **Four Qdrant round-trips per document** re-asserting a collection that already existed.

This slide is the single most transferable artifact in the deck — it is a debugging pattern, not
a result.

### Worker-kill resilience `[s10]`

Headline: **"Crash-safe code isn't a no-loss guarantee until something re-triggers the work."**

- First `--resilience` run: **0 of 4.** A SIGKILL is not an exception — Prefect marks the run
  `Crashed` and never reschedules it.
- **Idempotency was never the problem.** Deterministic point IDs and commit-then-complete made
  re-running safe; nothing *re-ran* it.
- Fix: a **reconciler** — an always-up sweep that re-enqueues stranded sources. Second run: **4 of 4.**
- Then capped it: a document that *kills* its worker never gets marked failed, so it came back
  forever. **Dead-letter after N attempts.**

---

## 6. Task 3.4 — Trade-off arguments `[s11]`

Headline: **"Prefect Cloud because the client has no platform team — not because it is impressive."**

| Chosen | Deliberately not |
| --- | --- |
| Managed queue — less ops, faster to prod | Self-hosted broker — control we can't staff |
| Workers: outbound HTTPS, no inbound ports | Header-derived tenancy — caller-controlled |
| Tenant derived from the API key, never a header | A second collection for documents |
| int8 in RAM, f32 rescore on the shortlist | LibreOffice in the image for one format |

Closing argument: *"Choose what survives in the client's environment — their infra, budget and
team — not what demos best."* Every trade-off is framed against a hypothetical client's staffing,
not against benchmark scores. That framing is what makes it read as FDE work.

---

## 7. How it handles the hard problems

**Fusing time-based video hits with page-based document hits.** Handled almost entirely at
*index* time rather than rank time: one collection, `kind` in the payload, `video_id = source_id`,
and locator fields (`page`/`slide`/`start_ms`) carried in the payload `[s5]`. At rank time it is
two branches (CLIP frames, bge text) fused with **RRF** plus a **gate** that can abstain `[s4]`.
Documents are deliberately excluded from the time-windowing that video hits get, because all doc
chunks share `t=0` and would collapse a whole paper into one citation `[s5]`.

**Citation grounding.** "Locator or nothing" `[s4]`. Evidence is never rendered without an exact
locator; the UI deep-links to the timestamp, page, or slide. The abstain path is free — no hit
above threshold means no answer rather than a hallucinated one.

**Worker resilience.** Three layers: deterministic point IDs + commit-then-complete (safe to
re-run), a reconciler sweep (something *does* re-run it), and a dead-letter cap after N attempts
(a poison document can't loop forever) `[s10]`.

---

## 8. Weaknesses, gaps, and where we can do better

Their own stated gaps `[s12]`:

1. **Throughput never re-measured.** Root-caused and fixed, but unverified until the benchmark
   runs at ≥2 workers. The one FAIL in the table is still a FAIL.
2. **Golden set is two papers, Recall@10 = 3/3.** Recall 1.00 on a 3-query golden set is not a
   result. Never tested on a mixed corpus — the exact case the system exists for.
3. **No graceful SIGTERM.** No signal handling anywhere; a killed run is recovered by the
   reconciler, not drained.
4. **PPTX image slides aren't captioned.** `python-pptx` can't rasterize; PDF decks are fine.
5. **Ranking bias compensated, not solved.** A document can never earn the cross-modal boost;
   coverage is reserved for documents instead of the scores being made comparable.

Additional gaps we can exploit:

6. **The PNG oversells the deck.** Semantic cache, GraphRAG / knowledge graph, agentic
   plan-reflect, and cross-encoder rerank appear in the architecture diagram but are never
   measured, demoed, or discussed in the 13 slides. If we build any of these, we should measure
   them or leave them out of the diagram.
7. **No cost or token numbers.** "Token & cost — per-tenant budgets · caps" is a box in the PNG.
   Each figure caption is an LLM call `[s6]`; nobody says what a 40-page paper costs to ingest.
   A cost-per-document number would be a differentiator.
8. **No mixed-corpus retrieval evidence.** The central claim — one query, three source types — is
   never demonstrated with a query that returns a video hit *and* a page hit *and* a slide hit
   ranked together. A single screenshot of that would be worth more than the SLA table.
9. **Recall is the only quality metric.** No precision, no MRR/nDCG, no citation-accuracy check
   (does the cited page actually support the claim?). Faithfulness is asserted architecturally
   ("locator or nothing") but never scored.
10. **Reconciler is a polling sweep, not an event.** It works, but it re-enqueued waiting rows
    and cost throughput `[s9]`. A lease/heartbeat model (visibility timeout) would be cleaner and
    is a defensible alternative in our 3.4 write-up.
11. **Ingest is single-tier.** No priority lanes — a 200-page backfill and a user's one-off upload
    share the same queue. Fair queueing is claimed on the read path only.

---

## 9. Deck structure — use as a template

Thirteen slides, dark theme, one idea per slide, big left-aligned headline written as a full
sentence with a claim in it (not a noun phrase). Section kickers in orange monospace.

| # | Slide | Function |
| --- | --- | --- |
| 1 | Title — "Glimpse", one-line what-it-does, name, live URL | Product framing, not assignment framing |
| 2 | Agenda — "Four tasks, one measured claim, and the bugs I found on the way", six numbered items keyed to 01 / 3.1 / 3.2 / 3.3 / 3.4 / 05 | Maps directly onto the rubric |
| 3 | Thesis — why evidence + answer together | The one memorable sentence |
| 4 | Architecture — write row / read row / rented-state row, two numbers highlighted in orange | Whole system on one slide |
| 5 | Task 3.1 — ingestion & index design, 5 bullets | Rubric part |
| 6 | Task 3.1 — the enrich bug, WAS / NOW two-column | Depth, shows debugging |
| 7 | Task 3.2 — queue design, 5 bullets | Rubric part |
| 8 | Task 3.3 — SLA table: metric / target / measured / verdict | The proof |
| 9 | Task 3.3 — "the one I got wrong", pull-quote + 4 root causes | Credibility through admission |
| 10 | Task 3.3 — resilience narrative, 0/4 → 4/4 → capped | Story arc, not a status |
| 11 | Task 3.4 — CHOSEN / DELIBERATELY NOT two-column + client-context closer | Rubric part |
| 12 | What is not built — 5 self-declared gaps | Pre-empts the reviewer |
| 13 | So what — 4 takeaways + closing pull-quote | Lands the thesis |

Patterns worth stealing:
- **Every headline is a claim**, e.g. "Crash-safe code isn't a no-loss guarantee until something
  re-triggers the work" — not "Resilience".
- **Numbers highlighted in accent color inside the architecture diagram** so the measurement and
  the architecture are the same picture.
- **A dedicated "what I got wrong" slide** and a dedicated "what is not built" slide. Two of
  thirteen slides are admissions, and they raise credibility rather than lowering it.
- **The failing SLA stays in the table**, labeled RE-MEASURING rather than removed.
- **Pull-quotes with a left accent rule** used exactly three times (s3, s9, s13) for the lines he
  wants remembered.
- Slide 4 doubles as the architecture slide *and* the tech-stack slide by adding a "rented state"
  row — avoids a separate boring stack slide.
