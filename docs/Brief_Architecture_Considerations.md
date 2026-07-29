# Brief — Architecture Considerations (MomentSearch)

Distilled from `Architecture_Considerations_Moment_Search.pdf` (15 slides, image-only PDF — no
text layer; all content read from rendered pages). Page numbers below are PDF page numbers.

**Core argument in one line:** the demo failed because one synchronous process did upload,
decode, embed, and search; the fix is a message queue that splits *intake* from *execution*, so
workers scale, retry, and crash independently of the query path — plus quantized vector storage
so the index fits in RAM at a fraction of the cost.

---

## 1. The problem statement the deck sets up

**The product** (p.2): "Ask Lenny's Podcast a hard question — get the exact moment." A digital
twin over **10 podcast episodes**. Output contract: *a cited answer, synthesized across episodes,
streamed live; clicking any citation pops the source video at that timestamp with a synced
transcript.* Citation grounding is stated as a product feature, not an afterthought — every claim
resolves to a playable timestamp.

**The demo architecture that couldn't scale** (p.3) — a 10-stage monolith, each stage named as a
specific failure:

| # | Stage | Failure it caused |
|---|---|---|
| 1 | Browser uploads video **through** the app | gigabytes into app memory |
| 2 | App writes video to local disk | ephemeral, node-bound state |
| 3 | Extract **every** frame to disk, then reopen each | disk + IO thrash |
| 4 | No dedup — keep all frames | duplicate vectors |
| 5 | Load CLIP model in-process | **~15–30 s cold load on every run** |
| 6 | Embed all frames synchronously | blocking, no queue, **no retries** |
| 7 | Store all float32 vectors in RAM | memory pressure |
| 8 | Query = brute-force scan over all vectors in RAM | latency scales linearly with corpus |
| 9 | Return matches | same moment repeated many times |
| 10 | Send to LLM | — |

Summary line on p.3: "memory pressure, duplicate work, slow startup, and no fault recovery."
These four are the scoring rubric for anything we build.

---

## 2. Principle 1 — The queue separates intake from execution

**Target flow** (p.4), five stages: (1) upload video → (2) **separate upload queue** → (3) frame +
transcript embeddings → (4) vector search over frames *and* transcripts → (5) matching moment +
scoring/ranking.

Two explicit sub-claims on p.4:
- **"A different queue."** Two lanes: an **Ingestion Lane (uploads)** and a **Query Lane
  (searches)**. Rationale: "uploads aren't queries — heavy media ingestion gets its own background
  lane" so it "doesn't block queries."
- **"Image and Text."** A result may be a timestamped clip, a transcript sentence, **or both** —
  the retrieval unit is deliberately heterogeneous, and results from different modalities are
  merged into one ranked list.

**The scalable stack** (p.5), left to right:
`User/Client → API Gateway (authenticate + route) → Rate Limiting (protect shared compute) →
Message Queue (buffer async work, decouple intake from processing) → Workers (pull jobs when
capacity is available, process in parallel) → Data & Storage (Video Store, Transcript Store,
Embedding Store)`.

The stated **core shift**: "Uploads no longer wait for processing. The queue separates intake from
execution, allowing workers to scale and recover independently." Four claimed properties:
*Resilient* (retries + fault recovery), *Scalable* (workers scale independently), *Efficient*
(better throughput and resource usage), *Protected* (secure, rate-limited).

**Before/after restatement** (p.10). Before: one process, synchronous, blocking, fragile —
"requests blocked until processing finished", "one failure could stop the whole pipeline." After:
independent worker pools for **frame extraction**, **embedding generation**, and **index updates**,
each scaled separately; "failed jobs are retried automatically."

**Why queues are essential for RAG specifically** (p.7) — four arguments, in the deck's order:
1. **Smooths out traffic spikes.** The queue is a buffer absorbing surges and feeding the system at
   a stable rate — the vector DB and LLM see constant load, not the user's burst.
2. **Asynchronous document ingestion.** "Instantly confirm uploads and offload the heavy data
   pipeline to background workers." The worker pipeline is drawn as **Parse → Chunk → Embed →
   Index** — note this is the *document* pipeline, not the video pipeline, and it is exactly the
   PDF/slide path Assignment 3.1 asks for.
3. **Guaranteed multi-step processing.** Retries plus a **dead-letter queue**: on failure at the
   embed step the job either retries automatically or lands in the DLQ; "data is never lost and
   failed chunks are re-processed automatically."
4. **Horizontal scaling of AI workers.** Worker 1..N pull from the same LLM/embedding queue; "spin
   up more worker nodes when queues back up. More workers consume tasks in parallel, keeping
   latency low." Queue depth is the implied autoscaling signal.

---

## 3. Principle 2 — Managed vs. self-hosted: the queue bake-off

**Platform decision** (p.11). Four options evaluated, one selected:

| Option | Why considered | Why rejected / selected |
|---|---|---|
| **Redis-based queue** | simple, fast, familiar | not built for multi-step workflows or reliability at scale; **higher cost** (managed service + persistence) |
| **Custom queue (build from scratch)** | full control, built for exact needs | high engineering overhead, hard to maintain and scale — **high ops burden** |
| **Temporal** | built for durable workflows, retries, visibility | costly and complex to operate for this use case — **"locked by cost"**; strong product, but not for us |
| **Prefect** ✅ | — | **selected**: easy to run, open source, powerful orchestration, cost-effective — **"low cost, high control"** |

The trade-off axis the deck argues on is *ops burden vs. cost vs. workflow durability*, not raw
throughput. Prefect wins because it is open-source (no per-workflow vendor cost) while still giving
durable multi-step semantics. This is the material for Assignment 3.4's "managed vs. self-hosted
queues" discussion — note the deck's own framing: Redis and Temporal were rejected primarily on
**cost/complexity**, not capability.

**Prefect ↔ RabbitMQ mapping** (p.8) — the translation table to use when reasoning about Prefect in
classical queue terms:

| RabbitMQ pattern | Prefect equivalent | How it works |
|---|---|---|
| Publishing a message | **Triggering a deployment** | app fires an API call to Prefect to start a pipeline run, passing the document URL or text as an argument |
| The message queue | **Work pool queue** | Prefect holds the scheduled run in a queue until a worker is ready to pull it |
| The worker process | **Prefect worker** | a lightweight Python process on your server polls the Prefect queue and executes the RAG code |
| Dead-letter queue (DLQ) | **Failed-state UI** | failed tasks are visibly flagged in the Prefect dashboard, where you trigger automated or manual retries |

Two consequences worth flagging for our build: (a) the "message" is *arguments to a deployment* —
so a PDF job is naturally `{document_url | text, doc_type, user_id}`; (b) Prefect's DLQ is a *UI
state*, not a separate durable queue, so any "nothing is lost" proof (3.3) must rest on Postgres
job status + Prefect's run persistence, not on a DLQ topic.

---

## 4. Principle 3 — Fair queueing, so one big backfill can't starve everyone

**Definition** (p.12): "A Fair Queueing (FQ) scheduling algorithm is a resource-allocation method
designed to divide system capacity equally and prevent single users or applications from
monopolizing data paths."

**Mechanism** (p.13): a **Fair Queue Dispatcher** sits between clients and workers. It tracks
requests per user, allocates turns fairly, and prevents any one user from dominating. Worked
example: User A has 10 requests, User B has 3, User C has 2 → dispatch order is round-robin
`A B C A B A C …` rather than FIFO. Stated outcome: "One heavy user cannot block others →
everyone makes progress."

This is directly load-bearing for the 3.3 benchmark: the "search stays fast during a big backfill"
claim depends on (a) separate ingestion/query lanes (p.4) and (b) per-user fair dispatch (p.13).
Fair queueing also appears inside the API on the production diagram (p.15, box 6: "rate limiting +
fair queue") — so it guards the *read* path too, not just ingestion.

**Worker definition** (p.14): a worker continuously loops **Fetch → Process → Complete** against
the job queue, producing processed documents, embeddings, updated index entries, and
transcriptions. Value claimed: background execution, parallelism, "improve reliability &
isolation", easy scale-out. The Fetch/Process/Complete loop is the idempotency seam — a crash
between Process and Complete is exactly the failure Assignment 3.3 asks us to survive.

---

## 5. Principle 4 — Index design: scalar quantization and the two-tier store

**Scalar quantization** (p.6) — the deck's only quantitative storage argument:

- **RAM tier (fast search):** `int8`, normalised vectors — **¼ the footprint of fp32** — used for
  fast ANN search.
- **Disk tier (exact accuracy):** full-precision (`float32`) vectors on disk; fetch **only the top
  candidates** for re-ranking.
- **Result: 4× more vectors per node.** Lower RAM usage, lower infrastructure cost.
- **Accuracy caveat (stated explicitly):** scalar quantization "is fundamentally a **lossy
  compression** technique. However, the accuracy loss is generally incredibly small (**often less
  than 1 %**) when converting from higher-precision formats (float32) to 8-bit integers (int8)."

Diagram flow (p.6): `Query → RAM tier (quantised vectors) → Top-K candidates → Disk tier
(full-precision) → Results`. That is a **two-phase retrieve-then-rescore** design, not a single
ANN lookup — the accuracy budget is spent in phase 2, the latency budget in phase 1.

---

## 6. Reference architecture — production at scale (p.15)

The one full-system diagram. Numbered 1–10, split into a write path and a read path, with a
"Data & Infrastructure" column on the right.

**INGEST (write path)**
1. **Browser** — user uploads video. A dashed arrow goes **directly to Object Storage** via
   **multipart upload** (GCS / S3 / Tigris) — the bytes *bypass the app*, fixing demo failure #1.
2. **API (FastAPI)** — two fan-out arrows only: `register manifest` → **Postgres (Neon)**
   (video manifest / status) and `enqueue job` → **Job Queue** (background jobs). The API never
   touches media.
3. **Ingest Worker (Background)** — three inner stages: `sample frames (~2 s)` → `pHash dedup` →
   `extract captions`. Sampling replaces "every frame"; pHash replaces "keep all frames".
4. **CLIP service** — "one warm model, **800D**", called by the worker. Then `upsert (int8 +
   payload)` into **Qdrant**.

   *(Note: "800D" is as printed on the slide. Standard CLIP ViT-B/32 is 512-D; treat the exact
   dimension as unverified and confirm against the repo before building on it.)*

Three green check callouts under the ingest lane (p.15):
- **one CLIP model, shared & warm (CPU→GPU swap)** — kills the 15–30 s cold load (p.3 failure #5)
- **~4× less RAM via int8**
- **dedup kills near-duplicate vectors**

**QUERY (read path)**
5. **Browser** — text query.
6. **API (FastAPI)** — carries two named concerns: **rate limiting + fair queue** and **time to
   first byte**. It issues `embed text` and `search (user_id filter)` against Qdrant.
7. **Rerank + Fusion** — **RRF (Reciprocal Rank Fusion) + cross-modal boost**, returning
   **top 6 moments**.
8. **Object Storage (thumbs / clips)** — fetch clipped frames.
9. **Multimodal LLM** (GPT-4o / vLLM / others) — receives the clipped frames.
10. **Browser** — results as **cited moments**: "cited answer + timestamps + thumbnails".

**Qdrant configuration (stated twice, p.15):** `int8 in RAM + float32 on disk · HNSW on disk ·
multi-tenant`, and the retrieval recipe: **HNSW on int8 (RAM) → rescore on float32 (disk) →
return 20 candidates**. So the pipeline is **20 candidates retrieved → fused/reranked → top 6
returned to the LLM**.

**Component inventory — "Data & Infrastructure" column (p.15):**
| Component | Role |
|---|---|
| Object Storage (GCS / S3 / Tigris) | videos + thumbnails |
| Postgres (Neon) | manifest + status |
| Qdrant | int8 in RAM + float32 on disk, HNSW on disk, multi-tenant |
| Job Queue | background jobs |
| Compute | containers / autoscaling workers |

Multi-tenancy is enforced at the query with a **`user_id` filter** on the Qdrant search — a single
collection, filtered, rather than a collection per tenant.

---

## 7. Concrete numbers (everything quantitative in the deck)

| Figure | Value | Page |
|---|---|---|
| CLIP cold load in the broken demo | **~15–30 s every run** | 3 |
| Frame sampling interval (production) | **~2 s** | 15 |
| int8 vs fp32 memory | **¼ footprint / ~4× more vectors per node** | 6, 15 |
| Quantization accuracy loss | **often < 1 %** | 6 |
| Candidates returned from Qdrant | **20** | 15 |
| Moments returned after rerank/fusion | **top 6** | 15 |
| Embedding dimension (CLIP service) | **800D** *(as printed; verify)* | 15 |
| Corpus in the reference demo | **10 podcast episodes** | 2 |
| Fair-queue worked example | 10 / 3 / 2 requests → round-robin A B C A B A C | 13 |

**What is *not* in the deck — gaps we must fill ourselves:** no latency SLOs (accept latency,
p95 search, time-to-first-byte targets are *named as concerns* on p.15 but never given values), no
throughput targets (docs/min, frames/s), no autoscaling thresholds (queue-depth trigger, worker
counts), no dollar cost figures, no recall/nDCG targets, no retry counts or backoff schedule, no
visibility-timeout or lease duration. Every number in Assignment 3.3's benchmark table will be
ours to define and defend.

---

## 8. Failure modes and resilience guidance

Explicit in the deck:
- **Retries + dead-letter queue** for multi-step processing; failed chunks are re-processed
  automatically and "data is never lost" (p.7).
- **Automatic retry of failed jobs** as a property of the queue-based design (p.10).
- **Fault isolation:** "one failure could stop the whole pipeline" is listed as a *before* problem;
  after, workers fail independently (p.10).
- **Worker isolation** — "improve reliability & isolation" as a stated worker benefit (p.14).
- **Failed-state UI** as Prefect's DLQ analogue, supporting automated *or manual* retries (p.8).
- **Rate limiting** at the API gateway to "protect shared compute" (p.5, p.15).
- **Fair queueing** so one heavy tenant cannot starve others (p.12–13).

Implicit but unstated — the deck asserts durability without specifying the mechanism. To actually
prove "nothing is lost" we must supply what the deck omits:
- **Idempotency keys.** The worker loop is Fetch → Process → **Complete** (p.14). A crash after
  Process but before Complete means at-least-once delivery and a re-run. Qdrant `upsert` with a
  **deterministic point ID** (e.g. hash of `doc_id + chunk_index`) makes replay safe; pHash dedup
  (p.15) is a second, content-level guard against duplicate vectors.
- **Status lifecycle in Postgres.** The manifest/status table (p.15) is the durable record of
  truth, not the queue. Per-stage status (`queued → parsing → embedding → indexed → failed`) is
  what a kill-a-worker test can be measured against.
- **No claim is made about exactly-once.** Design for at-least-once + idempotent writes.

---

## 9. What this directly implies for Assignment 3

**3.1 — PDF / slide ingestion into one index.** The deck already draws the document pipeline:
**Parse → Chunk → Embed → Index**, fed by an ingestion queue with an instant upload confirmation
(p.7). Mirror the video worker's shape (p.15): bytes go **straight to object storage via multipart,
never through the API**; the API only *registers a manifest row* and *enqueues a job*. The video
worker's `sample → dedup → extract captions` maps cleanly onto a doc worker's
`extract pages → dedup near-identical slides (pHash on rendered page images) → extract text +
captions`. Slides are visual — the pHash dedup step is as relevant to deck ingestion as to video.

**Unified indexing.** One Qdrant collection, `int8` in RAM + `float32` on disk, HNSW on disk,
multi-tenant with a `user_id` filter (p.15). The deck's "Image and Text" claim (p.4) — a result can
be a clip, a sentence, or both — is the precedent for mixing modalities in one space. Cross-source
ranking is handled *after* retrieval by **RRF + cross-modal boost** (p.15), not by forcing all
sources into one embedding model — which is the escape hatch that lets CLIP-embedded frames/slide
images and bge-embedded text coexist and still be ranked together.

**Citation grounding.** The contract is set on p.2 and p.15: every answer is a *cited answer +
timestamps + thumbnails*, and clicking a citation opens the source at that exact point. The PDF/
deck analogue is **page number + page thumbnail + the rendered region**, carried in the Qdrant
**payload** (the diagram's upsert is explicitly `int8 + payload`, p.15) and fetched from object
storage at answer time (step 8). Grounding is a *payload-and-storage* design decision made at
ingest, not a prompt-engineering step at answer time.

**3.2 — Queue lifecycle parity.** Prefect deployment-per-flow, arguments carrying the document URL
(p.8). Match the video status lifecycle and retries; treat the Prefect failed-state UI as the DLQ,
but persist real status in Postgres.

**3.3 — Benchmark methodology, derived from the deck's own claims:**
- **Accept latency** is the metric that proves "uploads no longer wait for processing" (p.5).
  Measure API accept time (manifest write + enqueue only) and show it is flat and decoupled from
  document size and worker load.
- **Search-during-backfill** is the "different queue" claim (p.4) plus fair queueing (p.13).
  Measure p50/p95 query latency with the ingestion lane idle vs. saturated; the deck's design
  predicts no meaningful change.
- **Throughput** scales with worker count (p.7, item 4) — run 1 / 2 / N workers and show the slope,
  since "more workers consume tasks in parallel" is an explicit claim to test.
- **Recall** must be measured *against the quantization claim* (p.6): int8-in-RAM + float32 rescore
  should cost **< 1 %** accuracy vs. an fp32-only baseline. Compare recall@6 (the deck's answer
  size) over a 20-candidate retrieval set.
- **Kill-a-worker** targets the Fetch → Process → **Complete** loop (p.14). Kill mid-Process, show
  the job returns to the queue, re-runs, and — because point IDs are deterministic — produces the
  same vector count, not duplicates.

**3.4 — FDE judgment.** The p.11 bake-off is the ready-made managed-vs-self-hosted argument
(cost and ops burden as the deciding axes, capability roughly at parity). The p.3 failure list is
the ready-made "what a demo does that production can't" narrative.

---

## 10. Source map (page → topic)

| Page | Topic |
|---|---|
| 1 | Section title — "Deepdive: Video based RAG at Scale" |
| 2 | MomentSearch product intro (Lenny's Podcast, 10 episodes, cited moments) |
| 3 | The monolithic demo pipeline — 10 stages, 4 failure classes |
| 4 | Target flow: separate upload queue, two lanes, image-and-text results |
| 5 | Scalable stack: gateway → rate limit → queue → workers → storage |
| 6 | Scalar quantization: int8 RAM / float32 disk, 4×, <1 % loss |
| 7 | Four reasons queues are essential for RAG (incl. Parse→Chunk→Embed→Index, DLQ) |
| 8 | Prefect ↔ RabbitMQ pattern mapping |
| 9 | Further reading — YouTube links on message queues, Kafka vs RabbitMQ |
| 10 | Before/after: blocking pipeline → parallel worker pools |
| 11 | Platform bake-off: Redis / custom / Temporal / **Prefect (selected)** |
| 12 | Fair queueing — definition |
| 13 | Fair queueing — dispatcher diagram, round-robin example |
| 14 | Worker — Fetch → Process → Complete loop |
| 15 | **Production architecture at scale** — full write/read path + infra inventory |
