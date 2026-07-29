# Brief — Class Deck, Assignment 3 (Moment Search)

Source: `Class_Deck_Assignment_3_Moment_Search.pdf` — **4 slides only**, all rendered as
images (no text layer; quotes below are transcribed from the slide images at 300 dpi).
Slide numbers = PDF page numbers.

**Headline caveat:** the deck is a framing deck, not a spec. It contains **no grading rubric,
no point weights, no submission instructions, no deadline, no dataset list, and no numeric
thresholds.** Everything the graders will actually check has to be inferred from the four
bullets on slide 3 and the architecture on slide 4. Do not go looking for detail in this PDF
that isn't here — the depth lives in `Architecture_Considerations_Moment_Search.pdf` and the
`momentsearch` repo.

---

## Slide 1 — Title

> "Ship Your Third **Full-Stack AI Product**"
> "**Project:** Build a scalable multimodal RAG pipeline"

Framing to echo: this is a *product ship*, the third in a series — not a notebook exercise.
The word "scalable" is in the project title itself; scale is the graded theme, not a bonus.

---

## Slide 2 — The Next Challenge: Unifying Text and Visual Retrieval

Subtitle, verbatim:

> "Text RAG finds the answer. Moment Search finds the evidence. The goal is to return both together."

Three-box diagram the write-up should mirror:

| Box | Pipeline shown | Returns |
|---|---|---|
| **TEXT RAG** (left) | Transcripts → Retrieval → LLM | "Answer" |
| **MULTIMODAL FUSION** (center) | "Combine language and visual evidence" (💬 + 🎞) | "Answer + relevant clip" |
| **MOMENT SEARCH** (right) | Frames → Embeddings → Match | "Relevant video moment" |

Bottom box: **"UNIFIED RESPONSE — Answer + supporting video clip."**

Takeaway the graders are primed for: the deliverable is not "search over PDFs" bolted next to
"search over video." It is **one fused response** where the language answer and the visual
evidence arrive together. If the demo returns two separate result lists, it misses the slide's
whole premise.

---

## Slide 3 — Project 3: Build a scalable multimodal RAG pipeline

The four graded parts, transcribed verbatim (this is the closest thing to a rubric in the deck;
each bullet's last sentence is a *learning objective* — treat it as the thing to demonstrate,
not just do):

**3.1 Build a multi-source ingestion pipeline.**
> "Add PDF papers and slide decks to a video-search app. Parse, chunk, enrich, embed. Put
> everything in one Qdrant index. Learn how mixed sources become one searchable space."

- Two new source types explicitly named: **PDF papers** and **slide decks**. Not "documents."
- Four verbs, in order — **parse, chunk, enrich, embed**. "Enrich" is the one teams drop; it
  sits between chunking and embedding (captions/titles/section metadata/page numbers).
- Hard constraint: **one Qdrant index** for everything. Not a second collection per modality.
  Implies a shared payload schema and a modality/source-type field for filtering.
- Demonstrate: mixed sources behaving as **"one searchable space"** — i.e. a single query
  returning video moments and paper/slide chunks ranked against each other.

**3.2 Learn async work queues.**
> "Add paper and deck flows to the existing Prefect queue. Match video's status lifecycle and
> retries. Learn why queues separate ingestion from search."

- Tool is prescribed: **Prefect**, and specifically **the existing queue** — extend, don't
  build a parallel one.
- "**Match** video's status lifecycle and retries" — the paper/deck flows must reuse the same
  status states and retry policy the video flow already has. Parity is the test; read the
  repo's existing states and mirror them rather than inventing new ones.
- Demonstrate: **why queues separate ingestion from search** — the argument, not just the code.

**3.3 Scale and prove decoupling.**
> "Benchmark accept latency, throughput, and recall. Confirm search stays fast during a big
> backfill. Kill a worker mid-ingest. Prove nothing is lost."

- Three named metrics, exactly: **accept latency** (time for the API to accept/enqueue an
  upload — *not* end-to-end ingest time), **throughput**, **recall**.
- Two named experiments: **(a)** search latency measured *while a big backfill is running*
  — the decoupling proof; **(b)** **kill a worker mid-ingest** and prove **no loss** — the
  durability proof (requires at-least-once delivery, idempotent upserts, and a resumable
  status lifecycle from 3.2).
- The verbs are "Benchmark… Confirm… Prove." Numbers and before/after evidence are expected;
  assertions are not.

**3.4 Think like a Forward Deployed Engineer.**
> "Weigh managed vs. self-hosted queues. Keep every citation grounded. Make workers resilient.
> Turn a demo into something that survives a real backfill."

- Four discussion obligations: **managed vs. self-hosted queue trade-off**, **grounded
  citations** (every claim traceable to a timestamp/page), **worker resilience**, and the
  demo→production gap framed as **"survives a real backfill."**

---

## Slide 4 — MomentSearch: Production Architecture at Scale

One dense diagram, ten numbered steps across two paths. This is the reference architecture the
write-up should be legible against; the parts I could not read are flagged at the end.

### INGEST (write path)

1. **Browser** — user uploads video. Dashed side-arrow: **upload (multipart) → Object Storage
   (GCS / S3 / Tigris)** (client uploads bytes directly; the API never proxies the file).
2. **API (FastAPI)** — two fan-out edges: **"register manifest" → Postgres (Neon)** (video
   manifest / status) and **"enqueue job" → Job Queue** (background jobs). This is where
   *accept latency* is measured.
3. **Ingest Worker** (labeled **Background**) — three stages: **sample frames (~2s)** →
   **pHash dedup** → **extract captions**.
4. **CLIP service** — "**one warm model, 800D**".
   Worker → Qdrant edge is labeled **"upsert (int8 + payload)"**.

Three green check callouts under the ingest path (optimizations to name in the write-up):
- **"one CLIP model, shared & warm (CPU→GPU swap)"**
- **"~4x less RAM via int8"**
- **"dedup kills near-duplicate vectors"**

### QUERY (read path)

5. **Browser** — text query.
6. **API (FastAPI)** with two sub-boxes: **"rate limiting + fair queue"** and **"time to first
   byte."** Two labeled edges to Qdrant: **"embed text"** and **"search (user_id filter)."**
7. **Rerank + Fusion** — "**(RRF + cross-modal boost)**", "**top 6 moments**."
8. **Object Storage** (thumbs / clips) → labeled **"fetch clipped frames"**.
9. **Multimodal LLM** — "(GPT-4o / vLLM / Others)".
10. **Browser** — "results (cited moments)", fed by an edge labeled **"cited answer +
    timestamps + thumbnails."**

### Qdrant configuration (called out twice — main node and sidebar)

- **int8 in RAM + float32 on disk · HNSW on disk · multi-tenant**
- **"HNSW on int8 (RAM) → rescore on float32 (disk) · return 20 candidates"**
  (so: 20 candidates retrieved → reranked/fused → **top 6** shown.)

### Data & Infrastructure sidebar

- **Object Storage** (videos + thumbnails)
- **Postgres (Neon)** (manifest + status)
- **Qdrant** (int8 in RAM + float32 on disk · HNSW on disk · multi-tenant)
- **Job Queue** (background jobs)
- **Compute** (containers / autoscaling workers)

---

## Prescribed tools, numbers, and defaults (everything numeric in the deck)

| Item | Value | Slide |
|---|---|---|
| Vector store | Qdrant, **one index**, multi-tenant | 3, 4 |
| Queue | **Prefect** (existing queue, extended) | 3 |
| API | FastAPI | 4 |
| Metadata/status store | Postgres (Neon) | 4 |
| Embedding model | CLIP, **one warm model, 800D** | 4 |
| Vector precision | **int8 in RAM, float32 on disk** (~4x RAM saving) | 4 |
| Index | **HNSW on disk**, search on int8 then **rescore on float32** | 4 |
| Frame sampling | **~2s** interval | 4 |
| Dedup | **pHash** near-duplicate removal | 4 |
| Fusion | **RRF + cross-modal boost** | 4 |
| Candidates → results | **20 candidates → top 6 moments** | 4 |
| Tenant isolation | **user_id filter** on search | 4 |
| Answer LLM | GPT-4o / vLLM / "Others" | 4 |
| Metrics to benchmark | accept latency, throughput, recall | 3 |

No target thresholds are given for any metric. No dataset is named — choice of papers/decks is
left to the builder.

---

## Terminology the write-up should use (graders will look for these words)

`multimodal fusion` · `unified response` · `one searchable space` · `parse / chunk / enrich /
embed` · `status lifecycle` · `retries` · `accept latency` · `throughput` · `recall` ·
`backfill` · `decoupling` (ingestion vs. search) · `grounded citations` · `worker resilience` ·
`managed vs. self-hosted` · `RRF` / `reciprocal rank fusion` · `cross-modal boost` ·
`pHash dedup` · `int8 quantization` / `rescore on float32` · `HNSW` · `multi-tenant` /
`user_id filter` · `warm model` · `time to first byte` · `fair queue` / `rate limiting` ·
`manifest`.

---

## Easy to miss / surprising

1. **The slide-4 diagram is cropped.** At the bottom of the image, two more panel outlines
   begin and are cut off by the slide edge — the deck shows a truncated version of a taller
   architecture graphic. Assume there is at least one more band (likely scaling/failure notes)
   that only exists in `Architecture_Considerations_Moment_Search.pdf`. Don't treat slide 4 as
   the complete architecture.
2. **"Accept latency," not ingest latency.** 3.3's first metric is about how fast the API says
   yes and returns — the queue's whole point. Measuring end-to-end ingest time instead misses
   the metric being asked for.
3. **"Match video's status lifecycle and retries"** is a parity requirement against existing
   repo code. It is checkable by diff; inventing a different set of states for papers/decks
   fails the literal wording.
4. **"one Qdrant index"** rules out the easy path of a second collection for text. Cross-modal
   ranking in a single space is the hard part being graded.
5. **"Enrich"** appears in the 3.1 verb chain and is the most commonly skipped step.
6. **Fair queue + rate limiting** appear on the read path (step 6) — read-path protection is
   part of "search stays fast during a big backfill," not just worker-side concurrency limits.
7. **20 → 6** is a concrete retrieval budget worth preserving when documents join the index;
   the candidate pool is now shared across modalities.
8. **CLIP is 800D**, not the stock 512D — the deck implies a concatenated/extended embedding.
   Whatever PDF/slide embeddings are added must land in the same vector space and dimension to
   share one index.
9. **No rubric, no deadline, no bonus section in the deck.** The only submission signal in this
   folder is in `Assignment_3_Moment_Search_Background.md` ("Project • Submit by WED. JUL 29",
   "View 1 submission") — that is course-portal text, **not** from this PDF. Treat any other
   grading detail as coming from the portal or the architecture PDF, not here.
10. **The deck never mentions bge or text embeddings at all** — only CLIP. The text-embedding
    side of the multi-source index is unspecified in the deck and is a design decision the
    write-up will need to justify on its own.
