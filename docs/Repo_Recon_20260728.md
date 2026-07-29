# MomentSearch — Repository Recon (2026-07-28)

Produced by an Explore sub-agent against the fresh clone at `momentsearch/`
(upstream `traversaal-ai/momentsearch`, commit `8526743`). All paths repo-relative.
This is the canonical architecture reference for Assignment 3 planning — read this
instead of re-exploring the repo.

> Access note: `.env.example` (15.6 KB) was blocked by permission settings during recon.
> §2 is reconstructed from `src/config.py` (the authoritative env surface), `DEPLOYMENT.md`,
> `fly.toml`, and `docker-compose.yml`.

---

## 1. Stack overview

**Language:** Python 3.11 (`Dockerfile:6`). Frontend is a single hand-written HTML file with Tailwind CDN — no build step, no Node app (`ui/index.html`, 445 lines).

**Frameworks / key deps** (`requirements.txt`):

| Area | Package | Note |
|---|---|---|
| Web | `fastapi>=0.110`, `uvicorn[standard]>=0.27`, `python-dotenv>=1.0` | |
| Manifest DB | `psycopg[binary,pool]>=3.1` | Neon Postgres |
| Orchestration | `prefect>=3.0` | Prefect Cloud — genuinely used, not vestigial |
| Object storage | `boto3>=1.34`, `google-cloud-storage>=2.16` | S3-protocol + GCS-native |
| Vector DB | `qdrant-client>=1.12` | |
| Visual embed | `sentence-transformers>=2.7`, `pillow`, `numpy` | pulls torch |
| Text embed | `fastembed>=0.3` | bge via ONNX, no torch |
| Video acquisition | `yt-dlp>=2026.7.4` | |
| LLM | `openai>=1.30`; `anthropic` commented out at `requirements.txt:31` | |

**Container:** one image, four entrypoints selected by CMD (`Dockerfile:1-5`). Installs `ffmpeg` + `nodejs` (`Dockerfile:11-14`; Node is mandatory for modern yt-dlp signature extraction). Installs CPU-only torch from the PyTorch CPU index first to avoid ~6 GB CUDA wheels (`Dockerfile:21`). Default CMD `uvicorn src.app:app --port 8000` (`Dockerfile:28`).

**Runtimes:**
- API + UI — `uvicorn src.app:app` :8000
- Worker — `python -m src.worker` (no ports, outbound only)
- CLIP service — `uvicorn src.clip_service:app` :8001
- Seed gate — `python -m src.seed` (one-shot, exits)

**Deploy target:** Fly.io. `fly.toml:26-31` defines three process groups (`api`, `worker`, `clip`); `fly.toml:24` sets `release_command = "sh -c 'CLIP_SERVICE_URL= python -m src.seed'"`. VM sizes: api 512 MB shared-cpu-1x, worker 2 GB shared-cpu-2x, clip 2 GB shared-cpu-2x (`fly.toml:63-78`). Restart policies at `fly.toml:43-53`. CI: `.github/workflows/fly-deploy.yml` runs `flyctl deploy --remote-only` on push to **`dev`** (not main).

---

## 2. External services & credentials

Five external dependencies, all "rented managed services" by design (README:34-37): **Neon Postgres, Prefect Cloud, Qdrant Cloud, object storage, a vision LLM**.

### Required to run locally (minimum viable `docker compose up`)

| Var | Where read | Notes |
|---|---|---|
| `DATABASE_URL` | `src/config.py:39` | **Hard requirement.** No fallback — `src/db.py:30` passes it straight to `ConnectionPool`. Neon Postgres free tier. |
| `PREFECT_API_URL`, `PREFECT_API_KEY` | not in config.py — read by the Prefect SDK directly (`src/config.py:256-258`) | Required for the worker path. `examples/quickstart.py` runs the flow **in-process** with these unset. |
| `QDRANT_URL` / `QDRANT_API_KEY` (or `QDRANT_TOKEN`) | `src/config.py:263-264` | If `QDRANT_URL` is blank, falls back to embedded local Qdrant at `QDRANT_LOCAL_PATH` (`src/config.py:265`, `src/rag/vector_store.py:68-72`) — **single-process only**, README:543 flags that api+worker can't share it. |
| `STORAGE_PROVIDER` | `src/config.py:53` | Defaults to `local` → `./data`, credential-free, no presigning (`src/storage.py:87-89`). |

Everything else has a working default. Notably **search works with zero API keys**: CLIP runs locally, the transcript branch defaults to fastembed/bge (`src/config.py:184`), and with no LLM configured the read path returns a similarity summary instead (`src/rag/search.py:232-238`).

### Optional locally / required for a real deploy

**Auth & tenancy**
- `ADMIN_TOKEN` (`src/config.py:46`) — unset = **all mutating endpoints open** (`src/api/videos.py:44-48`). README:519 flags this as required on any public deploy.
- `DEFAULT_USER_ID` (default `"default"`); tenant comes from the `X-User-Id` header.

**Object storage** — `STORAGE_PROVIDER` ∈ `local | aws | gcp | gcp_native | flyio` (`src/config.py:50-66`)
- S3-protocol path (aws/gcp/flyio): `STORAGE_BUCKET` (also accepts `BUCKET_NAME`, `GCS_BUCKET_NAME`, `GOOGLE_CLOUD_BUCKET_NAME`), `STORAGE_ACCESS_KEY_ID`/`AWS_ACCESS_KEY_ID`, `STORAGE_SECRET_ACCESS_KEY`/`AWS_SECRET_ACCESS_KEY`, `STORAGE_REGION`/`AWS_REGION`, `AWS_ENDPOINT_URL_S3`. Endpoints hard-coded per provider at `src/config.py:61-65` (flyio → `https://fly.storage.tigris.dev`, gcp → `https://storage.googleapis.com`).
- `gcp_native`: exploded service-account JSON — `GOOGLE_CLOUD_PROJECT_ID`, `_PRIVATE_KEY_ID`, `_PRIVATE_KEY`, `_CLIENT_EMAIL`, `_CLIENT_ID`, `_AUTH_URI`, `_TOKEN_URI`, `_AUTH_PROVIDER_X509_CERT_URL`, `_CLIENT_X509_CERT_URL`, `_UNIVERSE_DOMAIN` (`src/config.py:69-90`). DEPLOYMENT.md:47 says this is the actual production config, bucket `momentsearch-media`.
- Bucket needs a **CORS rule allowing PUT** from the site origin or browser uploads fail (DEPLOYMENT.md:191).
- Tunables: `PRESIGN_EXPIRY_S` (900), `PRESIGN_GET_EXPIRY_S` (3600), `MAX_UPLOAD_MB` (2048).

**LLM (answer synthesis only)** — `LLM_PROVIDER` ∈ `openai | nvidia | anthropic` (`src/config.py:287`), `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` (default `gpt-4o-mini`), `LLM_MAX_TOKENS`, `LLM_IMAGE_MAX_PX`. `llm_configured()` at `src/config.py:295-297` returns true on key **or** base_url. NVIDIA endpoint hard-coded at `src/llm.py:32`. Per-tenant override lives in Postgres (`ms_user_llms`), resolved at `src/rag/search.py:199-208`.

**Text embeddings** — `TEXT_EMBED_PROVIDER` (`fastembed` | `openai`), `TEXT_EMBED_MODEL`, `TEXT_EMBED_DIM`, `TEXT_EMBED_API_KEY`, `TEXT_EMBED_BASE_URL`, `TEXT_EMBED_VERSION` (`src/config.py:184-194`). Falls back to `LLM_API_KEY` when provider is openai (`src/rag/embeddings.py:120`).

**YouTube hardening** (needed for any datacenter deploy) — `YT_COOKIES_FILE`, `YT_COOKIES_B64`, `YT_PROXY_URL`, `YT_PLAYER_CLIENTS`, `YT_FALLBACK_CLIENTS`, `YT_JS_RUNTIMES` (default `node`), `YT_REMOTE_COMPONENTS` (default `ejs:github`) — `src/config.py:212-248`, consumed at `src/ingest/fetch.py:44-100`. DEPLOYMENT.md:189 notes cookies expire in 2-3 weeks.

**Fly-specific** — `FLY_API_TOKEN` (flyctl / GH secret). DEPLOYMENT.md:62-67 shows `.env` also carries `FLY_IO_TOKEN`, deliberately excluded from `fly secrets import`.

**Behavioral knobs** — `CLIP_MODEL`, `CLIP_BATCH`, `CLIP_SERVICE_URL`, `CLIP_DIM`, `EMBED_VERSION`, `FRAME_STRATEGY`, `FRAME_INTERVAL_SEC`, `SCENE_THRESHOLD`, `MAX_FRAMES`, `THUMB_WIDTH`, `THUMB_QUALITY`, `DEDUP_ENABLED`, `DEDUP_MAX_DISTANCE`, `ENABLE_TRANSCRIPT`, `TEXT_COLLECTION`, `TRANSCRIPT_CHUNK_SECONDS`, `TRANSCRIPT_LANGS`, `RRF_K`, `FUSION_WINDOW_S`, `CROSS_MODAL_BOOST`, `BRANCH_TOP_K`, `TOP_K`, `KNN_K`, `CONFIDENCE_THRESHOLD`, `TEXT_CONFIDENCE_THRESHOLD`, `QDRANT_COLLECTION`, `QDRANT_QUANTIZATION`, `QDRANT_ON_DISK`, `QDRANT_HNSW_ON_DISK`, `ENABLE_FAIR_DISPATCH`, `DISPATCH_MAX_INFLIGHT`, `DISPATCH_INTERVAL_S`, `WORKER_CONCURRENCY`, `SEED_SAMPLE_VIDEOS`.

`.gitignore` ignores `.env` and `.env.*` but whitelists `.env.example`. It also ignores an unpublished internal design note `MULTIMODAL_RETRIEVAL.md`.

---

## 3. Architecture map — video ingestion end to end

### Write path

**1. Presign** — `POST /api/videos/presign` (`src/api/videos.py:66-83`). Bearer-auth'd. Validates size ≤ `MAX_UPLOAD_MB` and content-type against `ALLOWED_UPLOAD_TYPES = ("video/",)` (`src/config.py:103`). **The server mints the key** `uploads/{user_id}/{video_id}{ext}` with `video_id = f"up_{uuid4().hex[:10]}"` — never trusted from the client. If `STORAGE_PROVIDER=local`, returns `{"mode": "direct", ...}` pointing at `PUT /api/videos/{video_id}/content` (`src/api/videos.py:86-105`) instead.

**2. Browser PUTs** straight to the bucket (`ui/index.html:328-333`).

**3. Register** — `POST /api/videos` → **202** (`src/api/videos.py:117-149`). Two accepted shapes:
- `{url}` — matched against `_YT_RE` (`src/api/videos.py:40-41`, watch/shorts/live/embed/youtu.be); `video_id = f"yt_{11-char-id}"`, `source="youtube"`.
- `{video_id, key}` — key prefix re-verified against the user, object HEAD-verified for existence and size; `source="upload"`.
- Anything else → 400. Then `db.upsert_pending` writes a `pending` row and returns immediately.

**4. Queue.** Prefect **is** present and load-bearing:
- `src/jobs.py:15` `INGEST_DEPLOYMENT = "ms-ingest-video/ingest"`; `enqueue_video` calls `run_deployment(..., timeout=0)` fire-and-forget.
- `src/worker.py:39` `ingest_video.serve(name="ingest", limit=WORKER_CONCURRENCY)` registers the deployment and long-polls. Wrapped in a retry-forever loop (`src/worker.py:36-45`) so a Prefect blip pauses ingest instead of killing the machine.
- **WFQ dispatcher in front of Prefect** (`src/dispatcher.py`). When `ENABLE_FAIR_DISPATCH=true` (default), register leaves the row `pending` and a daemon thread in the worker tops up: `slots = DISPATCH_MAX_INFLIGHT - count_inflight()`, then `db.wfq_claim(slots)` (`src/db.py:190-225`) atomically claims rows in round-robin-across-users order via `row_number() OVER (PARTITION BY user_id ORDER BY created_at, id)` and flips `pending → queued` with `UPDATE ... WHERE status='pending' RETURNING`. Set false → plain FIFO enqueue at register time.

**5. Flow** — `src/ingest/pipeline.py:157` `@flow(name="ms-ingest-video", log_prints=True, timeout_seconds=3600)`. Four tasks:

| Task | Line | Retries |
|---|---|---|
| `t_fetch` | `pipeline.py:38` | `retries=2, retry_delay_seconds=[30, 120]` |
| `t_sample` | `pipeline.py:67` | none |
| `t_embed_index` | `pipeline.py:95` | `retries=2, retry_delay_seconds=60` |
| `t_transcript` | `pipeline.py:123` | `retries=1, retry_delay_seconds=30`; swallows all exceptions (`pipeline.py:152-154`) — never fails the flow |

Flow body `pipeline.py:158-177`: `bump_attempts` → fetch → (empty path = duplicate, early return) → sample → embed → transcript; on exception sets status `failed` and re-raises; `finally` unlinks scratch.

### Status lifecycle
`VIDEO_STATUSES = ("pending","queued","fetching","sampling","embedding","indexed","skipped","failed")` (`src/config.py:110-111`); `INFLIGHT_STATUSES = ("queued","fetching","sampling","embedding")` (`src/config.py:113`). `progress` is a 0..1 float within the current stage (`db.set_progress`, updated every 25 thumbnail uploads at `pipeline.py:86-87`). `skipped` = duplicate `(user_id, source_hash)` — `find_duplicate` at `src/db.py:140-150` only matches rows already `indexed`. Manual retry: `POST /api/videos/{id}/retry` resets to `pending` (`src/api/videos.py:179-188`).

### Chunking / embedding

**Visual branch:**
- `src/ingest/frames.py` — one ffmpeg pass, MJPEG piped to **memory** (never to disk), split on SOI/EOI markers (`frames.py:31-44`). Two strategies: `interval` (`fps=1/N`, spacing auto-widened to respect `MAX_FRAMES`, `frames.py:77-87`) or `scene` (`select='gt(scene,THRESH)',showinfo`, timestamps parsed from stderr `pts_time:`, `frames.py:89-102`). Downscaled to `THUMB_WIDTH=480` in the same pass.
- `src/ingest/dedup.py` — dHash (9×8 grayscale gradient → 64-bit) + mean-luminance guard; drops frames within `DEDUP_MAX_DISTANCE=4` Hamming of the **previous kept** frame (`dedup.py:39-50`).
- Thumbnails uploaded to `frames/{user_id}/{video_id}/NNNNNN.jpg` via an 8-way `ThreadPoolExecutor` (`pipeline.py:81-90`); prior run's prefix deleted first for idempotency.
- CLIP embed in `CLIP_BATCH=128` chunks (`pipeline.py:103`), L2-normalized (`embeddings.py:51-54`).

**Text branch:** `src/ingest/transcript.py` — yt-dlp subtitle-only fetch in json3 format reusing the same auth path as download (`transcript.py:22-34`), parsed to cues (`_parse_json3`), then `chunk_cues` groups cues into ~`TRANSCRIPT_CHUNK_SECONDS=20` windows each with real `t_start`/`t_end` (`transcript.py:82-101`). Embedded with bge (fastembed) or OpenAI.

**Embedding-as-a-URL:** `src/rag/embeddings.py:155-196` dispatches on `CLIP_SERVICE_URL` — set → HTTP POST to `src/clip_service.py` (`/embed/images`, `/embed/text`, `/embed/docs`, `/embed/query`, `/healthz`); unset → in-process. Retries 12× at 5 s for a cold service (`embeddings.py:141-150`).

### Qdrant schema (`src/rag/vector_store.py`)

Two collections, both created by `_ensure()` (`vector_store.py:91-126`):

| | `moments` (`QDRANT_COLLECTION`) | `moments_text` (`TEXT_COLLECTION`) |
|---|---|---|
| Vector size | `_dim()` → `CLIP_DIM` or lookup table `vector_store.py:47-52` (`clip-ViT-B-32` = **512**) | `TEXT_EMBED_DIM` = **384** (bge) or 1536 (OpenAI) |
| Distance | COSINE | COSINE |
| Point ID | `uuid5(NAMESPACE_URL, f"{video_id}:{frame_idx}")` (`:76-77`) | `uuid5(NAMESPACE_URL, f"{video_id}:text:{i}")` (`:182-183`) |
| Payload | `user_id, video_id, ms, idx, modality:"frame", t_start, t_end, embed_version` (`pipeline.py:110-114`) | `user_id, video_id, modality:"text", t_start, t_end, ms, text, embed_version` (`pipeline.py:146-149`) |

Storage profile: `on_disk` vectors, INT8 scalar quantization `always_ram=True`, HNSW on disk — all default-on. Payload indexes: `user_id` as a **tenant** keyword index (`is_tenant=True`, with graceful fallback), `video_id` as plain keyword. Every search/upsert/delete is `user_id`-filtered (`_user_filter`, `:80-88`). Missing-collection errors are swallowed into `[]` (`:166-171`).

Titles/URLs deliberately live **only in Postgres** and are joined at answer time (`db.videos_by_ids`).

### Search & citation (`src/rag/search.py`)

`POST /api/ask {question, video_id?, video_ids?, top_k?}` (`src/api/search.py:139-148`).

1. **Both branches always** (no query router): CLIP text→image against `moments`, bge query→chunk against `moments_text`, each `BRANCH_TOP_K=20` (`search.py:115-126`).
2. **`_fuse`** (`search.py:31-74`) — RRF `1/(RRF_K + rank)` per branch; hits within `FUSION_WINDOW_S=15 s` of each other in the same video collapse into one "moment"; only the **best hit per modality** is kept per window (explicitly to prevent a burst of near-identical frames outranking a true frame+transcript match, `search.py:61-65`); score = best-frame rrf + best-text rrf, ×`CROSS_MODAL_BOOST=1.5` when both modalities are present.
3. **Gate 1** (`search.py:225-229`) — abstains with a canned message and **zero LLM cost** when `best_visual < CONFIDENCE_THRESHOLD (0.2)` AND `best_text < TEXT_CONFIDENCE_THRESHOLD (0.35)`. Thresholds are per-branch raw cosines, not RRF.
4. **Citations** (`search.py:131-154`) — each carries `n, video_id, title, url, source, ms, timestamp (mm:ss via _seconds), idx, thumbnail, media_url, deeplink, score, transcript, modalities`. `_deeplink` (`search.py:77-82`) → `{youtube_url}?t={secs}` or `/api/video/{id}#t={secs}`. `_thumb_url` returns a presigned GET when the provider supports it, else the local `/api/frame/...` route.
5. **Generate** — `_build_moments` (`search.py:182-196`) fetches frame JPEGs 6-way in parallel and hands the LLM `{image, transcript, timestamp}` per moment. System prompt at `src/llm.py:36-65`. Images downscaled to `LLM_IMAGE_MAX_PX=512` (`llm.py:106-116`).
6. **`_validate_citations`** (`search.py:170-179`) regex-strips `[n]` references outside `1..len(citations)`.
7. No-LLM fallback: `_fallback_answer` ranks the closest moments honestly (`search.py:158-167`).

---

## 4. Extension points for PDFs / slide decks

Where each layer would need to change (in dependency order):

**a) Upload gate** — `src/config.py:103` `ALLOWED_UPLOAD_TYPES = ("video/",)`; enforced at `src/api/videos.py:70`. Extension bound to `_EXT_RE` at `src/api/videos.py:39`. Key prefix `UPLOAD_KEY_PREFIX = "uploads/"` (`src/config.py:96`) — a doc prefix or reuse.

**b) Ingestion routing** — `POST /api/videos` (`src/api/videos.py:117-149`) branches on `req.url` (YouTube regex) vs `req.video_id + req.key`. Adding a source type = a third branch here plus a new `video_id` prefix convention (`yt_` / `up_` today, `src/samples.py:33-34` and `search.py:400` both parse `yt_`).

**c) Manifest schema** — `src/db.py:37-57`. `source TEXT NOT NULL -- youtube | upload` is a free-text column, so `source='pdf'` needs no migration; `frame_count` would become a page/slide count. `SCHEMA` is applied via plain `CREATE TABLE IF NOT EXISTS` in `init_schema()` — there is **no migration framework**, so added columns need an `ALTER TABLE IF NOT EXISTS` appended to the same string.

**d) Parser layer** — `src/ingest/` is the natural home. The contract is narrow and reusable:
- `fetch.py` — `fetch_upload(storage_key, video_id) -> Path` (`fetch.py:35-38`) already downloads any object type from the bucket; no change needed for PDF bytes. `sha256_file` gives the dedup identity.
- `frames.py:25-28` — `@dataclass Frame(ms: int, jpeg: bytes)`. A PDF/slide parser producing rendered page images as JPEGs plus a page-index would slot straight into `t_sample` / `t_embed_index`, but `ms` is the only positional field, so a page number has to be encoded into `ms` or the dataclass extended.
- `transcript.py:82-101` — `chunk_cues(cues) -> [{text, t_start, t_end}]` is the text-chunk contract. A PDF text extractor emitting `{text, page_start, page_end}` mirrors this exactly.

**e) Flow / task layer** — `src/ingest/pipeline.py`. The flow is a hard-coded linear chain (`pipeline.py:162-171`) with `t_fetch → t_sample → t_embed_index → t_transcript`. Extension = either a `source`-conditional branch inside `ingest_video`, or a second `@flow` plus a second deployment name in `src/jobs.py:15` and a second `.serve()` in `src/worker.py:39` (note: `serve()` currently serves exactly one flow and blocks).

**f) Embedding step** — `src/rag/embeddings.py:155-196` is the dispatch layer; `src/clip_service.py:66-86` is the warm-model service. Page images can reuse `embed_jpegs` unchanged (CLIP handles slides/diagrams well — that's the sample corpus's whole premise). PDF body text reuses `embed_docs`/`embed_query`. A new modality-specific model would need a new endpoint on `clip_service.py` + a new dispatch fn.

**g) Index schema** — `src/rag/vector_store.py:129-136`. `_ensure(collection, dim)` is already generic, so a third collection (e.g. `docs_text`, `docs_pages`) is a two-line addition. But three call sites assume exactly two collections: `delete_video` iterates a hard-coded tuple (`vector_store.py:215`), `collection_ready()` checks only `QDRANT_COLLECTION` (`:222-226`), and `src/app.py:36-38` ensures only the two. Payload already carries `modality` (`"frame"` / `"text"`) — that's the natural discriminator for `"page"` / `"slide"`.

**h) Fusion / retrieval** — `src/rag/search.py:40-65`. `_fuse` is hard-wired to two branches and joins on **time** (`abs(w["t"] - h["t"]) <= FUSION_WINDOW_S`). A page-based source has no time axis, so the window join key is the main structural assumption to break. `retrieve()` (`:103-155`) is likewise a two-call function, and the gate (`:225-229`) checks exactly two thresholds.

**i) Citation shape** — `search.py:139-154` builds `ms` / `timestamp` (`_seconds`, `:26-28`) / `idx` / `deeplink`. `_thumb_url` (`:85-91`) hard-codes the `frames/…/NNNNNN.jpg` key layout (`storage.frame_key`, `storage.py:49-50`), and the local media route validates `^\d{6}\.jpg$` at `src/api/search.py:24`. A page citation needs a page-number analogue of `timestamp` and a doc-anchor analogue of `deeplink`.

**j) LLM prompt** — `src/llm.py:36-65` (SYSTEM), `:95-103` (`_intro`), `:148-154` (`_label`) all speak in terms of "video / frame / transcript / timestamp". Adding a page/slide moment type means editing these three.

**k) UI** — `ui/index.html`. The add-source tabs are at lines 82-97 (`#tab-yt` YouTube input, `#tab-up` file input with `accept="video/*"` at line 94); handlers at 321-340 (`registerIngest`, presign+PUT). Status chips/badges at 197-296; the modality tags rendering `frame`/`text` at 369-374; the playback modal at 402-429 which switches on YouTube-iframe vs `<video>`. `applyMode()` (line 151) gates `/` (sample, read-only) vs `/get-started` (full). `MS_MODE` is injected server-side by `_render()` at `src/api/search.py:215-224`.

---

## 5. Local dev story

### docker-compose (`docker-compose.yml`, 4 services, 1 named volume)

| Service | Command | Ports | depends_on |
|---|---|---|---|
| `clip` | `uvicorn src.clip_service:app --port 8001` | internal | — |
| `seed` | `python -m src.seed` | — | `clip` (started); `restart: "no"` |
| `api` | default CMD (uvicorn :8000) | **8000:8000** | clip started + **seed `service_completed_successfully`** |
| `worker` | `python -m src.worker` | none | same as api |

All four `env_file: .env`. `CLIP_SERVICE_URL` defaults to `http://clip:8001`. `worker` gets `WORKER_CONCURRENCY: 2`. Volumes: `./data:/app/data` (local storage provider only) and a named `hf_cache:/root/.cache` so model weights survive restarts.

**The startup gate is the notable design choice** (`docker-compose.yml:22-39`): api and worker will not start until `seed` exits 0, so `localhost:8000` is never reachable with a half-indexed corpus. First run takes minutes (~600 MB model download + 4 videos); later runs exit in seconds because state is durable in Qdrant/Neon. `docker compose logs -f seed` to watch. Scale with `docker compose up -d --scale worker=3`.

**No Qdrant service is defined in compose** — `docker-compose.yml:6-8` explicitly says the vector store is Qdrant Cloud via `QDRANT_URL`, and to add a local qdrant service yourself if you want one. (README:107 and the examples README:26 assume `docker run -p 6333:6333 qdrant/qdrant` separately. This is a small doc/compose inconsistency worth knowing.)

### Bare processes (README:146-151)
```
uvicorn src.app:app --port 8000            # API + UI
python -m src.worker                        # ingest worker
uvicorn src.clip_service:app --port 8001    # optional; else leave CLIP_SERVICE_URL empty
python -m src.seed                          # one-shot: index the 4 samples
```
A bare `docker run` of the image starts uvicorn only and **skips seeding** (README:140-142).

### Seed / example data
- `src/samples.py:12-29` — `SAMPLE_VIDEOS`: three 3Blue1Brown explainers (8m / 27m / 26m) + Karpathy's 1-hour "Intro to LLMs". `SAMPLE_IDS` is a frozenset; `is_sample()` protects them — `DELETE /api/videos/{id}` returns 403 for them (`src/api/videos.py:196-198`).
- `src/seeding.py:50-96` — `seed_to_completion()`: 3 retry passes, idempotent (skips anything already `indexed`), waits for the CLIP service `/healthz` up to 600 s (`seeding.py:22-38`), and **mutates config in-process** to lighter sampling for the demo (`MAX_FRAMES ≤ 60`, `FRAME_INTERVAL_SEC ≥ 5.0` — `seeding.py:63-64`); user uploads run in separate Prefect subprocesses so they keep full quality. Returns bool → `src/seed.py:16-18` maps to exit 0/1.
- `examples/quickstart.py` — the manual, **worker-free** route: runs `ingest_video()` in-process (Prefect executes locally when `PREFECT_API_URL` is unset), sets demo defaults `FRAME_INTERVAL_SEC=4`, `MAX_FRAMES=40` (`quickstart.py:42-43`), then runs 5 canned visual queries (`quickstart.py:56-62`). Flags: `--skip-ingest`, `--ask "<question>"`.
- `examples/README.md` documents the corpus table and prerequisites (Qdrant container, ffmpeg, deps, `DATABASE_URL`).
- There are **no test files and no test framework** anywhere in the repo.
