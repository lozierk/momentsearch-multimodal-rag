"""MomentSearch — unified API (one service, one port).

Two routers on one FastAPI app (:8000):
  - src/api/videos.py  /api/videos/*  — presigned uploads + registration +
                                        ingest status (Bearer auth)
  - src/api/search.py  public         — / (web UI), /api/ask, /api/config,
                                        local-dev media, /api/health

Heavy processing never happens here — the videos router only schedules Prefect
flow runs; worker.py (separate process, same image) executes the ingest
pipeline. Every durable byte lives in object storage, Qdrant, or Postgres, so
this process is stateless and disposable.

Run:
    uvicorn src.app:app --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import config, db, metrics
from .api.search import router as search_router
from .api.videos import router as videos_router
from .rag import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    # Create the Qdrant collection up front (known CLIP dims resolve without
    # loading the model) so a question before the first ingest returns
    # "no moments" instead of a 500. Qdrant being down must not block boot.
    try:
        vector_store.ensure_collection()          # visual (CLIP frames)
        if config.ENABLE_TRANSCRIPT:
            vector_store.ensure_text_collection()  # transcript (bge text)
    except Exception as exc:
        print(f"[startup] Qdrant not ready ({exc!r}) — search degrades to empty results")
    yield


app = FastAPI(title="Moment Search", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def _meter(request, call_next):
    """Time every request for /metrics (src/metrics.py).

    Labels use the ROUTE TEMPLATE that Starlette resolved (`/api/videos/{video_id}`),
    never the raw path — a label per document id would both explode cardinality
    and put ids on a public page. Unmatched paths collapse to one bucket.
    """
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        label = getattr(route, "path", None) or "unmatched"
        metrics.metrics.record_request(label, status,
                                       (time.perf_counter() - start) * 1000)


app.include_router(videos_router)
app.include_router(search_router)
