"""Prefect Cloud trigger layer — the API schedules runs, workers execute them.

Two flows (the "ms-" prefix keeps them distinct from the digital-twin-akash
flow living in the same Prefect workspace), two deployments, both served by the
one worker process (src/worker.py):

  ms-ingest-video/ingest      videos  (yt_… / up_… ids)
  ms-ingest-doc/ingest-doc    PDFs    (doc_… ids)

The id prefix is the whole routing rule — every caller (the API's register /
retry and the WFQ dispatcher) goes through enqueue_video(), so neither has to
know which corpus a row belongs to. The API never imports a pipeline or its
heavy deps (torch, ffmpeg, PyMuPDF); it just asks Prefect Cloud to schedule a
run and any live worker picks it up. Retries/backoff live on the flows' tasks;
failed runs are visible + retryable in the Prefect Cloud UI.
"""
from __future__ import annotations

from prefect.deployments import run_deployment

from .config import DOC_ID_PREFIX

INGEST_DEPLOYMENT = "ms-ingest-video/ingest"
DOC_INGEST_DEPLOYMENT = "ms-ingest-doc/ingest-doc"


def deployment_for(video_id: str) -> str:
    """Which deployment ingests this id — documents are the doc_ prefix."""
    return DOC_INGEST_DEPLOYMENT if video_id.startswith(DOC_ID_PREFIX) else INGEST_DEPLOYMENT


def enqueue_video(video_id: str, user_id: str) -> str:
    """Schedule the right ingest flow for one item. Returns the flow-run id."""
    flow_run = run_deployment(
        name=deployment_for(video_id),
        parameters={"video_id": video_id, "user_id": user_id},
        timeout=0,  # fire-and-forget: don't block the API waiting for the run
        flow_run_name=f"ingest-{video_id}",
    )
    return str(flow_run.id)
