"""Per-document ingest pipeline — the PDF sibling of pipeline.py.

pending -> fetching -> sampling -> embedding -> indexed | skipped | failed

Deliberately the SAME status strings as the video flow: for a document
"sampling" means parsing (render every page, chunk the text), so the UI status
chips and the WFQ dispatcher work unchanged. Stages:

  1. fetch    download the uploaded PDF into worker scratch, hash it, skip
              duplicates — the identical contract as the video flow's fetch
  2. parse    documents.parse_document -> page JPEGs + text chunks
  3. embed    page JPEGs through CLIP into the SAME visual collection as video
              frames, text chunks through bge into the SAME text collection —
              one searchable space, so a question fuses video and document hits

A page is to a document what a moment is to a video, so the page images land in
the existing frames/{user}/{id}/NNNNNN.jpg thumbnail layout (0-based index =
page - 1) and citations render as "p. 3" / "slide 3" instead of "01:23".

Point ids come from the same uuid5 scheme as video (vector_store.point_id and
upsert_chunks), so a re-ingest overwrites rather than duplicates.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prefect import flow, task

# Absolute imports on purpose: Prefect loads this file directly from the
# deployment entrypoint (outside the package), where relative imports break.
from src import db, storage
from src.config import CLIP_BATCH, DOC_KINDS, EMBED_VERSION, TEXT_EMBED_VERSION
from src.ingest import documents
from src.ingest import fetch as fetch_mod
from src.rag import vector_store
from src.rag.embeddings import embed_docs, embed_jpegs

_UPLOAD_POOL = 8  # concurrent page-thumbnail PUTs (I/O-bound), as in pipeline.py


@task(name="doc-fetch", retries=2, retry_delay_seconds=[30, 120])
def t_fetch(video_id: str, user_id: str) -> str:
    """Uploaded PDF -> worker scratch file; duplicate check via source_hash.

    Returns "" when this user already indexed the same bytes (row marked
    'skipped' — an outcome, not a retryable error)."""
    db.set_status(video_id, "fetching")
    row = db.get_video(video_id)
    if row is None:
        raise ValueError(f"no manifest row for {video_id}")
    if not row.get("storage_key"):
        raise ValueError(f"{video_id} has no storage_key — nothing to fetch")

    path = fetch_mod.fetch_upload(row["storage_key"], video_id)
    source_hash = fetch_mod.sha256_file(path)
    db.set_status(video_id, "fetching", source_hash=source_hash)

    dup = db.find_duplicate(user_id, source_hash, exclude_id=video_id)
    if dup:
        path.unlink(missing_ok=True)
        db.set_status(video_id, "skipped", error=f"duplicate of {dup['id']}")
        return ""
    return str(path)


@task(name="doc-parse")
def t_parse(video_id: str, kind: str, path: str) -> tuple[list, list]:
    """PDF -> (page JPEGs, text chunks). The document flow's 'sampling' stage."""
    db.set_status(video_id, "sampling", progress=0.0)
    pages, chunks = documents.parse_document(Path(path), kind)
    if not pages:
        raise RuntimeError("No pages could be rendered from the PDF.")
    print(f"[parse] {video_id}: {len(pages)} pages -> {len(chunks)} text chunks ({kind})")
    # Page count is this corpus's frame_count — same column, same UI chip.
    db.set_status(video_id, "sampling", frame_count=len(pages), progress=1.0)
    return pages, chunks


@task(name="doc-embed-index", retries=2, retry_delay_seconds=60)
def t_embed_index(video_id: str, user_id: str, kind: str,
                  pages: list, chunks: list) -> tuple[int, int]:
    """Pages -> CLIP -> visual collection (+ thumbnails); chunks -> bge -> text
    collection. Both deletes run first so a re-ingest replaces, never appends."""
    db.set_status(video_id, "embedding", progress=0.0)
    vector_store.ensure_collection()
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, video_id)          # stale points from a prior run
    storage.delete_prefix(storage.frame_prefix(user_id, video_id))  # …and its thumbnails

    def _put(i_p: tuple[int, documents.DocPage]) -> None:
        i, p = i_p
        storage.put_bytes(storage.frame_key(user_id, video_id, i), p.jpeg, "image/jpeg")

    total = 0
    for start in range(0, len(pages), CLIP_BATCH):
        batch = pages[start:start + CLIP_BATCH]
        vectors = embed_jpegs([p.jpeg for p in batch])
        vector_store.upsert_frames(
            user_id, video_id,
            ids=range(start, start + len(batch)),
            vectors=vectors,
            # No timestamp: `page` (1-based, what a human reads) is the locator,
            # and its presence is what tells fusion to group by page, not time.
            payloads=[{"user_id": user_id, "video_id": video_id,
                       "modality": "page", "kind": kind, "page": p.index + 1,
                       "embed_version": EMBED_VERSION}
                      for p in batch],
        )
        with ThreadPoolExecutor(max_workers=_UPLOAD_POOL) as ex:
            list(ex.map(_put, enumerate(batch, start)))
        total += len(batch)
        db.set_progress(video_id, total / len(pages))

    n_text = 0
    if chunks:
        vecs = embed_docs([c.text for c in chunks])
        vector_store.upsert_chunks(user_id, video_id, vecs, payloads=[
            {"user_id": user_id, "video_id": video_id, "modality": "doc_text",
             "kind": kind, "page_start": c.page_start, "page_end": c.page_end,
             "text": c.text, "embed_version": TEXT_EMBED_VERSION} for c in chunks])
        n_text = len(chunks)

    db.set_status(video_id, "indexed", frame_count=total,
                  embed_version=EMBED_VERSION, progress=1.0)
    return total, n_text


@flow(name="ms-ingest-doc", log_prints=True, timeout_seconds=3600)
def ingest_doc(video_id: str, user_id: str) -> dict:
    attempt = db.bump_attempts(video_id)
    path: str | None = None
    try:
        row = db.get_video(video_id)
        if row is None:
            raise ValueError(f"no manifest row for {video_id}")
        kind = row.get("source")  # the manifest stores the kind as the source
        if kind not in DOC_KINDS:
            raise ValueError(f"{video_id} is not a document (source={kind!r})")
        path = t_fetch(video_id, user_id)
        if not path:  # duplicate — already marked 'skipped' by t_fetch
            print(f"[ingest-doc] {video_id} skipped (duplicate content)")
            return {"video_id": video_id, "skipped": True}
        pages, chunks = t_parse(video_id, kind, path)
        n, t = t_embed_index(video_id, user_id, kind, pages, chunks)
        print(f"[ingest-doc] {video_id} indexed: {n} pages + {t} text chunks "
              f"(attempt {attempt})")
        return {"video_id": video_id, "pages": n, "text_chunks": t}
    except Exception as exc:
        db.set_status(video_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if path:  # scratch only — the durable PDF lives in object storage
            Path(path).unlink(missing_ok=True)
