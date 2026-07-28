"""Tests for the document branch's seams: registration, queue routing, fusion
and citation formatting.

Everything here is unit-level — no Postgres, no Qdrant, no Prefect Cloud, no
network. The three collaborators (db, vector_store, the embedders) are stubbed
at the module boundary, which is exactly where the real code injects them, so
these run from a clean checkout with zero env vars set.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from src import config, jobs
from src.api import videos as videos_api
from src.rag import search as rag_search


# ── Registration: which corpus is this, and did the client say so? ────────────

def test_presign_pdf_mints_a_doc_id():
    req = videos_api.PresignRequest(filename="attention.pdf",
                                    content_type="application/pdf", size=1_000_000)
    out = videos_api.presign(req, uid="tester")
    assert out["video_id"].startswith("doc_")
    assert out["key"] == f"uploads/tester/{out['video_id']}.pdf"


def test_presign_video_still_mints_an_up_id():
    req = videos_api.PresignRequest(filename="talk.mp4", content_type="video/mp4",
                                    size=1_000_000)
    out = videos_api.presign(req, uid="tester")
    assert out["video_id"].startswith("up_")
    assert out["key"].endswith(".mp4")


def test_presign_rejects_other_types():
    req = videos_api.PresignRequest(filename="notes.txt", content_type="text/plain",
                                    size=10)
    with pytest.raises(HTTPException) as e:
        videos_api.presign(req, uid="tester")
    assert e.value.status_code == 415


@pytest.mark.parametrize("kind", ["paper", "deck"])
def test_doc_source_is_the_kind(kind):
    assert videos_api.upload_source("doc_abc1234567", kind) == kind


@pytest.mark.parametrize("kind", [None, "", "slides", "PAPER"])
def test_doc_without_a_valid_kind_is_rejected(kind):
    with pytest.raises(HTTPException) as e:
        videos_api.upload_source("doc_abc1234567", kind)
    assert e.value.status_code == 400


def test_video_upload_needs_no_kind():
    assert videos_api.upload_source("up_abc1234567", None) == "upload"


def test_kind_on_a_video_is_rejected():
    with pytest.raises(HTTPException) as e:
        videos_api.upload_source("up_abc1234567", "paper")
    assert e.value.status_code == 400


def test_register_stores_the_kind_as_the_source(monkeypatch):
    """The manifest row for a PDF carries source='deck' — free-text column, so
    the whole document branch needs no migration."""
    saved: dict = {}

    def fake_upsert(row):
        saved.update(row)
        return {**row, "status": "pending"}

    monkeypatch.setattr(videos_api.db, "upsert_pending", fake_upsert)
    monkeypatch.setattr(videos_api.storage, "head", lambda key: {"size": 1234})
    monkeypatch.setattr(videos_api.config, "ENABLE_FAIR_DISPATCH", True)

    out = videos_api.register(videos_api.RegisterRequest(
        video_id="doc_abc1234567", key="uploads/tester/doc_abc1234567.pdf",
        kind="deck", title="Q3 deck"), uid="tester")

    assert saved["source"] == "deck"
    assert out == {"video_id": "doc_abc1234567", "status": "pending"}


# ── Queue routing: the id prefix picks the deployment ─────────────────────────

def test_enqueue_routes_by_id_prefix():
    assert jobs.deployment_for("doc_abc1234567") == "ms-ingest-doc/ingest-doc"
    assert jobs.deployment_for("up_abc1234567") == "ms-ingest-video/ingest"
    assert jobs.deployment_for("yt_LPZh9BOjkQs") == "ms-ingest-video/ingest"


def test_doc_pipeline_imports_without_env():
    """The flow module must import clean — the worker serves it at boot."""
    from src.ingest import doc_pipeline

    assert doc_pipeline.ingest_doc.name == "ms-ingest-doc"


# ── Fusion: a page is a moment, a time window is a moment ─────────────────────

def _page_hit(video_id, page, **extra):
    return {"user_id": "u", "video_id": video_id, "modality": "page",
            "kind": "paper", "page": page, "score": 0.3, **extra}


def _chunk_hit(video_id, page_start, page_end=None, text="…"):
    return {"user_id": "u", "video_id": video_id, "modality": "doc_text",
            "kind": "paper", "page_start": page_start,
            "page_end": page_end or page_start, "text": text, "score": 0.6}


def _frame_hit(video_id, ms, idx):
    return {"user_id": "u", "video_id": video_id, "modality": "frame",
            "ms": ms, "idx": idx, "t_start": ms / 1000.0, "score": 0.3}


def _text_hit(video_id, t_start, text="…"):
    return {"user_id": "u", "video_id": video_id, "modality": "text",
            "t_start": t_start, "ms": int(t_start * 1000), "text": text,
            "score": 0.6}


def test_doc_hits_group_by_page_not_time():
    """Every doc hit has t=0 (no time axis) — grouping must NOT collapse them."""
    windows = rag_search._fuse(
        [_page_hit("doc_a", 1), _page_hit("doc_a", 7), _page_hit("doc_a", 12)], [])
    assert sorted(w["page"] for w in windows) == [1, 7, 12]


def test_page_image_and_page_text_fuse_into_one_moment():
    windows = rag_search._fuse([_page_hit("doc_a", 4)], [_chunk_hit("doc_a", 4, 5)])
    assert len(windows) == 1
    w = windows[0]
    assert w["page"] == 4 and w["modalities"] == {"frame", "text"}
    # Both modalities agreeing on one page earns the same boost a video moment
    # gets when a frame and its transcript agree.
    single = rag_search._fuse([_page_hit("doc_a", 4)], [])[0]
    assert w["rrf"] == pytest.approx(2 * single["rrf"] * config.CROSS_MODAL_BOOST)


def test_a_text_chunk_is_grouped_on_the_page_it_starts_on():
    windows = rag_search._fuse([_page_hit("doc_a", 5)], [_chunk_hit("doc_a", 5, 6)])
    assert len(windows) == 1 and windows[0]["page"] == 5


def test_pages_of_different_documents_never_merge():
    windows = rag_search._fuse([_page_hit("doc_a", 3), _page_hit("doc_b", 3)], [])
    assert len(windows) == 2


def test_video_hits_still_group_by_time_window():
    """Two frames 2s apart are one moment; one 60s away is another."""
    windows = rag_search._fuse(
        [_frame_hit("yt_x", 10_000, 5), _frame_hit("yt_x", 12_000, 6),
         _frame_hit("yt_x", 72_000, 36)], [])
    assert len(windows) == 2
    assert all(w["page"] is None for w in windows)


def test_video_and_document_hits_coexist_in_one_result_set():
    windows = rag_search._fuse([_frame_hit("yt_x", 10_000, 5), _page_hit("doc_a", 2)],
                               [_text_hit("yt_x", 10.5), _chunk_hit("doc_a", 2)])
    assert len(windows) == 2
    by_id = {w["video_id"]: w for w in windows}
    assert by_id["yt_x"]["page"] is None and by_id["doc_a"]["page"] == 2
    assert all(w["modalities"] == {"frame", "text"} for w in windows)


# ── Citations: locator-or-nothing, in the words of the corpus ────────────────

def test_page_label_by_kind():
    assert rag_search._page_label("paper", 3) == "p. 3"
    assert rag_search._page_label("deck", 3) == "slide 3"
    assert rag_search._page_label("upload", 3) == "p. 3"  # unknown kind -> pages


@pytest.fixture
def retrieval(monkeypatch):
    """retrieve() with its three collaborators stubbed: the CLIP/bge embedders,
    Qdrant, and the Postgres metadata join. Returns a callable taking the hits
    each branch should 'find' plus the manifest rows they belong to."""
    def run(visual_hits, text_hits, videos):
        monkeypatch.setattr(rag_search, "embed_text", lambda q: q)
        monkeypatch.setattr(rag_search, "embed_query", lambda q: q)
        monkeypatch.setattr(rag_search.config, "ENABLE_TRANSCRIPT", True)
        monkeypatch.setattr(rag_search.vector_store, "search",
                            lambda *a, **k: visual_hits)
        monkeypatch.setattr(rag_search.vector_store, "search_text",
                            lambda *a, **k: text_hits)
        monkeypatch.setattr(rag_search.db, "videos_by_ids", lambda ids: videos)
        # Local-dev URLs: no bucket, no signing, no network.
        monkeypatch.setattr(rag_search.storage, "presign_capable", lambda: False)
        return rag_search.retrieve("what is attention?", "tester")["citations"]
    return run


def test_paper_citation_is_a_page(retrieval):
    (c,) = retrieval([_page_hit("doc_a", 4)], [],
                     {"doc_a": {"title": "Attention Is All You Need",
                                "source": "paper", "url": None}})
    assert c["timestamp"] == "p. 4"
    assert c["page"] == 4
    assert c["ms"] is None            # no time axis — null-safe, never 0
    assert c["idx"] == 3              # page 4 -> frames/.../000003.jpg
    assert c["thumbnail"] == "/api/frame/doc_a/000003.jpg?u=tester"
    assert c["deeplink"] == c["thumbnail"]   # the page image IS the destination


def test_deck_citation_is_a_slide(retrieval):
    (c,) = retrieval([], [_chunk_hit("doc_b", 7, text="Roadmap")],
                     {"doc_b": {"title": "Q3 deck", "source": "deck", "url": None}})
    assert c["timestamp"] == "slide 7"
    assert c["transcript"] == "Roadmap"
    assert c["thumbnail"] == "/api/frame/doc_b/000006.jpg?u=tester"


def test_video_citation_still_uses_a_timestamp(retrieval):
    (c,) = retrieval([_frame_hit("yt_x", 83_000, 41)], [],
                     {"yt_x": {"title": "A talk", "source": "youtube",
                               "url": "https://youtu.be/x"}})
    assert c["timestamp"] == "01:23"
    assert c["page"] is None
    assert c["ms"] == 83_000
    assert c["deeplink"] == "https://youtu.be/x?t=83"


def test_mixed_corpus_citations_are_labelled_per_source(retrieval):
    cites = retrieval([_frame_hit("yt_x", 83_000, 41), _page_hit("doc_a", 2)], [],
                      {"yt_x": {"title": "A talk", "source": "youtube",
                                "url": "https://youtu.be/x"},
                       "doc_a": {"title": "A paper", "source": "paper", "url": None}})
    labels = {c["video_id"]: c["timestamp"] for c in cites}
    assert labels == {"yt_x": "01:23", "doc_a": "p. 2"}


def test_diversity_cap_demotes_a_flooding_source_without_dropping_it():
    # Source vid_a has 6 strong moments (better branch ranks); doc_b has one
    # weaker page. Uncapped, vid_a would fill the head of the ranking and a
    # top-k trim would never show doc_b.
    flood = [_frame_hit("vid_a", ms, i) for i, ms in
             enumerate(range(0, 6 * 60_000, 60_000))]
    windows = rag_search._fuse(flood + [_page_hit("doc_b", 2)], [])
    head_ids = [w["video_id"] for w in windows[:config.MAX_PER_SOURCE + 1]]
    assert head_ids.count("vid_a") == config.MAX_PER_SOURCE
    assert "doc_b" in head_ids            # surfaces right after the cap
    assert len(windows) == 7              # demoted, never dropped
    assert [w["video_id"] for w in windows[config.MAX_PER_SOURCE + 1:]] \
        == ["vid_a"] * 3


def test_diversity_cap_is_a_noop_for_a_single_source():
    flood = [_frame_hit("vid_a", ms, i) for i, ms in
             enumerate(range(0, 6 * 60_000, 60_000))]
    windows = rag_search._fuse(flood, [])
    assert len(windows) == 6
    rrfs = [w["rrf"] for w in windows]
    assert rrfs == sorted(rrfs, reverse=True)


def test_page_span_chunks_are_cited_with_the_full_range():
    # The citation audit caught quotes on a chunk's LAST page cited to its
    # first. A span gets "pp. 9-10"; a single page stays "p. 9".
    assert rag_search._page_label("paper", 9, 10) == "pp. 9–10"
    assert rag_search._page_label("deck", 3, 5) == "slides 3–5"
    assert rag_search._page_label("paper", 9, 9) == "p. 9"
    assert rag_search._page_label("paper", 9, None) == "p. 9"
    assert rag_search._page_label("deck", 3) == "slide 3"
