"""PDF parsing — the document branch's source, mirroring frames + transcript.

A PDF gives us both modalities in one file, so it splits the same way a video
does: every page is rendered to a JPEG (the visual branch, exactly what
frames.sample() produces for video) and its text is pulled out and chunked (the
text branch, exactly what transcript.chunk_cues() produces). The locator is a
page number instead of a timestamp — that is the whole difference.

The two document kinds chunk differently because their layout means different
things. A paper's page break is an accident of typesetting, so chunk_paper()
streams words across pages into ~target_words passages and records the page
span each one covers. A deck's page break is the author's own unit of thought
— one PDF page IS one slide — so chunk_deck() never merges or splits pages.

Rendering nothing frame-shaped touches disk here either: JPEG bytes go straight
on to dedup, CLIP and object storage. This module deliberately imports nothing
from the app (no config, no db) so it stays unit-testable on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

# A runaway PDF (a scanned book, a generated report) would render thousands of
# pages before anyone noticed. Refuse early rather than melt the worker.
MAX_PAGES = 500

# Never render a small page at more than 2x its natural size — upscaling adds
# pixels but no detail, and CLIP gains nothing from a blurry enlargement.
_MAX_ZOOM = 2.0

_WS_RE = re.compile(r"\s+")


@dataclass
class DocPage:
    index: int     # 0-based page number, as PyMuPDF counts them
    jpeg: bytes    # downscaled JPEG, ready for CLIP + thumbnail upload


@dataclass
class DocChunk:
    text: str
    page_start: int    # 1-based, inclusive — what a human sees in a PDF reader
    page_end: int      # 1-based, inclusive; equal to page_start for deck slides


def _normalize(text: str) -> str:
    """Collapse PDF whitespace (line breaks, column padding) into single spaces."""
    return _WS_RE.sub(" ", text).strip()


def _open(pdf_path: Path) -> fitz.Document:
    """Open a PDF, refusing pathological page counts."""
    doc = fitz.open(str(pdf_path))
    count = doc.page_count  # read before closing — a closed Document has no attrs
    if count > MAX_PAGES:
        doc.close()
        raise ValueError(f"PDF has {count} pages, max is {MAX_PAGES}")
    return doc


def render_pages(pdf_path: Path, width: int = 480, quality: int = 80) -> list[DocPage]:
    """Render every page to a JPEG scaled to ~width px (aspect ratio preserved)."""
    doc = _open(pdf_path)
    pages: list[DocPage] = []
    try:
        for i, page in enumerate(doc):
            zoom = min(width / page.rect.width, _MAX_ZOOM) if page.rect.width else 1.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pages.append(DocPage(index=i, jpeg=pix.tobytes("jpeg", jpg_quality=quality)))
    finally:
        doc.close()
    return pages


def extract_page_texts(pdf_path: Path) -> list[str]:
    """Plain text per page, whitespace-normalized. Unreadable pages yield ""."""
    doc = _open(pdf_path)
    texts: list[str] = []
    try:
        for page in doc:
            try:
                texts.append(_normalize(page.get_text("text")))
            except Exception:  # scanned/broken page — visual branch still indexes it
                texts.append("")
    finally:
        doc.close()
    return texts


def chunk_paper(page_texts: list[str], target_words: int = 300,
                overlap_words: int = 45) -> list[DocChunk]:
    """Stream words across pages into ~target_words chunks with 15% overlap.

    Each chunk carries the 1-based page span its words came from, so a hit can
    be cited as "pages 3-4" even though the passage ignored the page break.
    """
    # (word, 1-based page) pairs — the page tag is what survives the flattening.
    words: list[tuple[str, int]] = []
    for i, text in enumerate(page_texts):
        for w in _normalize(text).split():
            words.append((w, i + 1))
    if not words:
        return []

    overlap = max(0, min(overlap_words, target_words - 1))
    chunks: list[DocChunk] = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        span = words[start:end]
        chunks.append(DocChunk(
            text=" ".join(w for w, _ in span),
            page_start=span[0][1],
            page_end=span[-1][1],
        ))
        if end >= len(words):  # stop here, else the tail chunk is pure overlap
            break
        start = end - overlap
    return chunks


def chunk_deck(page_texts: list[str]) -> list[DocChunk]:
    """One chunk per slide — a deck PDF page is already the author's own unit.

    Blank slides produce no chunk; render_pages() still makes them findable
    visually, which is how an image-only slide gets retrieved anyway.
    """
    chunks: list[DocChunk] = []
    for i, text in enumerate(page_texts):
        clean = _normalize(text)
        if clean:
            chunks.append(DocChunk(text=clean, page_start=i + 1, page_end=i + 1))
    return chunks


def parse_document(pdf_path: Path, kind: str) -> tuple[list[DocPage], list[DocChunk]]:
    """Parse a PDF into (rendered pages, text chunks). kind is "paper" or "deck"."""
    if kind not in ("paper", "deck"):
        raise ValueError(f"unknown document kind: {kind!r} (expected 'paper' or 'deck')")
    pages = render_pages(pdf_path)
    texts = extract_page_texts(pdf_path)
    chunks = chunk_paper(texts) if kind == "paper" else chunk_deck(texts)
    return pages, chunks
