"""Tests for src/ingest/documents.py.

Fixture PDFs are generated with fitz at test time — no binary files in the repo,
and the expected text is written right next to the assertion that checks it.
"""
from __future__ import annotations

import struct

import fitz
import pytest

from src.ingest.documents import (
    DocChunk,
    DocPage,
    chunk_deck,
    chunk_paper,
    extract_page_texts,
    parse_document,
    render_pages,
)

A4 = fitz.paper_rect("a4")

PAGE_1 = "Attention Is All You Need"
PAGE_2 = "Recurrent models preclude parallelization"
PAGE_4 = "The Transformer allows significantly more parallelization"


def _make_pdf(path, page_texts: list[str], rect=A4) -> str:
    """Write a PDF with one page per entry; "" makes a blank page."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=rect.width, height=rect.height)
        if text:
            m = min(50, rect.width / 10)  # margin, scaled down for tiny pages
            page.insert_textbox(fitz.Rect(m, m, rect.width - m, rect.height - m),
                                text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) parsed from the JPEG SOF marker."""
    i = 2
    while i < len(data):
        assert data[i] == 0xFF, "not at a JPEG marker"
        marker = data[i + 1]
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):  # SOF0/1/2/3 carry the dimensions
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + seg_len
    raise AssertionError("no SOF marker in JPEG")


@pytest.fixture
def paper_pdf(tmp_path):
    """4 pages: text, text, blank, text-heavy."""
    heavy = " ".join(f"{PAGE_4} sentence{i}." for i in range(30))
    return _make_pdf(tmp_path / "paper.pdf", [PAGE_1, PAGE_2, "", heavy])


@pytest.fixture
def deck_pdf(tmp_path):
    """3 slides, the middle one blank (an image-only slide)."""
    return _make_pdf(tmp_path / "deck.pdf", ["Slide one agenda", "", "Slide three results"])


# --- render_pages -----------------------------------------------------------

def test_render_pages_returns_one_page_each(paper_pdf):
    pages = render_pages(paper_pdf)
    assert len(pages) == 4
    assert [p.index for p in pages] == [0, 1, 2, 3]
    assert all(isinstance(p, DocPage) for p in pages)


def test_render_pages_emits_jpeg_bytes(paper_pdf):
    for page in render_pages(paper_pdf):
        assert page.jpeg.startswith(b"\xff\xd8")
        assert page.jpeg.endswith(b"\xff\xd9")


def test_render_pages_scales_to_target_width(paper_pdf):
    pages = render_pages(paper_pdf, width=480)
    width, height = _jpeg_size(pages[0].jpeg)
    assert width == 480
    # A4 is taller than wide; aspect ratio must survive the downscale.
    assert height == pytest.approx(480 * A4.height / A4.width, abs=2)


def test_render_pages_honours_custom_width(paper_pdf):
    assert _jpeg_size(render_pages(paper_pdf, width=240)[0].jpeg)[0] == 240


def test_render_pages_never_upscales_beyond_2x(tmp_path):
    tiny = fitz.Rect(0, 0, 100, 100)
    pdf = _make_pdf(tmp_path / "tiny.pdf", ["small page"], rect=tiny)
    width, _ = _jpeg_size(render_pages(pdf, width=480)[0].jpeg)
    assert width == 200  # 2x cap, not the requested 480


def test_render_pages_rejects_oversized_pdf(tmp_path, monkeypatch):
    from src.ingest import documents

    monkeypatch.setattr(documents, "MAX_PAGES", 2)
    pdf = _make_pdf(tmp_path / "long.pdf", ["a", "b", "c"])
    with pytest.raises(ValueError, match="max is 2"):
        documents.render_pages(pdf)


# --- extract_page_texts -----------------------------------------------------

def test_extract_page_texts_maps_text_to_the_right_page(paper_pdf):
    texts = extract_page_texts(paper_pdf)
    assert len(texts) == 4
    assert PAGE_1 in texts[0]
    assert PAGE_2 in texts[1]
    assert PAGE_1 not in texts[1]
    assert PAGE_4 in texts[3]


def test_extract_page_texts_blank_page_is_empty_string(paper_pdf):
    assert extract_page_texts(paper_pdf)[2] == ""


def test_extract_page_texts_normalizes_whitespace(paper_pdf):
    text = extract_page_texts(paper_pdf)[0]
    assert "\n" not in text
    assert "  " not in text
    assert text == text.strip()


# --- chunk_paper ------------------------------------------------------------

def _pages(counts: list[int]) -> list[str]:
    """Page texts of the given word counts, each word globally unique."""
    out, n = [], 0
    for count in counts:
        out.append(" ".join(f"w{n + i}" for i in range(count)))
        n += count
    return out


def test_chunk_paper_empty_input_yields_nothing():
    assert chunk_paper([]) == []
    assert chunk_paper(["", "   ", "\n\t "]) == []


def test_chunk_paper_short_document_is_one_chunk():
    chunks = chunk_paper(_pages([10, 10]), target_words=300, overlap_words=45)
    assert len(chunks) == 1
    assert isinstance(chunks[0], DocChunk)
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    assert len(chunks[0].text.split()) == 20


def test_chunk_paper_respects_target_size():
    chunks = chunk_paper(_pages([500, 500]), target_words=100, overlap_words=15)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 100 for c in chunks)
    assert all(len(c.text.split()) > 15 for c in chunks[:-1])


def test_chunk_paper_page_spans_are_one_based_and_ordered():
    chunks = chunk_paper(_pages([120, 120, 120]), target_words=100, overlap_words=15)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    assert all(c.page_start >= 1 and c.page_end >= c.page_start for c in chunks)
    starts = [c.page_start for c in chunks]
    ends = [c.page_end for c in chunks]
    assert starts == sorted(starts)
    assert ends == sorted(ends)


def test_chunk_paper_spans_pages_when_text_crosses_a_break():
    # 60 words on page 1 + 60 on page 2, chunked at 100 -> chunk 1 covers both.
    chunks = chunk_paper(_pages([60, 60]), target_words=100, overlap_words=10)
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_chunk_paper_consecutive_chunks_share_overlap():
    chunks = chunk_paper(_pages([400]), target_words=100, overlap_words=15)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.text.split()[-15:] == nxt.text.split()[:15]


def test_chunk_paper_skips_blank_pages_but_keeps_numbering():
    chunks = chunk_paper(["", "alpha beta", ""], target_words=300)
    assert len(chunks) == 1
    assert chunks[0].text == "alpha beta"
    assert (chunks[0].page_start, chunks[0].page_end) == (2, 2)


def test_chunk_paper_from_a_real_pdf(paper_pdf):
    chunks = chunk_paper(extract_page_texts(paper_pdf), target_words=40, overlap_words=6)
    assert chunks
    assert PAGE_1 in chunks[0].text
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 4  # blank page 3 never breaks the numbering


# --- chunk_deck -------------------------------------------------------------

def test_chunk_deck_one_chunk_per_non_empty_slide(deck_pdf):
    chunks = chunk_deck(extract_page_texts(deck_pdf))
    assert len(chunks) == 2
    assert "agenda" in chunks[0].text
    assert "results" in chunks[1].text


def test_chunk_deck_locators_are_the_slide_number(deck_pdf):
    chunks = chunk_deck(extract_page_texts(deck_pdf))
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert (chunks[1].page_start, chunks[1].page_end) == (3, 3)  # slide 2 was blank


def test_chunk_deck_skips_whitespace_only_slides():
    assert chunk_deck(["", "  \n ", "\t"]) == []
    assert len(chunk_deck(["one", "", "two", "three"])) == 3


def test_chunk_deck_never_splits_a_long_slide():
    long_slide = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_deck([long_slide])
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == 1000


# --- parse_document ---------------------------------------------------------

def test_parse_document_paper_dispatch(paper_pdf):
    pages, chunks = parse_document(paper_pdf, "paper")
    assert len(pages) == 4
    # The paper path merges pages, so it produces fewer chunks than pages here.
    assert len(chunks) == 1
    assert chunks[0].page_start == 1 and chunks[0].page_end == 4


def test_parse_document_deck_dispatch(deck_pdf):
    pages, chunks = parse_document(deck_pdf, "deck")
    assert len(pages) == 3           # blank slide is still rendered
    assert len(chunks) == 2          # ...but produces no text chunk
    assert [c.page_start for c in chunks] == [1, 3]


def test_parse_document_rejects_unknown_kind(paper_pdf):
    with pytest.raises(ValueError, match="unknown document kind"):
        parse_document(paper_pdf, "slides")
