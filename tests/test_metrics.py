"""Tests for the metrics counters, the cost math, and the privacy boundary.

Unit-level like the rest of the suite: no Postgres, no network. The one test
that touches the corpus view stubs `db` at the module boundary.
"""
from __future__ import annotations

import pytest

import src
from src import config, metrics as m


@pytest.fixture
def met():
    """A fresh Metrics instance — the module-level one is process-wide."""
    return m.Metrics()


# ── Cost math ─────────────────────────────────────────────────────────────────

def test_opus_5_priced_at_5_and_25_per_mtok():
    """1M in + 1M out on Claude Opus 5 = $5.00 + $25.00."""
    assert m.cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.00)


def test_cost_scales_linearly_below_a_million():
    # A realistic answer: ~12k in (six page images + text), ~700 out.
    got = m.cost_usd("claude-opus-5", 12_000, 700)
    assert got == pytest.approx(12_000 / 1e6 * 5.0 + 700 / 1e6 * 25.0)
    assert got == pytest.approx(0.0775)


def test_input_and_output_are_not_priced_the_same():
    """Guards against a transposed rate — output is 5x input on Opus 5."""
    assert m.cost_usd("claude-opus-5", 1_000_000, 0) == pytest.approx(5.0)
    assert m.cost_usd("claude-opus-5", 0, 1_000_000) == pytest.approx(25.0)


def test_zero_tokens_costs_nothing():
    assert m.cost_usd("claude-opus-5", 0, 0) == 0.0


def test_unknown_model_is_none_not_zero():
    """The important case: an unpriced model must not report $0.00, which would
    silently understate the bill."""
    assert m.cost_usd("some-future-model", 1_000_000, 1_000_000) is None


def test_unpriced_calls_are_counted_separately(met):
    met.record_llm("claude-opus-5", 1_000_000, 0)
    met.record_llm("mystery-model", 1_000_000, 0)
    llm = met.snapshot()["llm"]
    assert llm["calls"] == 2
    assert llm["unpriced_calls"] == 1
    assert llm["estimated_cost_usd"] == pytest.approx(5.0)   # only the priced one
    assert llm["input_tokens"] == 2_000_000                  # but tokens count fully


# ── Request counters ──────────────────────────────────────────────────────────

def test_requests_count_per_route(met):
    met.record_request("/api/ask", 200, 10.0)
    met.record_request("/api/ask", 200, 20.0)
    met.record_request("/api/health", 200, 1.0)
    snap = met.snapshot()
    assert snap["requests_total"] == 3
    by_route = {r["route"]: r["count"] for r in snap["routes"]}
    assert by_route == {"/api/ask": 2, "/api/health": 1}


def test_latency_summary(met):
    for v in range(1, 101):                      # 1..100 ms
        met.record_request("/api/ask", 200, float(v))
    lat = met.snapshot()["routes"][0]["latency_ms"]
    assert lat["avg"] == pytest.approx(50.5)
    assert lat["p50"] == 50
    assert lat["p95"] == 95
    assert lat["max"] == 100
    assert lat["samples"] == 100


def test_status_counts_track_both_exact_and_class(met):
    met.record_request("/api/ask", 200, 1.0)
    met.record_request("/api/ask", 429, 1.0)
    met.record_request("/api/ask", 429, 1.0)
    status = met.snapshot()["status"]
    assert status["200"] == 1 and status["429"] == 2
    assert status["2xx"] == 1 and status["4xx"] == 2


def test_latency_ring_is_bounded(met):
    """Percentiles come from a bounded window, so a long-running process can't
    leak memory one request at a time."""
    for i in range(m.LATENCY_WINDOW * 3):
        met.record_request("/api/ask", 200, float(i))
    r = met.snapshot()["routes"][0]
    assert r["count"] == m.LATENCY_WINDOW * 3        # the counter is exact
    assert r["latency_ms"]["samples"] == m.LATENCY_WINDOW   # the window is not


def test_empty_snapshot_is_well_formed(met):
    snap = met.snapshot()
    assert snap["requests_total"] == 0
    assert snap["routes"] == [] and snap["status"] == {}
    assert snap["llm"]["calls"] == 0 and snap["llm"]["estimated_cost_usd"] == 0.0


def test_cost_per_call_is_the_mean(met):
    met.record_llm("claude-opus-5", 1_000_000, 0)   # $5
    met.record_llm("claude-opus-5", 3_000_000, 0)   # $15
    assert met.snapshot()["llm"]["cost_per_call_usd"] == pytest.approx(10.0)


def test_reset_clears_everything(met):
    met.record_request("/api/ask", 200, 5.0)
    met.record_llm("claude-opus-5", 100, 100)
    met.reset()
    assert met.snapshot()["requests_total"] == 0
    assert met.snapshot()["llm"]["calls"] == 0


# ── Corpus view: aggregate only ───────────────────────────────────────────────

def test_corpus_snapshot_aggregates_by_kind_and_status(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_USER_ID", "public")
    seen = {}

    class FakeDB:
        @staticmethod
        def status_counts(user_id):
            seen["user_id"] = user_id
            return [("paper", "indexed", 5), ("paper", "failed", 1),
                    ("youtube", "indexed", 4)]

    monkeypatch.setattr(src, "db", FakeDB)
    out = m.corpus_snapshot()
    assert seen["user_id"] == "public"          # scoped to the served tenant
    assert out["total"] == 10
    assert out["by_kind"]["paper"] == {"indexed": 5, "failed": 1}
    assert out["by_status"]["indexed"] == 9


def test_corpus_snapshot_never_leaks_document_identity(monkeypatch):
    """The metrics page is public. Whatever the query returns, the shape that
    reaches the response has no ids and no titles in it."""
    class FakeDB:
        @staticmethod
        def status_counts(user_id):
            return [("deck", "indexed", 9)]

    monkeypatch.setattr(src, "db", FakeDB)
    text = repr(m.corpus_snapshot())
    for forbidden in ("doc_", "yt_", "up_", "title", "Somml", "Florence"):
        assert forbidden not in text


def test_corpus_snapshot_degrades_instead_of_raising(monkeypatch):
    """A database blip must not 500 the observability page."""
    class BoomDB:
        @staticmethod
        def status_counts(user_id):
            raise RuntimeError("pool exhausted")

    monkeypatch.setattr(src, "db", BoomDB)
    out = m.corpus_snapshot()
    assert out["error"] == "RuntimeError"
    assert out["by_kind"] == {}


# ── Prometheus rendering ──────────────────────────────────────────────────────

def test_prometheus_text_exposes_the_core_series(monkeypatch):
    monkeypatch.setattr(m, "metrics", m.Metrics())
    m.metrics.record_request("/api/ask", 200, 12.5)
    m.metrics.record_llm("claude-opus-5", 1000, 100)

    class FakeDB:
        @staticmethod
        def status_counts(user_id):
            return [("paper", "indexed", 5)]

    monkeypatch.setattr(src, "db", FakeDB)
    text = m.prometheus_text()
    assert 'momentsearch_requests_total{route="/api/ask"} 1' in text
    assert 'momentsearch_responses_total{status="200"} 1' in text
    assert 'momentsearch_llm_tokens_total{direction="output"} 100' in text
    assert 'momentsearch_corpus_items{status="indexed"} 5' in text
    assert text.endswith("\n")


# ── The LLM meter is non-fatal ────────────────────────────────────────────────

def test_meter_swallows_a_broken_usage_object():
    """A metrics bug must never turn a good answer into a 500."""
    from src import llm

    class NoUsage:
        def __getattr__(self, name):
            raise RuntimeError("provider changed its response shape")

    llm._meter("claude-opus-5", NoUsage(), "input_tokens", "output_tokens")


def test_meter_records_missing_usage_as_zero(monkeypatch):
    monkeypatch.setattr(m, "metrics", m.Metrics())
    from src import llm
    llm._meter("claude-opus-5", None, "input_tokens", "output_tokens")
    llm_stats = m.metrics.snapshot()["llm"]
    assert llm_stats["calls"] == 1          # the call happened
    assert llm_stats["input_tokens"] == 0   # but reported no tokens
