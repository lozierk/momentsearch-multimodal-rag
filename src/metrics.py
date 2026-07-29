"""In-process metrics: what the deployed app is doing, and what it costs.

The benchmarks in docs/bench/ prove the system was fast once, on a laptop, on a
known corpus. This is the running answer to the same questions — request rate
and latency per route, HTTP status mix, LLM tokens and dollars, and the live
queue depth straight out of Postgres.

Three deliberate limits:

* **In-process.** A restart zeroes every counter, and a second api machine
  would keep its own set. Fine for one demo VM; Prometheus + a real scrape
  target is the answer if these numbers ever have to be authoritative.
* **Bounded memory.** Latencies live in a fixed-size ring per route, so p95 is
  computed over a recent window rather than all history. Percentiles from a
  ring are honest about being recent; an unbounded list is a slow leak.
* **Aggregate only.** Nothing here names a document, a query, or a tenant. The
  page is linked publicly, so it reports shapes and totals — never content.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque

from . import config

# Per-million-token prices, USD. Source: Anthropic model pricing table as of
# 2026-06-24 (Claude Opus 5 $5.00 in / $25.00 out; Sonnet 5 $3.00 / $15.00;
# Haiku 4.5 $1.00 / $5.00). These are list rates and go stale — an unknown
# model is reported with cost null rather than guessed at, so a wrong number
# never quietly reaches the page.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

LATENCY_WINDOW = 512   # samples kept per route


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Dollars for one call, or None when the model isn't in the price table.

    None is a real answer here: reporting $0.00 for an unpriced model would
    understate the bill, which is worse than admitting we don't know."""
    price = PRICING_USD_PER_MTOK.get(model)
    if price is None:
        return None
    in_rate, out_rate = price
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile — no interpolation, so every number reported is
    a latency that actually happened."""
    if not sorted_vals:
        return 0.0
    k = math.ceil(pct / 100 * len(sorted_vals)) - 1
    return sorted_vals[max(0, min(len(sorted_vals) - 1, k))]


class Metrics:
    """Thread-safe counters. One instance per process (see `metrics` below)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._route_count: dict[str, int] = defaultdict(int)
            self._route_latency: dict[str, deque[float]] = defaultdict(
                lambda: deque(maxlen=LATENCY_WINDOW))
            self._status: dict[str, int] = defaultdict(int)
            self._llm_calls = 0
            self._llm_in = 0
            self._llm_out = 0
            self._llm_cost = 0.0
            self._llm_unpriced = 0            # calls we could not price
            self._llm_models: dict[str, int] = defaultdict(int)

    # ── recording ────────────────────────────────────────────────────────────

    def record_request(self, route: str, status: int, ms: float) -> None:
        with self._lock:
            self._route_count[route] += 1
            self._route_latency[route].append(ms)
            self._status[f"{status // 100}xx"] += 1
            self._status[str(status)] += 1

    def record_llm(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """One answer-synthesis call. Called from src/llm.py, which is the only
        place tokens are actually spent."""
        with self._lock:
            self._llm_calls += 1
            self._llm_in += input_tokens
            self._llm_out += output_tokens
            self._llm_models[model] += 1
            c = cost_usd(model, input_tokens, output_tokens)
            if c is None:
                self._llm_unpriced += 1
            else:
                self._llm_cost += c

    # ── reporting ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            routes = []
            for name, count in sorted(self._route_count.items()):
                vals = sorted(self._route_latency[name])
                routes.append({
                    "route": name,
                    "count": count,
                    "latency_ms": {
                        "avg": round(sum(vals) / len(vals), 1) if vals else 0.0,
                        "p50": round(_percentile(vals, 50), 1),
                        "p95": round(_percentile(vals, 95), 1),
                        "max": round(vals[-1], 1) if vals else 0.0,
                        "samples": len(vals),
                    },
                })
            total = sum(self._route_count.values())
            llm = {
                "calls": self._llm_calls,
                "input_tokens": self._llm_in,
                "output_tokens": self._llm_out,
                "estimated_cost_usd": round(self._llm_cost, 4),
                "unpriced_calls": self._llm_unpriced,
                "cost_per_call_usd": (round(self._llm_cost / self._llm_calls, 4)
                                      if self._llm_calls else 0.0),
                "models": dict(self._llm_models),
            }
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "requests_total": total,
                "routes": routes,
                "status": dict(sorted(self._status.items())),
                "llm": llm,
            }


metrics = Metrics()


def corpus_snapshot() -> dict:
    """Live queue + index state from Postgres — counts by kind and status only.

    Scoped to the tenant this deployment serves, and deliberately returns no
    titles or ids: the metrics page is public, and a document name is content.
    Never raises — an observability page that 500s when the database blinks is
    worse than one that says so."""
    try:
        from . import db
        rows = db.status_counts(config.DEFAULT_USER_ID)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}", "by_kind": {}, "by_status": {}}

    by_kind: dict[str, dict[str, int]] = {}
    by_status: dict[str, int] = defaultdict(int)
    total = 0
    for source, status, n in rows:
        by_kind.setdefault(source, {})[status] = n
        by_status[status] += n
        total += n
    inflight = sum(by_status.get(s, 0) for s in config.INFLIGHT_STATUSES)
    return {
        "total": total,
        "by_kind": by_kind,
        "by_status": dict(by_status),
        "inflight": inflight,
        "pending": by_status.get("pending", 0),
    }


def prometheus_text() -> str:
    """The same snapshot in Prometheus exposition format, so this is scrapeable
    without writing a JSON exporter first."""
    s = metrics.snapshot()
    out: list[str] = [
        "# HELP momentsearch_uptime_seconds Process uptime.",
        "# TYPE momentsearch_uptime_seconds gauge",
        f"momentsearch_uptime_seconds {s['uptime_s']}",
        "# HELP momentsearch_requests_total Requests by route.",
        "# TYPE momentsearch_requests_total counter",
    ]
    for r in s["routes"]:
        out.append(f'momentsearch_requests_total{{route="{r["route"]}"}} {r["count"]}')
    out += ["# HELP momentsearch_request_latency_ms Request latency by route.",
            "# TYPE momentsearch_request_latency_ms gauge"]
    for r in s["routes"]:
        for q in ("avg", "p50", "p95"):
            out.append(f'momentsearch_request_latency_ms{{route="{r["route"]}",'
                       f'quantile="{q}"}} {r["latency_ms"][q]}')
    out += ["# HELP momentsearch_responses_total Responses by HTTP status.",
            "# TYPE momentsearch_responses_total counter"]
    for code, n in s["status"].items():
        out.append(f'momentsearch_responses_total{{status="{code}"}} {n}')
    llm = s["llm"]
    out += [
        "# HELP momentsearch_llm_tokens_total Answer-synthesis tokens.",
        "# TYPE momentsearch_llm_tokens_total counter",
        f'momentsearch_llm_tokens_total{{direction="input"}} {llm["input_tokens"]}',
        f'momentsearch_llm_tokens_total{{direction="output"}} {llm["output_tokens"]}',
        "# HELP momentsearch_llm_cost_usd Estimated spend at list prices.",
        "# TYPE momentsearch_llm_cost_usd counter",
        f'momentsearch_llm_cost_usd {llm["estimated_cost_usd"]}',
    ]
    corpus = corpus_snapshot()
    out += ["# HELP momentsearch_corpus_items Manifest rows by status.",
            "# TYPE momentsearch_corpus_items gauge"]
    for status, n in corpus.get("by_status", {}).items():
        out.append(f'momentsearch_corpus_items{{status="{status}"}} {n}')
    return "\n".join(out) + "\n"
