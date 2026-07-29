"""Tests for the per-IP demo budget on /api/ask.

Same rules as the rest of the suite: unit-level, no network, no Postgres. The
clock is injected (`now=`) rather than slept through, so the 45-minute and
24-hour boundaries are tested in microseconds and the results don't depend on
how fast the machine is.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from src import config, ratelimit
from src.api import search as search_api


HOUR = 3600.0


def budget(max_calls=3, window_s=2700, reset_s=86400):
    return ratelimit.DemoBudget(max_calls=max_calls, window_s=window_s,
                                reset_s=reset_s)


# ── The shipped defaults ──────────────────────────────────────────────────────

def test_shipped_window_is_45_minutes():
    """Widened from 15m after a live test: a visitor who reads a cited answer and
    follows a citation before asking again was returning to a closed window."""
    assert config.DEMO_BUDGET_WINDOW_S == 45 * 60
    assert config.DEMO_BUDGET_MAX == 10
    assert config.DEMO_BUDGET_RESET_S == 24 * 3600


def test_refusal_message_states_the_real_window():
    """The 429 text is derived from the configured window, so it can't drift out
    of sync with it — the bug that told a user '15m' would be caught here."""
    b = ratelimit.DemoBudget()          # shipped defaults
    for _ in range(b.max_calls + 1):
        d = b.check("ip", now=0.0)
    assert not d.allowed
    msg = b.message(d)
    assert "10 answers per 45m" in msg
    assert "resets in 24h" in msg


# ── The window ────────────────────────────────────────────────────────────────

def test_first_call_opens_the_window():
    b = budget()
    d = b.check("1.2.3.4", now=0.0)
    assert d.allowed
    assert d.remaining == 2


def test_calls_inside_the_window_are_allowed_up_to_the_cap():
    b = budget(max_calls=3)
    assert [b.check("ip", now=t).allowed for t in (0.0, 1.0, 2.0)] == [True] * 3


def test_cap_hit_refuses_with_a_reset_time():
    b = budget(max_calls=3, reset_s=int(24 * HOUR))
    for t in (0.0, 1.0, 2.0):
        b.check("ip", now=t)
    d = b.check("ip", now=3.0)
    assert not d.allowed
    assert d.reason == "cap_reached"
    # Refusal always carries an expiry — never a permanent lockout.
    assert 0 < d.retry_after_s <= 24 * HOUR
    assert "resets in" in b.message(d)


def test_only_allowed_calls_are_counted():
    """A refusal must not push the reset further out, or a client polling in a
    loop would lock itself out forever."""
    b = budget(max_calls=1, reset_s=100)
    b.check("ip", now=0.0)
    first = b.check("ip", now=10.0).retry_after_s
    second = b.check("ip", now=20.0).retry_after_s
    assert first == 90 and second == 80


def test_window_closing_refuses_even_with_budget_left():
    """Three calls of an allowance of ten, then a long pause: the window is the
    limit, not just the count."""
    b = budget(max_calls=10, window_s=2700)
    b.check("ip", now=0.0)
    d = b.check("ip", now=2701.0)
    assert not d.allowed
    assert d.reason == "window_closed"


def test_reset_forgives_the_ip_and_opens_a_fresh_window():
    b = budget(max_calls=3, window_s=2700, reset_s=int(24 * HOUR))
    for t in (0.0, 1.0, 2.0):
        b.check("ip", now=t)
    assert not b.check("ip", now=HOUR).allowed
    d = b.check("ip", now=24 * HOUR + 1)
    assert d.allowed
    assert d.remaining == 2                      # a whole new allowance
    assert b.check("ip", now=24 * HOUR + 2).allowed


def test_budgets_are_per_ip():
    b = budget(max_calls=1)
    assert b.check("a", now=0.0).allowed
    assert not b.check("a", now=1.0).allowed
    assert b.check("b", now=1.0).allowed         # unaffected by a's spending


def test_old_entries_are_swept_so_the_map_cannot_grow_forever():
    b = budget(reset_s=10)
    for i in range(10_001):
        b.check(f"ip-{i}", now=0.0)
    b.check("late", now=100.0)                   # trips the sweep
    assert len(b._state) == 1


# ── Whose IP is it? ───────────────────────────────────────────────────────────

def test_untrusted_deployment_ignores_proxy_headers(monkeypatch):
    """Off Fly, the socket peer is the only truth — otherwise anyone could mint
    a fresh budget per request by inventing a header."""
    monkeypatch.setattr(config, "TRUST_PROXY_IP", False)
    hdrs = {"fly-client-ip": "9.9.9.9", "x-forwarded-for": "8.8.8.8"}
    assert ratelimit.client_ip(hdrs, "10.0.0.1") == "10.0.0.1"


def test_x_forwarded_for_is_never_trusted(monkeypatch):
    """Even behind the proxy: X-Forwarded-For is client-appendable, so it is
    not consulted at all. Fly-Client-IP is overwritten by the edge, so it is."""
    monkeypatch.setattr(config, "TRUST_PROXY_IP", True)
    hdrs = {"x-forwarded-for": "8.8.8.8"}
    assert ratelimit.client_ip(hdrs, "10.0.0.1") == "10.0.0.1"
    hdrs["fly-client-ip"] = "9.9.9.9"
    assert ratelimit.client_ip(hdrs, "10.0.0.1") == "9.9.9.9"


def test_spoofed_header_cannot_buy_extra_budget(monkeypatch):
    """The attack this is really about: one caller, ten forged identities."""
    monkeypatch.setattr(config, "TRUST_PROXY_IP", False)
    b = budget(max_calls=1)
    peer = "10.0.0.1"
    assert b.check(ratelimit.client_ip({"fly-client-ip": "1.1.1.1"}, peer), now=0.0).allowed
    d = b.check(ratelimit.client_ip({"fly-client-ip": "2.2.2.2"}, peer), now=1.0)
    assert not d.allowed


def test_missing_peer_still_yields_a_key(monkeypatch):
    monkeypatch.setattr(config, "TRUST_PROXY_IP", False)
    assert ratelimit.client_ip({}, None) == "unknown"


# ── Admin bypass ──────────────────────────────────────────────────────────────

def test_admin_token_is_recognised(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "s3cret")
    assert ratelimit.is_admin("Bearer s3cret")


def test_wrong_or_missing_token_is_not_admin(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "s3cret")
    assert not ratelimit.is_admin("Bearer nope")
    assert not ratelimit.is_admin("s3cret")      # missing the scheme
    assert not ratelimit.is_admin(None)


def test_unset_admin_token_grants_nobody_admin(monkeypatch):
    """A dev box with no ADMIN_TOKEN must not turn every anonymous caller into
    an admin — that would silently disable the limiter in the one place it is
    easiest to misconfigure."""
    monkeypatch.setattr(config, "ADMIN_TOKEN", "")
    assert not ratelimit.is_admin(None)
    assert not ratelimit.is_admin("Bearer ")


# ── The route ─────────────────────────────────────────────────────────────────

class FakeRequest:
    """Just the two attributes the handler touches."""
    def __init__(self, peer="10.0.0.1", headers=None):
        self.headers = headers or {}
        self.client = type("C", (), {"host": peer})()


@pytest.fixture
def route(monkeypatch):
    """/api/ask with the limiter on, the retrieval+LLM path stubbed out."""
    monkeypatch.setattr(config, "DEMO_BUDGET_ENABLED", True)
    monkeypatch.setattr(config, "TRUST_PROXY_IP", False)
    monkeypatch.setattr(config, "ADMIN_TOKEN", "s3cret")
    monkeypatch.setattr(config, "LOCK_TENANT", True)
    monkeypatch.setattr(ratelimit, "budget", budget(max_calls=2))
    monkeypatch.setattr(search_api.rag_search, "ask",
                        lambda *a, **k: {"answer": "ok", "citations": []})
    return search_api


def call(route, *, peer="10.0.0.1", auth=None, question="what is attention?"):
    return route.ask(route.AskRequest(question=question), FakeRequest(peer),
                     x_user_id=None, authorization=auth)


def test_route_allows_up_to_the_cap_then_429s(route):
    assert call(route)["answer"] == "ok"
    assert call(route)["answer"] == "ok"
    with pytest.raises(HTTPException) as e:
        call(route)
    assert e.value.status_code == 429
    assert "resets in" in e.value.detail
    assert e.value.headers["Retry-After"]


def test_route_admin_bearer_bypasses_the_budget(route):
    for _ in range(5):
        assert call(route, auth="Bearer s3cret")["answer"] == "ok"
    assert call(route)["answer"] == "ok"          # the IP never spent anything


def test_route_empty_question_does_not_spend_budget(route):
    for _ in range(3):
        with pytest.raises(HTTPException) as e:
            call(route, question="   ")
        assert e.value.status_code == 400
    assert call(route)["answer"] == "ok"          # allowance still intact


def test_route_is_a_no_op_when_the_budget_is_disabled(route, monkeypatch):
    monkeypatch.setattr(config, "DEMO_BUDGET_ENABLED", False)
    for _ in range(10):
        assert call(route)["answer"] == "ok"
